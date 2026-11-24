"""ANOMATTER v0.5: explicit-feedback archive, comparisons and parameter search.

No weight updates or silent aesthetic ranking: the artist supplies scores and labels.
Only runs with known generation settings inform parameter suggestions.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

CRITERIA = ('ambiguity', 'mutation', 'strangeness', 'coherence', 'potential')
TAGS = ('Structure impossible', 'Motif invasif', 'Matière hybride', 'Présence fantôme',
        'Rupture spatiale', 'Photographie incertaine', 'Textile', 'Abstraction')
TAG_PROMPTS = {
    'Structure impossible': 'impossible structural continuity, coherent details and incoherent whole',
    'Motif invasif': 'repeating pattern becoming three-dimensional structure',
    'Matière hybride': 'surfaces merging into unclassifiable material',
    'Présence fantôme': 'residual ghost silhouette, ambiguous presence',
    'Rupture spatiale': 'contradictory spatial logic, impossible depth',
    'Photographie incertaine': 'uncertain archival documentary photograph',
    'Textile': 'folds, seams, woven material, garment memory',
    'Abstraction': 'unclassifiable amorphous object',
}
DEFAULT_PROMPT = ('ambiguous unclassifiable object, uncanny materiality, partially recognizable '
                  'garment architecture, fragmented structure, incorrect but coherent geometry, '
                  'unresolved surfaces, archival experimental image')
VALID_MODES = ('FRACTURE', 'DRIFT', 'ECHO', 'GHOST', 'PARASITE', 'ABLATE')
VALID_LAYERS = ('early', 'middle', 'late', 'multi')

class LearningArchive:
    def __init__(self, root):
        self.root = Path(root)
        self.out = self.root / 'resultats'
        self.ref_dir = self.root / 'references'
        self.state_dir = self.root / 'archive_learning'
        self.import_dir = self.state_dir / 'imports'
        self.db_file = self.state_dir / 'feedback.json'
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.import_dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        if not self.db_file.exists():
            self._save({'ratings': {}, 'imports': {}, 'comparisons': []})

    def _load(self):
        try:
            with self.db_file.open('r', encoding='utf-8') as f:
                obj = json.load(f)
            return {'ratings': obj.get('ratings', {}), 'imports': obj.get('imports', {}), 'comparisons': obj.get('comparisons', [])}
        except (OSError, ValueError, TypeError):
            raise RuntimeError('Archives illisibles. Vérifier archive_learning/feedback.json sans le supprimer.')

    def _save(self, state):
        temp = self.db_file.with_suffix('.tmp')
        with temp.open('w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(temp, self.db_file)

    def _generated(self):
        result = []
        if not self.out.exists():
            return result
        for meta_file in sorted(self.out.glob('*/session.json'), key=lambda p:p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(meta_file.read_text(encoding='utf8'))
                jid = meta_file.parent.name
                if not re.fullmatch(r'[a-f0-9]{8,24}', jid):
                    continue
                opts = {k:v for k,v in meta.get('options', {}).items() if k not in ('images','sources')}
                for entry in meta.get('outputs', []):
                    filename = Path(str(entry.get('url', ''))).name
                    if not re.fullmatch(r'(specimen-\d{2}|temoin-sans-bending)\.png', filename):
                        continue
                    if not (meta_file.parent/filename).is_file():
                        continue
                    result.append({
                        'id': f'job:{jid}:{filename}', 'name': entry.get('name','AM / spécimen'),
                        'url': f'/outputs/{jid}/{filename}', 'kind': 'generated',
                        'control': bool(entry.get('control', False)),
                        'options': entry.get('parameters') or opts, 'seed': entry.get('seed', opts.get('seed')),
                        'created_at': meta.get('created_at', ''),
                        'parent_id': opts.get('parent_id'),
                    })
            except (OSError, ValueError, TypeError):
                continue
        return result

    def entries(self):
        with self.lock:
            state = self._load()
            items = self._generated()
            for file in sorted(self.ref_dir.glob('AM-*.png')):
                items.append({'id':f'ref:{file.stem}', 'name':file.stem,
                              'url':'/references/'+file.name, 'kind':'reference',
                              'options': None, 'created_at': ''})
            for name, data in state['imports'].items():
                if (self.import_dir / f'{name}.png').is_file():
                    items.append({'id':f'import:{name}', 'name': data.get('name',name),
                                  'url':f'/imported/{name}.png', 'kind':'reference',
                                  'options':None, 'created_at':data.get('created_at', '')})
            ratings = state['ratings']
            for item in items:
                item['feedback'] = ratings.get(item['id'])
                item['score'] = sum((item['feedback'] or {}).get('scores', {}).values()) if item['feedback'] else None
            items.sort(key=lambda e:(e.get('created_at') or '0'), reverse=True)
            self._add_novelty(items)
            return items

    def save_feedback(self, payload):
        item_id = str(payload.get('id', ''))
        with self.lock:
            items = {item['id']:item for item in self.entries()}
            if item_id not in items:
                raise ValueError('Spécimen introuvable dans l’archive locale')
            scores = payload.get('scores') or {}
            if set(scores) != set(CRITERIA) or any(type(scores.get(k)) is not int or not 0 <= scores[k] <= 3 for k in CRITERIA):
                raise ValueError('Cinq notes entières entre 0 et 3 sont requises')
            tags = payload.get('tags', [])
            if not isinstance(tags, list) or any(tag not in TAGS for tag in tags):
                raise ValueError('Étiquettes inconnues')
            note = payload.get('note','')
            if not isinstance(note, str) or len(note)>1200:
                raise ValueError('Note trop longue')
            feedback = {'scores':scores, 'tags':list(dict.fromkeys(tags))[:8],
                        'favorite':bool(payload.get('favorite', False)),
                        'note':note, 'updated_at':datetime.now().isoformat()}
            state = self._load()
            state['ratings'][item_id] = feedback
            self._save(state)
            return {'id':item_id, 'feedback':feedback, 'total':sum(scores.values())}

    def import_image(self, image, name='Image ajoutée'):
        # The image must be an already parsed Pillow Image from the authenticated local UI.
        with self.lock:
            ident = uuid.uuid4().hex[:12]
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(self.import_dir / f'{ident}.png', optimize=True)
            state = self._load()
            state['imports'][ident] = {
                'name':str(name or 'Import')[:110], 'created_at':datetime.now().isoformat()}
            self._save(state)
            return {'id':f'import:{ident}','url':f'/imported/{ident}.png'}

    @staticmethod
    def _dhash(im):
        import numpy as np
        pixels=np.asarray(im.convert('L').resize((9,8)),dtype=np.uint8)
        bits=(pixels[:,1:]>pixels[:,:-1]).ravel()
        result=0
        for bit in bits:result=(result<<1)|int(bit)
        return result

    def _add_novelty(self,items):
        """Low-level image novelty, NOT semantic judgment or a trained model."""
        prior=[]
        for item in reversed(items[-250:]):
            url=item.get('url','')
            if url.startswith(('/outputs/','/references/')):path=self.root / url.lstrip('/')
            elif url.startswith('/imported/'):path=self.import_dir / Path(url).name
            else:continue
            try:
                with Image.open(path) as im:h=self._dhash(im)
                dist=min([(h^other).bit_count() for other in prior],default=64)
                item['novelty']=round(dist/64,2)
                prior.append(h)
            except (OSError,ValueError):continue

    def compare(self,payload):
        """Save an explicit choice between two archived specimens."""
        left=str(payload.get('left',''));right=str(payload.get('right',''))
        winner=str(payload.get('winner',''))
        with self.lock:
            valid={e['id'] for e in self.entries()}
            if left not in valid or right not in valid or left==right:
                raise ValueError('Sélectionne deux spécimens archivés distincts')
            if winner not in (left,right):raise ValueError('Vainqueur invalide')
            state=self._load()
            state['comparisons'].append({'left':left,'right':right,'winner':winner,
                                         'created_at':datetime.now().isoformat()})
            state['comparisons']=state['comparisons'][-3000:]
            self._save(state)
            return {'ok':True,'comparisons_count':len(state['comparisons'])}

    def recommend(self):
        """Explicit-feedback, rule-based suggestions: never presented as model training."""
        items = self.entries()
        assessed = [it for it in items if it.get('feedback')]
        comparisons=self._load().get('comparisons',[])
        win_count=Counter(e['winner'] for e in comparisons)
        positive = [it for it in assessed if it['score'] >= 9 or it['feedback'].get('favorite')]
        for it in items:
            if win_count[it['id']]>=2 and it not in positive:positive.append(it)
        generated = [it for it in positive if it['kind']=='generated' and not it['control'] and it.get('options')]
        generated.sort(key=lambda e:((e.get('score') or 0) + 2*bool((e.get('feedback') or {}).get('favorite')) + min(6,win_count[e['id']]*2),e.get('created_at','')),reverse=True)
        tag_count = Counter(t for item in positive for t in (item.get('feedback') or {}).get('tags', []))
        common_tags = [t for t,_ in tag_count.most_common(3)]
        prompt_influence = ', '.join(TAG_PROMPTS[t] for t in common_tags)
        base = {'mode':'FRACTURE','layer':'early','intensity':.30,
                'fraction':.10,'strength':.38,'influence':.60,
                'prompt':DEFAULT_PROMPT,'steps':25}
        if generated:
            best = generated[0]
            opts = best['options']
            for key in base:
                if key in opts and (key not in ('mode','layer') or opts[key] in (VALID_MODES if key=='mode' else VALID_LAYERS)):
                    base[key] = opts[key]
        if prompt_influence:
            base['prompt'] = (str(base['prompt']).rstrip(', ') + ', ' + prompt_influence)[:800]
        def clamp(v,lo,hi):return round(max(lo,min(hi,float(v))),3)
        variants=[]
        used_modes=Counter(e['options'].get('mode') for e in generated if e.get('options'))
        unexplored=next((m for m in VALID_MODES if used_modes[m]==0), 'GHOST')
        for title,delta_i,delta_f,delta_s,explanation in [
            ('PRÉSERVER',-.06,-.03,-.07,'Préserver davantage la structure avant de provoquer une nouvelle dérive'),
            ('EXPLORER',.06,.02,.02,'Déplacer légèrement les paramètres autour d’un résultat apprécié'),
            ('DÉRIVER',.13,.06,.09,'Explorer une mutation plus radicale et vérifier les témoins'),
            ('CONTRADICTION',.19,.04,.06,'Essayer un mode peu sélectionné pour préserver la surprise'),
        ]:
            p = dict(base)
            p['intensity']=clamp(float(base['intensity'])+delta_i,0,.95)
            p['fraction']=clamp(float(base['fraction'])+delta_f,.02,.65)
            p['strength']=clamp(float(base['strength'])+delta_s,.20,.85)
            if title=='CONTRADICTION':p['mode']=unexplored
            variants.append({'name':title,'description':explanation,'params':p})
        return {'rated_count':len(assessed),'comparisons_count':len(comparisons),'positive_count':len(positive),
                'parameter_evidence_count':len(generated),'favorite_tags':common_tags,
                'basis':('Variantes heuristiques autour de paramètres réellement notés' if generated
                         else 'Points de départ génériques : aucune génération notée avec ses paramètres'),
                'algorithm':'Recherche heuristique à partir des notes et comparaisons explicites, sans entraînement neuronal',
                'variants':variants}
