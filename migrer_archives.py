"""Merge v0.4 data into 0.5; no code or Python environments are copied."""
import argparse,json,shutil,datetime
from pathlib import Path

def main(src,dst):
    src=Path(src).expanduser().resolve();dst=Path(dst).resolve()
    if src==dst:raise SystemExit('Le dossier source et le dossier destination doivent être différents')
    source_db=src/'archive_learning'/'feedback.json';dest_db=dst/'archive_learning'/'feedback.json'
    if not source_db.is_file():raise SystemExit('Dossier v0.4 incorrect : archive_learning/feedback.json introuvable')
    source=json.loads(source_db.read_text(encoding='utf8'))
    dest=json.loads(dest_db.read_text(encoding='utf8')) if dest_db.is_file() else {}
    for key in ['ratings','imports']:
        merged=dict(source.get(key,{}) or {});merged.update(dest.get(key,{}) or {})
        dest[key]=merged
    old=source.get('comparisons',[]);new=dest.get('comparisons',[])
    dest['comparisons']=old+[c for c in new if c not in old]
    dest_db.parent.mkdir(parents=True,exist_ok=True)
    if dest_db.exists():
        backup=dest_db.with_name('feedback_backup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
        shutil.copy2(dest_db,backup)
    tmp=dest_db.with_suffix('.tmp');tmp.write_text(json.dumps(dest,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(dest_db)
    copied=0
    for sub,patterns in [('archive_learning/imports',('*.png',)),('resultats',('session.json','specimen-*.png','temoin-sans-bending.png','input-collision.png','source-*.png','*.png')),('references',('AM-*.png',))]:
        d=src/sub
        if not d.is_dir():continue
        found=set()
        for pat in patterns:found.update(d.rglob(pat))
        for f in found:
            if not f.is_file() or f.is_symlink():continue
            target=dst/sub/f.relative_to(d)
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target);copied+=1
    print(f'Migration effectuée : {copied} fichiers copiés, archives et notes fusionnées.')
    print('Les anciennes données restent dans leur dossier initial.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('--destination',default=str(Path(__file__).resolve().parent))
    a=ap.parse_args();main(a.source,a.destination)
