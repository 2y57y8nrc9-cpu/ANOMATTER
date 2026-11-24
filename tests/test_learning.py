import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from learning import LearningArchive

class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = LearningArchive(self.root)
        (self.root/'references').mkdir()
        Image.new('RGB', (32,32),'white').save(self.root/'references'/'AM-reftest.png')
        folder = self.root/'resultats'/'a1b2c3d4e5f6'
        folder.mkdir(parents=True)
        Image.new('RGB',(32,32),'black').save(folder/'specimen-01.png')
        (folder/'session.json').write_text(json.dumps({
            'created_at':'2026-11-13T12:30:00',
            'options':{'intensity':.42,'fraction':.12,'strength':.39,
                       'influence':.65,'mode':'ECHO','layer':'early',
                       'prompt':'image as material','seed':12},
            'outputs':[{'url':'/outputs/a1b2c3d4e5f6/specimen-01.png',
                        'name':'AM/abc/01','seed':12}]
        }), encoding='utf8')

    def tearDown(self):
        self.temp.cleanup()

    def test_archive_and_rating_and_recommendation(self):
        entries = self.archive.entries()
        self.assertEqual(len(entries),2)
        self.assertEqual(self.archive.recommend()['parameter_evidence_count'],0)
        ident = next(i['id'] for i in entries if i['kind']=='generated')
        rated = self.archive.save_feedback({
            'id':ident,'scores':{'ambiguity':3,'mutation':3,'strangeness':3,
                                 'coherence':2,'potential':3},
            'tags':['Matière hybride','Motif invasif'],'favorite':True,'note':'Un pli qui prolifère.'})
        self.assertEqual(rated['total'],14)
        self.assertEqual(self.archive.recommend()['parameter_evidence_count'],1)
        self.assertEqual(self.archive.recommend()['variants'][1]['params']['mode'],'ECHO')
        self.assertTrue(LearningArchive(self.root).entries()[0]['feedback']['favorite'])

    def test_import_ref_does_not_invent_generation_params(self):
        imp = self.archive.import_image(Image.new('RGB',(1200,900),'yellow'),'un objet')
        self.archive.save_feedback({'id':imp['id'],'scores':{k:3 for k in ('ambiguity','mutation','strangeness','coherence','potential')},'tags':['Structure impossible'],'favorite':True,'note':''})
        rec = self.archive.recommend()
        self.assertEqual(rec['parameter_evidence_count'],0)
        self.assertIn('structure',rec['variants'][0]['params']['prompt'])

    def test_reject_bad_scores(self):
        ident = next(i['id'] for i in self.archive.entries() if i['kind']=='generated')
        with self.assertRaises(ValueError):
            self.archive.save_feedback({'id':ident,'scores':{'ambiguity':5},'favorite':True})

if __name__=='__main__':unittest.main()
