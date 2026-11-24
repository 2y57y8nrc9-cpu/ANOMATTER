import base64,io,json,tempfile,threading,unittest,urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from PIL import Image
import server
from learning import LearningArchive

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.original=server.ARCHIVE;self.out=server.OUT
        server.ARCHIVE=LearningArchive(self.root);server.OUT=self.root/'resultats'
        (self.root/'references').mkdir()
        Image.new('RGB',(24,24),'blue').save(self.root/'references'/'AM-test.png')
        self.s=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        self.t=threading.Thread(target=self.s.serve_forever,daemon=True);self.t.start()
        self.addr=f'http://127.0.0.1:{self.s.server_port}'
    def tearDown(self):
        self.s.shutdown();self.s.server_close();self.t.join(timeout=4)
        server.ARCHIVE=self.original;server.OUT=self.out;self.temp.cleanup()
    def get(self,route):return json.load(urllib.request.urlopen(self.addr+route))
    def post(self,route,value):
        req=urllib.request.Request(self.addr+route,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req))
    @staticmethod
    def dataurl(color):
        f=io.BytesIO();Image.new('RGB',(80,80),color).save(f,format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(f.getvalue()).decode()
    def test_preview_and_archive_and_comparisons(self):
        self.assertEqual(self.get('/api/health')['version'],'0.5')
        data=self.post('/api/preview',{'sources':{'morphology':self.dataurl('grey'),'chromatic':self.dataurl('blue')}})
        self.assertEqual(len(data['roles']),2)
        self.assertIn('06_final',data['stages'])
        with urllib.request.urlopen(self.addr+data['url']) as r:self.assertEqual(r.headers.get('Content-Type'),'image/png')
        items=self.get('/api/archive')['items'];self.assertEqual(len(items),1)
        ident=items[0]['id']
        imp=self.post('/api/archive/import',{'image':self.dataurl('red'),'name':'rouge'})
        ret=self.post('/api/feedback',{'id':ident,'scores':{'ambiguity':2,'mutation':3,'strangeness':3,'coherence':1,'potential':2},'favorite':True,'tags':['Matière hybride'],'note':'test'})
        self.assertEqual(ret['total'],11)
        comp=self.post('/api/compare',{'left':ident,'right':imp['id'],'winner':ident})
        self.assertEqual(comp['comparisons_count'],1)
        self.assertEqual(self.get('/api/recommend')['comparisons_count'],1)
        self.assertIn('nodes',self.get('/api/lineage'))

if __name__=='__main__':unittest.main()
