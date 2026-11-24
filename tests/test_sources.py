import unittest
import numpy as np
from PIL import Image
from source_roles import prepare_sources

class SourceTests(unittest.TestCase):
    def test_palette_from_chromatic_without_subject(self):
        a=Image.new('RGB',(80,80),(100,100,100))
        b=Image.new('RGB',(80,80),(200,30,30))
        output=prepare_sources({'morphology':a,'chromatic':b},{'chromatic_weight':1,'morphology_lock':0})
        v=np.array(output.image).mean(axis=(0,1))
        self.assertGreater(v[0],v[1]+30)
        self.assertEqual(output.roles_used,['morphology','chromatic'])
        self.assertIn('04_chromatic',output.stages)
    def test_distinct_stages(self):
        ims={k:Image.new('RGB',(90,80),c) for k,c in [('morphology','gray'),('material','white'),('pattern','black'),('chromatic','pink'),('parasite','blue')]}
        output=prepare_sources(ims,{},92)
        self.assertEqual(len(output.stages),6)
        self.assertEqual(output.image.size,(512,512))
