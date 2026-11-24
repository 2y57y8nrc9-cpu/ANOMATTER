#!/usr/bin/env python3
"""ANOMATTER 0.5 — single-user Mac-local generative research engine.

The source preparation preview works before neural weights are downloaded.
Diffusers SD1.5 img2img + actual UNet activation hooks performs generation.
Experimental IP-Adapter is opt-in, and can require additional model weights.
No Internet listener: binds to 127.0.0.1 only.
"""
from __future__ import annotations
import io,json,os,random,threading,time,traceback,uuid,webbrowser,zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit
from PIL import Image,ImageOps,ImageFilter
from source_roles import decode_data_url,prepare_sources,palette_swatch,ROLES
from bending import BendSettings,bend_unet,MODES,LAYER_NAMES,REGIONS
from learning import LearningArchive

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'resultats';OUT.mkdir(exist_ok=True)
ARCHIVE=LearningArchive(ROOT)
PORT=int(os.environ.get('ANOMATTER_PORT','8770'))
MODEL=os.environ.get('ANOMATTER_MODEL','stable-diffusion-v1-5/stable-diffusion-v1-5')
PIPE=None;CONTROL_PIPE=None;IP_LOADED=False;PIPE_LOCK=threading.RLock();JOB_LOCK=threading.Lock()
CANCEL=threading.Event()
JOB={'status':'idle','message':'Moteur prêt — l’aperçu des sources fonctionne hors modèle IA',
     'outputs':[],'current':0,'total':0,'input_preview':None,'job_id':None}

class Cancelled(Exception):pass

def set_job(**values):
    with JOB_LOCK:JOB.update(values)

def job_copy():
    with JOB_LOCK:
        return dict(JOB,outputs=[dict(e) for e in JOB.get('outputs',[])])

def options_clamp(obj):
    def num(k,d,lo,hi):
        try:v=float(obj.get(k,d))
        except (TypeError,ValueError):raise ValueError(f'Paramètre {k} non numérique')
        if not (lo<=v<=hi):raise ValueError(f'Paramètre {k} doit être entre {lo} et {hi}')
        return v
    r={}
    r['seed']=int(num('seed',17701234,0,2**31-1));r['count']=int(num('count',4,1,12))
    r['steps']=int(num('steps',26,8,70));r['strength']=num('strength',.44,.20,.95)
    r['intensity']=num('intensity',.36,0,1);r['fraction']=num('fraction',.12,0,.9)
    r['morphology_lock']=num('morphology_lock',.58,0,1)
    for k,d in [('material_weight',.55),('pattern_weight',.5),('chromatic_weight',.85),('parasite_weight',.20)]:r[k]=num(k,d,0,1)
    r['onset']=num('onset',.1,0,1);r['cutoff']=num('cutoff',.95,0,1)
    if r['cutoff']<=r['onset']:raise ValueError('Fin de bending doit suivre le début')
    for k,d in [('early_weight',1),('middle_weight',.6),('late_weight',.45)]:r[k]=num(k,d,0,1)
    r['mode']=str(obj.get('mode','FRACTURE')).upper()
    if r['mode'] not in MODES:raise ValueError('Mode inconnu')
    r['layer']=str(obj.get('layer','multi'))
    if r['layer'] not in LAYER_NAMES:raise ValueError('Couches inconnues')
    r['region']=str(obj.get('region','all'))
    if r['region'] not in REGIONS:raise ValueError('Région inconnue')
    r['prompt']=str(obj.get('prompt') or 'uncertain object, material anomaly, incomplete structural logic, strange documentary photograph, anomalous garment memory')[:900]
    r['negative_prompt']=str(obj.get('negative_prompt') or 'advertisement, logo, text, polished commercial photography')[:500]
    r['compare_baseline']=bool(obj.get('compare_baseline',True))
    r['evolve']=bool(obj.get('evolve',False))
    r['parent_id']=str(obj.get('parent_id') or '')[:110] or None
    r['ip_adapter']=bool(obj.get('ip_adapter',False))
    r['controlnet']=bool(obj.get('controlnet',False))
    r['control_scale']=num('control_scale',.6,0,1.5)
    if r['ip_adapter'] and r['controlnet']:
        raise ValueError('Mode expérimental : sélectionne IP-Adapter OU ControlNet, pas les deux à la fois')
    r['ip_scale']=num('ip_scale',.4,0,1)
    r['ip_role']=str(obj.get('ip_role','material'))
    if r['ip_role'] not in ROLES:raise ValueError('Rôle IP-Adapter inconnu')
    return r


def read_sources(obj):
    incoming=obj.get('sources')
    if not isinstance(incoming,dict):
        old=obj.get('images') or []
        if not isinstance(old,list):raise ValueError('Sources attendues')
        incoming=dict(zip(ROLES,old))
    if not any(incoming.get(r) for r in ROLES):raise ValueError('Ajoute au moins une image')
    return {r:decode_data_url(incoming[r]) for r in ROLES if incoming.get(r)}


def save_preview(prepared,folder:Path,refs:dict):
    folder.mkdir(parents=True,exist_ok=True)
    for k,img in refs.items():
        img.save(folder/f'source-{k}.png')
    for k,img in prepared.stages.items():img.save(folder/f'{k}.png')
    return '/outputs/'+folder.name+'/06_final.png'


def loaded_pipe():
    global PIPE
    with PIPE_LOCK:
        if PIPE is not None:return PIPE
        set_job(message='Chargement du modèle Stable Diffusion 1.5 (première fois : téléchargement de plusieurs Go)')
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        mps=bool(torch.backends.mps.is_available())
        if not mps and os.environ.get('ANOMATTER_ALLOW_CPU')!='1':
            raise RuntimeError('Mac Apple Silicon / Metal requis. Pour tester CPU, régler ANOMATTER_ALLOW_CPU=1 (très lent).')
        kw=dict(torch_dtype=torch.float16 if mps else torch.float32,use_safetensors=True)
        if mps:kw['variant']='fp16'
        pipe=StableDiffusionImg2ImgPipeline.from_pretrained(MODEL,**kw)
        pipe.to('mps' if mps else 'cpu')
        try:pipe.enable_attention_slicing()
        except Exception:pass
        PIPE=pipe
        return PIPE


def control_edges(img):
    """High contrast edge-map fallback; simpler than genuine Canny."""
    gray=ImageOps.grayscale(img).filter(ImageFilter.FIND_EDGES)
    return gray.point(lambda x:255 if x>53 else 0).convert('RGB')


def loaded_controlnet_pipe(base):
    """Optional SD1.5 ControlNet. Shares base UNet components to limit memory."""
    global CONTROL_PIPE
    with PIPE_LOCK:
        if CONTROL_PIPE is not None:return CONTROL_PIPE
        import inspect
        import torch
        from diffusers import ControlNetModel,StableDiffusionControlNetImg2ImgPipeline
        set_job(message='Chargement ControlNet Canny optionnel — poids supplémentaires volumineux…')
        ctrl=ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_canny',
                       torch_dtype=base.unet.dtype,use_safetensors=True)
        valid=inspect.signature(StableDiffusionControlNetImg2ImgPipeline.__init__).parameters
        kwargs={k:v for k,v in base.components.items() if k in valid}
        kwargs['controlnet']=ctrl
        cp=StableDiffusionControlNetImg2ImgPipeline(**kwargs)
        cp.to(base.device)
        try:cp.enable_attention_slicing()
        except Exception:pass
        CONTROL_PIPE=cp
        return CONTROL_PIPE


def ensure_ip_adapter(pipe):
    global IP_LOADED
    if IP_LOADED:return
    set_job(message='Chargement IP-Adapter SD1.5 : poids supplémentaires nécessaires…')
    # Deliberately opt-in for devices with more memory. Stable Diffusion 1.5 only.
    pipe.load_ip_adapter('h94/IP-Adapter',subfolder='models',weight_name='ip-adapter_sd15.bin')
    IP_LOADED=True


def evolution_variant(o,index):
    """Diverse offspring of a real selected image, not a second artificial image generator."""
    r=dict(o)
    if not o['evolve']:return r
    strategies=('FRACTURE','ECHO','DRIFT','PARASITE','GHOST','ABLATE')
    r['mode']=strategies[(strategies.index(o['mode'])+index)%len(strategies)]
    rr=random.Random(o['seed']+index*1949)
    r['intensity']=round(max(.04,min(.9,o['intensity']+rr.uniform(-.16,.18))),3)
    r['fraction']=round(max(.02,min(.6,o['fraction']+rr.uniform(-.055,.085))),3)
    r['onset']=round(max(0,min(.65,o['onset']+rr.uniform(-.08,.18))),3)
    r['region']=REGIONS[index%len(REGIONS)]
    return r


def actual_steps(options):return max(1,round(options['strength']*options['steps']))


def _run_pipeline(pipe,source_img,params,seed,bend,progress_cb=None,ip_img=None,edge_img=None):
    import torch
    def on_step(pipe_instance,step,timestep,kwargs):
        if CANCEL.is_set():raise Cancelled('Génération annulée')
        if progress_cb:progress_cb(step+1)
        return kwargs
    call=dict(prompt=params['prompt'],negative_prompt=params['negative_prompt'],image=source_img,
              strength=params['strength'],num_inference_steps=params['steps'],guidance_scale=5.5,
              generator=torch.Generator(device='cpu').manual_seed(seed),callback_on_step_end=on_step)
    if params['controlnet'] and edge_img is not None:
        call['control_image']=edge_img
        call['controlnet_conditioning_scale']=params['control_scale']
    if params['ip_adapter'] and ip_img is not None:
        pipe.set_ip_adapter_scale(params['ip_scale'])
        call['ip_adapter_image']=ip_img
    elif IP_LOADED:
        # Adapter loaded in the UNet: scale zero for runs without condition.
        pipe.set_ip_adapter_scale(0.0)
    if not bend:
        return pipe(**call).images[0],0
    settings=BendSettings(mode=params['mode'],layer=params['layer'],intensity=params['intensity'],
          fraction=params['fraction'],seed=seed,onset=params['onset'],cutoff=params['cutoff'],
          early_weight=params['early_weight'],middle_weight=params['middle_weight'],late_weight=params['late_weight'],
          region=params['region'],expected_steps=actual_steps(params))
    with bend_unet(pipe.unet,settings) as observers:
        out=pipe(**call).images[0]
    calls=sum(x.calls for x in observers)
    if calls==0:raise RuntimeError('Le modèle n’a appelé aucune couche perturbée')
    return out,calls


def generate_worker(job_id,payload):
    try:
        opts=options_clamp(payload);refs=read_sources(payload)
        prepared=prepare_sources(refs,opts,opts['seed'])
        folder=OUT/job_id;url=save_preview(prepared,folder,refs)
        set_job(input_preview=url,reference_count=len(refs),notes=prepared.notes,
                palette=palette_swatch(refs['chromatic']) if 'chromatic' in refs else [])
        meta=dict(version='0.5',id=job_id,created_at=datetime.now().isoformat(),model=MODEL,
                  engine='SD1.5 Img2Img + dynamic UNet forward hooks; optional IP-Adapter',
                  source_roles=list(refs),options=opts,stages=list(prepared.stages),outputs=[],
                  role_notes=prepared.notes,conditioning='image-space differentiated; neural IP-Adapter only if enabled')
        def save_meta():
            tmp=folder/'session.tmp'
            tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
            os.replace(tmp,folder/'session.json')
        save_meta()
        pipe=loaded_pipe()
        edge_img=None
        if opts['controlnet']:
            edge_img=control_edges(refs.get('morphology') or prepared.image)
            edge_img.save(folder/'structure_edges.png')
            meta['stages'].append('structure_edges')
            save_meta()
            pipe=loaded_controlnet_pipe(pipe)
        ip_img=refs.get(opts['ip_role'])
        if opts['ip_adapter'] and ip_img is None:
            raise ValueError('Active une source dans le rôle choisi pour IP-Adapter')
        if opts['ip_adapter']:ensure_ip_adapter(pipe)
        count=opts['count'];total=count+int(opts['compare_baseline'])
        set_job(total=total,current=0,step=0,step_total=actual_steps(opts))
        if opts['compare_baseline']:
            set_job(message='Témoin neutre : sans perturbations de couches neuronales')
            image,calls=_run_pipeline(pipe,prepared.image,opts,opts['seed'],False,
                  progress_cb=lambda v:set_job(step=v),ip_img=ip_img,edge_img=edge_img)
            image.save(folder/'temoin-sans-bending.png')
            ent=dict(name='AM/'+job_id+'/TÉMOIN',seed=opts['seed'],url=f'/outputs/{job_id}/temoin-sans-bending.png',
                     control=True,mode='TÉMOIN',hook_calls=0,parameters=opts)
            meta['outputs'].append(ent);save_meta()
            with JOB_LOCK:JOB['outputs'].append(ent)
            set_job(current=1)
        for i in range(count):
            if CANCEL.is_set():raise Cancelled('Génération annulée')
            seed=opts['seed']+1943*i;child=evolution_variant(opts,i)
            set_job(message=f'Mutation {i+1}/{count} · {child["mode"]} · {child["region"]}',step=0,
                    step_total=actual_steps(child))
            image,calls=_run_pipeline(pipe,prepared.image,child,seed,True,
                         progress_cb=lambda v:set_job(step=v),ip_img=ip_img,edge_img=edge_img)
            filename=f'specimen-{i+1:02d}.png';image.save(folder/filename)
            entry=dict(name=f'AM/{job_id}/{i+1:02d}',seed=seed,url=f'/outputs/{job_id}/{filename}',
                       control=False,mode=child['mode'],hook_calls=calls,parameters=child)
            meta['outputs'].append(entry);save_meta()
            with JOB_LOCK:JOB['outputs'].append(entry)
            set_job(current=i+1+int(opts['compare_baseline']))
        set_job(status='done',message='Terminé · compare la source préparée, le témoin et les mutants.')
    except Cancelled:
        set_job(status='cancelled',message='Génération annulée. Les images déjà produites ont été conservées.')
    except BaseException as exc:
        traceback.print_exc();set_job(status='error',message=f'{type(exc).__name__} : {exc}')


def get_lineage():
    entries=ARCHIVE.entries();nodes=[];edges=[]
    ids={e['id'] for e in entries}
    for e in entries:
        parent=e.get('parent_id')
        nodes.append({'id':e['id'],'name':e['name'],'url':e['url'],'parent_id':parent,
                      'score':e.get('score'),'kind':e.get('kind')})
        if parent in ids:edges.append({'parent':parent,'child':e['id']})
    return {'nodes':nodes,'edges':edges}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):print('[AM]',fmt%args,flush=True)
    def _json(self,obj,code=200):
        b=json.dumps(obj,ensure_ascii=False).encode('utf-8')
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(b)
    def _input(self,max_size=48_000_000):
        try:n=int(self.headers.get('Content-Length','0'))
        except ValueError:raise ValueError('Taille de requête invalide')
        if not 0<n<=max_size:raise ValueError('Requête vide ou trop volumineuse')
        result=json.loads(self.rfile.read(n))
        if not isinstance(result,dict):raise ValueError('Objet JSON attendu')
        return result
    def do_POST(self):
        path=urlsplit(self.path).path
        try:
            if path=='/api/preview':
                data=self._input();opts=options_clamp(data);refs=read_sources(data)
                prep=prepare_sources(refs,opts,opts['seed'])
                jobid='preview-'+uuid.uuid4().hex[:12];d=OUT/jobid
                url=save_preview(prep,d,refs)
                self._json({'url':url,'roles':prep.roles_used,'notes':prep.notes,'stages':{
                    k:f'/outputs/{jobid}/{k}.png' for k in prep.stages},
                    'palette':palette_swatch(refs['chromatic']) if 'chromatic' in refs else []})
            elif path=='/api/generate':
                data=self._input();opts=options_clamp(data);read_sources(data)
                with JOB_LOCK:
                    if JOB.get('status') in ('running','loading'):
                        self._json({'error':'Une génération est déjà en cours'},409);return
                    jobid=uuid.uuid4().hex[:12]
                    JOB.update(status='running',job_id=jobid,message='Préparation des sources',outputs=[],current=0,
                               total=opts['count']+int(opts['compare_baseline']),input_preview=None)
                    CANCEL.clear()
                threading.Thread(target=generate_worker,args=(jobid,data),daemon=True).start()
                self._json({'job_id':jobid})
            elif path=='/api/cancel':
                CANCEL.set();self._json({'ok':True})
            elif path=='/api/feedback':self._json(ARCHIVE.save_feedback(self._input(12000)))
            elif path=='/api/compare':self._json(ARCHIVE.compare(self._input(12000)))
            elif path=='/api/archive/import':
                data=self._input(13_000_000);pic=decode_data_url(data.get('image'))
                self._json(ARCHIVE.import_image(pic,str(data.get('name','Import'))))
            else:self._json({'error':'Route inconnue'},404)
        except (ValueError,TypeError,json.JSONDecodeError) as e:self._json({'error':str(e)},400)
        except Exception as e:
            traceback.print_exc();self._json({'error':str(e)},500)
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=='/api/health':self._json({'ready':True,'version':'0.5','model':MODEL,'network_bending':'UNet hooks'});return
        if path=='/api/status':self._json(job_copy());return
        if path=='/api/archive':self._json({'items':ARCHIVE.entries()});return
        if path=='/api/recommend':self._json(ARCHIVE.recommend());return
        if path=='/api/lineage':self._json(get_lineage());return
        if path=='/':p=ROOT/'studio.html'
        elif path.startswith('/references/'):
            p=(ROOT/'references'/path[12:]).resolve();base=(ROOT/'references').resolve()
            if not p.is_relative_to(base):self.send_error(403);return
        elif path.startswith('/imported/'):
            p=(ARCHIVE.import_dir/path[10:]).resolve();base=ARCHIVE.import_dir.resolve()
            if not p.is_relative_to(base):self.send_error(403);return
        elif path.startswith('/outputs/'):
            p=(OUT/path[9:]).resolve()
            if not p.is_relative_to(OUT.resolve()):self.send_error(403);return
        else:self.send_error(404);return
        if not p.is_file():self.send_error(404);return
        data=p.read_bytes();typ='text/html; charset=utf-8' if p.suffix=='.html' else 'image/png' if p.suffix=='.png' else 'application/json' if p.suffix=='.json' else 'application/octet-stream'
        self.send_response(200);self.send_header('Content-Type',typ)
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(data)

def main():
    print(f'ANOMATTER v0.5 · http://127.0.0.1:{PORT}',flush=True)
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    threading.Timer(1.1,lambda:webbrowser.open(f'http://127.0.0.1:{PORT}')).start()
    try:server.serve_forever()
    except KeyboardInterrupt:print('Arrêt ANOMATTER')
    finally:server.server_close()

if __name__=='__main__':main()
