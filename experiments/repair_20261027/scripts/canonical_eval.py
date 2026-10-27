"""Identity-bound paired greedy QA evaluation, physical GPU 0 only."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import string
import sys
import time

if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
    raise RuntimeError('Must explicitly restrict CUDA_VISIBLE_DEVICES=0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import torch.nn.functional as F
from peft import PeftModel

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ['SELECOM_ROOT']).expanduser().resolve()
WORK = Path(os.environ['SELECOM_RUN_DIR']).expanduser().resolve()
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE_DIR / 'code'))
from model.encoder import Encoder
from model.generator import Generator
from model.projector import MLPProjector
from util.llm_utils import get_encode_prompt, get_qa_prompt

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def resource_guard():
    mem = dict((a.rstrip(':'), int(b)) for a,b,*_ in
               (line.split() for line in Path('/proc/meminfo').read_text().splitlines()))
    if mem['MemAvailable'] < 24 * 1024**2:
        raise RuntimeError('Resource guard: system MemAvailable below 24 GiB')
    if torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info(0)
        if free < 8*1024**3:
            raise RuntimeError('Resource guard: GPU 0 free memory below 8 GiB')

def norm(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())

def metrics(pred, golds):
    if pred is None:
        return 0., 0.
    p = norm(pred)
    em = float(any(p == norm(g) for g in golds))
    best = 0.
    for g in golds:
        a,b = p.split(),norm(g).split()
        common = sum((Counter(a)&Counter(b)).values())
        if common:
            best = max(best, 2*common/(len(a)+len(b)))
    return em, best

class Model:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        self.encoder = Encoder(args)
        self.generator = Generator(args)
        self.generator.tokenizer.pad_token = self.generator.tokenizer.eos_token
        self.projector = MLPProjector(
            self.encoder.encoder.config.hidden_size,
            self.generator.generate_model.config.hidden_size, 8, 2).to(torch.bfloat16)
        gen_ckpt = ROOT/'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator'
        self.generator.generate_model = PeftModel.from_pretrained(
            self.generator.generate_model, gen_ckpt, is_trainable=False, torch_device='cpu')
        for module in [self.encoder,self.generator,self.projector]:
            module.requires_grad_(False)
            module.eval()
            module.to('cuda:0')
        self.generation = copy.deepcopy(self.generator.generate_model.generation_config)
        self.generation.do_sample = False
        self.generation.temperature = None
        self.generation.top_p = None
        self.generation.top_k = None
        self.generation.max_new_tokens = args.max_new_tokens
        self.generation.repetition_penalty = 1.05
        self.generation.use_cache = True

    def load_frontend(self, path):
        pairs = [(self.encoder.encoder,'encoder.pt'),
                 (self.encoder.encode_token_embedding_layer,'encode_token_embedding_layer.pt'),
                 (self.projector,'projector.pt'),
                 (self.generator.soft_prompt_start_embedding_layer,'soft_prompt_start_embedding_layer.pt'),
                 (self.generator.soft_prompt_end_embedding_layer,'soft_prompt_end_embedding_layer.pt')]
        for module,name in pairs:
            weights = torch.load(path/name, map_location='cpu', weights_only=True, mmap=True)
            module.load_state_dict(weights, strict=True)
            del weights
        resource_guard()

    @torch.inference_mode()
    def batch(self, rows):
        questions = [x['question'] for x in rows]
        docs = [x['documents'][:1] for x in rows]
        golds = [x['answer'] if isinstance(x['answer'],list) else [x['answer']] for x in rows]
        answers = [g[0] for g in golds]
        enc = get_encode_prompt(self.args.encoder_name,questions,docs,answers,8,
                                tokenizer=self.encoder.tokenizer,max_length=2560)
        ei = self.encoder.tokenizer(enc,padding=True,truncation=True,max_length=2560,return_tensors='pt')
        assert all((row == self.encoder.encode_token_id).sum().item()==8 for row in ei['input_ids'])
        z = self.encoder(ei['input_ids'],ei['attention_mask'])
        z = self.projector(z.reshape(len(rows),8,-1)).reshape(len(rows)*2,-1)
        prompts = get_qa_prompt(self.args.generator_name,questions,docs,answers,2,test=True)
        tok = self.generator.tokenizer
        gi = tok(prompts,padding=True,truncation=True,max_length=1024,return_tensors='pt')
        embeds,mask = self.generator.prepare_input(z,gi['input_ids'],gi['attention_mask'])
        sequences = self.generator.generate_model.generate(inputs_embeds=embeds,attention_mask=mask,
                                                           generation_config=self.generation)
        raw_answers = tok.batch_decode(sequences,skip_special_tokens=True)

        full_prompts = get_qa_prompt(self.args.generator_name,questions,docs,answers,2,test=False)
        full = tok(full_prompts,padding=True,truncation=True,max_length=1024,
                   return_tensors='pt',return_offsets_mapping=True)
        offsets = full.pop('offset_mapping')
        inp,attn = self.generator.prepare_input(z,full['input_ids'],full['attention_mask'])
        logits = self.generator.generate_model(inputs_embeds=inp,attention_mask=attn).logits
        result = []
        for i,(row,raw) in enumerate(zip(rows,raw_answers)):
            match = re.search(r'<answer>(.*?)</answer>',raw,re.S)
            pred = match.group(1).strip() if match else None
            em,f1 = metrics(pred,golds[i])
            start = len(prompts[i])+len('<answer>')
            end = start+len(answers[i].strip('.'))
            positions = [j for j,(a,b) in enumerate(offsets[i].tolist()) if b>start and a<end and j>0]
            if not positions:
                raise RuntimeError('No content token positions; check offsets and prompt boundaries')
            target = full['input_ids'][i,positions].to('cuda:0')
            selected = logits[i,torch.tensor(positions,device='cuda:0')-1].float()
            nll = F.cross_entropy(selected,target,reduction='mean').item()
            result.append({'question':row['question'],'output':pred,'raw_output':raw,
                           'groundtruth':golds[i],'EM':em,'F1':f1,
                           'content_NLL':nll,'answer_tokens':len(positions),'format_valid':bool(match)})
        del logits
        return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--output-name',default='canonical-512-greedy')
    a=p.parse_args()
    a.encoder_name=str(ROOT/'baselineModel/Qwen3-Embedding-0.6B')
    a.generator_name=str(ROOT/'baselineModel/Qwen2.5-7B-Instruct')
    a.encoder_max_length=2560
    a.generator_max_length=1024
    a.num_emb_tokens=8
    a.num_doc_tokens=2
    a.attn_implementation='flash_attention_2'
    resource_guard()
    random.seed(a.seed)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rows=[]
    source_path=ROOT/'data/nq/eval/nq_eval.jsonl'
    with source_path.open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            row=json.loads(line)
            rows.append({'question':row['question'],'answer':row['answer'],
                         'documents':row['documents'][:1]})
    indices=sorted(random.Random(a.seed).sample(range(len(rows)),min(a.samples,len(rows))))
    rows=[rows[i] for i in indices]
    out=WORK/a.output_name
    out.mkdir(exist_ok=True)
    fronts={'stage2':ROOT/'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector',
            'stage1_continue':ROOT/'results/stage1_continue/checkpoint-79'}
    audit=json.loads((WORK/'historical_audit.json').read_text())
    gen_path=ROOT/'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator'
    manifest={'args':vars(a),'selected_indices':indices,'conditions':{k:str(v) for k,v in fronts.items()},
              'checkpoint_audit':audit['checkpoint_comparison'],
              'generator_adapter_sha256':sha(gen_path/'adapter_model.safetensors'),
              'generator_adapter_config_sha256':sha(gen_path/'adapter_config.json'),
              'source_data_sha256':sha(source_path),
              'evaluator_sha256':sha(Path(__file__)),
              'snapshot_code_sha256':{str(x.relative_to(SOURCE_DIR/'code')):sha(x) for x in (SOURCE_DIR/'code').rglob('*.py')},
              'policy':'paired greedy, same batch and prefix, first gold for NLL, all aliases for EM/F1; invalid format scores zero'}
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Existing eval manifest differs; use a new identity-bound directory')
    else:
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if (out/'metrics.json').exists():
        print('ALREADY_COMPLETE',out,flush=True)
        return
    started=time.monotonic()
    model=Model(a)
    summary={}
    for condition,path in fronts.items():
        model.load_frontend(path)
        pred_path=out/(condition+'.jsonl')
        existing=[json.loads(x) for x in pred_path.read_text().splitlines() if x.strip()] if pred_path.exists() else []
        assert all(x['question']==y['question'] for x,y in zip(existing,rows))
        values=existing
        with pred_path.open('a',encoding='utf-8') as f:
            for i in range(len(existing),len(rows),a.batch_size):
                resource_guard()
                batch=model.batch(rows[i:i+a.batch_size])
                for x in batch:
                    f.write(json.dumps(x,ensure_ascii=False)+'\n')
                f.flush()
                values.extend(batch)
                if len(values)%64==0 or len(values)==len(rows):
                    print(json.dumps({'condition':condition,'done':len(values),'total':len(rows),
                                      'seconds':round(time.monotonic()-started,2),
                                      'gpu_allocated_GiB':round(torch.cuda.memory_allocated()/1024**3,2)}),flush=True)
        summary[condition]={'n':len(values),'EM':100*sum(x['EM'] for x in values)/len(values),
                            'F1':100*sum(x['F1'] for x in values)/len(values),
                            'content_NLL':sum(x['content_NLL'] for x in values)/len(values),
                            'format_invalid':sum(not x['format_valid'] for x in values)}
    summary['runtime_seconds']=time.monotonic()-started
    summary['peak_gpu_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
