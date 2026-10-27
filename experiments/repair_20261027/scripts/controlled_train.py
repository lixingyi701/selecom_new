"""Matched frozen-G0 / frozen-G2 front-end training. Only physical GPU 0."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
    raise RuntimeError('Only physical GPU 0 is authorized')
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import torch
from canonical_eval import Model, ROOT, WORK, SOURCE_DIR, resource_guard, sha
from util.llm_utils import get_encode_prompt, get_select_prompt

def data_batch(model, rows):
    questions=[r['question'] for r in rows]
    docs=[[r['document']] for r in rows]
    answers=[r['answer'] for r in rows]
    enc_prompts=get_encode_prompt(model.args.encoder_name,questions,docs,answers,8)
    enc=model.encoder.tokenizer(enc_prompts,padding=True,truncation=True,max_length=640,return_tensors='pt')
    assert all((r==model.encoder.encode_token_id).sum().item()==8 for r in enc['input_ids'])
    prompts=get_select_prompt(model.args.generator_name,questions,answers,2)
    tok=model.generator.tokenizer
    gen=tok(prompts,padding=True,truncation=True,max_length=256,return_tensors='pt')
    labels=gen['input_ids'].clone()
    role=tok.convert_tokens_to_ids('<|im_start|>')
    for i,ids in enumerate(gen['input_ids']):
        last=(ids==role).nonzero(as_tuple=False)[-1,0].item()
        labels[i,:last+3]=-100
        labels[i,gen['attention_mask'][i]==0]=-100
        assert (labels[i]!=-100).any()
    return enc,gen,labels

def save_frontend(model, path):
    path.mkdir(exist_ok=True)
    modules=[(model.encoder.encoder,'encoder.pt'),
             (model.encoder.encode_token_embedding_layer,'encode_token_embedding_layer.pt'),
             (model.projector,'projector.pt'),
             (model.generator.soft_prompt_start_embedding_layer,'soft_prompt_start_embedding_layer.pt'),
             (model.generator.soft_prompt_end_embedding_layer,'soft_prompt_end_embedding_layer.pt')]
    for module,name in modules:
        weights={k:v.detach().to('cpu',dtype=torch.bfloat16).contiguous().clone()
                 for k,v in module.state_dict().items()}
        temporary=path/(name+'.tmp')
        torch.save(weights,temporary)
        temporary.replace(path/name)
        del weights

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--condition',choices=['G0','G2'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--global-batch',type=int,default=64)
    p.add_argument('--seed',type=int,default=2025)
    p.add_argument('--lr',type=float,default=5e-6)
    p.add_argument('--warmup',type=int,default=50)
    args=p.parse_args()
    assert args.global_batch%args.microbatch==0
    args.encoder_name=str(ROOT/'baselineModel/Qwen3-Embedding-0.6B')
    args.generator_name=str(ROOT/'baselineModel/Qwen2.5-7B-Instruct')
    args.encoder_max_length=640
    args.generator_max_length=256
    args.num_emb_tokens=8
    args.num_doc_tokens=2
    args.max_new_tokens=128
    args.attn_implementation='flash_attention_2'
    out=WORK/args.output
    out.mkdir(exist_ok=True)
    if (out/'train_metrics.json').exists():
        print('ALREADY_COMPLETE',out,flush=True)
        return
    if (out/'train_started.json').exists():
        raise RuntimeError('A previous attempt exists; inspect it before explicitly starting a new attempt')
    resource_guard()
    torch.set_num_threads(2)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path=WORK/'historical-5000-seed2025.jsonl'
    rows=[json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    assert len(rows)==5000
    order=list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    manifest={'args':vars(args),'data_sha256':sha(data_path),'training_order':order,
              'source_sha256':sha(Path(__file__)),
              'model_helper_sha256':sha(SOURCE_DIR/'canonical_eval.py'),
              'protocol':'one epoch; frozen generator EVAL state; full encoder/projector FP32 parameters with BF16 autocast; answer CE; AdamW beta0.9/0.999 eps1e-8 wd0; clip1; warmup50/79; final partial batch averaged; no worker processes',
              'difference_from_history':'single GPU and stable frozen-generator eval mode; FP32 trainable master values; matched mechanism comparison, not exact old-run reproduction'}
    (out/'train_started.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    started=time.monotonic()
    model=Model(args)
    model.load_frontend(ROOT/'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector')
    model.encoder.encoder.float().requires_grad_(True).train()
    model.projector.float().requires_grad_(True).train()
    model.encoder.encode_token_embedding_layer.requires_grad_(False)
    model.generator.requires_grad_(False).eval()
    for m in [model.encoder.encoder,model.generator.generate_model]:
        m.config.use_cache=False
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ep=[p for p in model.encoder.encoder.parameters() if p.requires_grad]
    pp=[p for p in model.projector.parameters() if p.requires_grad]
    trainable=ep+pp
    assert not any(p.requires_grad for p in model.generator.parameters())
    opt=torch.optim.AdamW(trainable,lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    updates=math.ceil(len(rows)/args.global_batch)
    assert updates==79
    losses=[]
    first_gradient={}
    gen_context=model.generator.generate_model.disable_adapter() if args.condition=='G0' else nullcontext()
    with gen_context, (out/'training.jsonl').open('w',encoding='utf-8') as log:
        for step,start in enumerate(range(0,len(order),args.global_batch)):
            resource_guard()
            group=order[start:start+args.global_batch]
            scale=step/args.warmup if step<args.warmup else max(0.,(updates-step)/(updates-args.warmup))
            for pg in opt.param_groups:
                pg['lr']=args.lr*scale
            opt.zero_grad(set_to_none=True)
            total_loss=0.
            for offset in range(0,len(group),args.microbatch):
                resource_guard()
                ids=group[offset:offset+args.microbatch]
                enc,gen,labels=data_batch(model,[rows[i] for i in ids])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z=model.encoder(enc['input_ids'],enc['attention_mask'])
                    z=model.projector(z.reshape(len(ids),8,-1)).reshape(len(ids)*2,-1)
                    loss=model.generator(z,gen['input_ids'],gen['attention_mask'],labels)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                total_loss+=loss.detach().float().item()*len(ids)/len(group)
                (loss*(len(ids)/len(group))).backward()
            grad=torch.nn.utils.clip_grad_norm_(trainable,1.)
            if not torch.isfinite(grad):
                raise RuntimeError('Nonfinite global gradient')
            if step==0:
                first_gradient={name:torch.stack([x.grad.detach().norm() for x in pars if x.grad is not None]).norm().item()
                                for name,pars in [('encoder',ep),('projector',pp)]}
                assert all(v>0 and math.isfinite(v) for v in first_gradient.values())
                assert all(x.grad is None for x in model.generator.parameters())
            opt.step()
            losses.append(total_loss)
            rec={'step':step+1,'updates':updates,'examples_seen':start+len(group),'loss':total_loss,
                 'lr':args.lr*scale,'gradient_norm_before_clip':grad.item(),
                 'elapsed_seconds':time.monotonic()-started,
                 'gpu_allocated_GiB':torch.cuda.memory_allocated()/1024**3}
            log.write(json.dumps(rec)+'\n')
            log.flush()
            if (step+1)%5==0 or step==0 or step+1==updates:
                print(json.dumps(rec),flush=True)
    resource_guard()
    save_frontend(model,out/'checkpoint-final')
    stats={'condition':args.condition,'updates':updates,'examples':len(rows),'mean_training_loss':sum(losses)/len(losses),
           'first_gradient_norms_after_clip':first_gradient,'generator_frozen':True,'generator_eval':True,
           'runtime_seconds':time.monotonic()-started,'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
           'frontend_sha256':{f.name:sha(f) for f in (out/'checkpoint-final').glob('*.pt')}}
    (out/'train_metrics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':
    main()
