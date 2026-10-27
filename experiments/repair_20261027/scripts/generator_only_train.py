"""Third and final planned training: matched G2 LoRA-only update on GPU 0."""
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
    raise RuntimeError('Only physical GPU0 is authorized')
import torch
from canonical_eval import Model, ROOT, WORK, SOURCE_DIR, resource_guard, sha
from controlled_train import data_batch

def main():
    args=SimpleNamespace(seed=2025,encoder_name=str(ROOT/'baselineModel/Qwen3-Embedding-0.6B'),
                         generator_name=str(ROOT/'baselineModel/Qwen2.5-7B-Instruct'),
                         encoder_max_length=640,generator_max_length=256,num_emb_tokens=8,
                         num_doc_tokens=2,max_new_tokens=128,attn_implementation='flash_attention_2')
    out=WORK/'controlled-generator-only'
    out.mkdir(exist_ok=True)
    if (out/'train_metrics.json').exists():
        print('ALREADY_COMPLETE',flush=True)
        return
    if (out/'train_started.json').exists():
        raise RuntimeError('Previous attempt exists; inspect before any retry')
    resource_guard()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    source=WORK/'historical-5000-seed2025.jsonl'
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    assert len(rows)==5000
    order=list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    manifest={'args':vars(args),'source_sha256':sha(Path(__file__)),'data_sha256':sha(source),
              'training_order':order,'helper_sha256':sha(SOURCE_DIR/'controlled_train.py'),
              'updated':'G2 LoRA only; original front end and all base/special embeddings frozen',
              'protocol':'same5000/order/79updates/global64/micro4/5e-6/warmup50/AdamW beta0.9,0.999 eps1e-8 wd0/clip1 as front-end controls; FP32 optimizer weights; BF16 autocast; generator eval to hold dropout fixed'}
    (out/'train_started.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    started=time.monotonic()
    model=Model(args)
    model.load_frontend(ROOT/'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector')
    model.encoder.requires_grad_(False).eval()
    model.projector.requires_grad_(False).eval()
    # Match the front-end controls' forward precision even though this front end is frozen.
    model.encoder.encoder.float()
    model.projector.float()
    model.encoder.encoder.config.use_cache=False
    model.generator.requires_grad_(False).eval()
    pars=[]
    for name,param in model.generator.generate_model.named_parameters():
        if 'lora_' in name:
            param.data=param.data.float()
            param.requires_grad_(True)
            pars.append(param)
    assert pars
    assert all(not p.requires_grad for n,p in model.generator.generate_model.named_parameters() if 'lora_' not in n)
    opt=torch.optim.AdamW(pars,lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    losses=[]
    first_grad=None
    model.generator.generate_model.config.use_cache=False
    with (out/'training.jsonl').open('w',encoding='utf-8') as log:
        for step,start in enumerate(range(0,5000,64)):
            resource_guard()
            group=order[start:start+64]
            scale=step/50 if step<50 else (79-step)/29
            for pg in opt.param_groups:
                pg['lr']=5e-6*scale
            opt.zero_grad(set_to_none=True)
            loss_value=0.
            for j in range(0,len(group),4):
                resource_guard()
                ids=group[j:j+4]
                enc,gen,labels=data_batch(model,[rows[i] for i in ids])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():
                        z=model.encoder(enc['input_ids'],enc['attention_mask'])
                        z=model.projector(z.reshape(len(ids),8,-1)).reshape(len(ids)*2,-1)
                    loss=model.generator(z,gen['input_ids'],gen['attention_mask'],labels)
                assert torch.isfinite(loss)
                loss_value+=loss.detach().float().item()*len(ids)/len(group)
                (loss*len(ids)/len(group)).backward()
            grad=torch.nn.utils.clip_grad_norm_(pars,1.)
            assert torch.isfinite(grad)
            if step==0:
                first_grad=grad.item()
                assert first_grad>0
                assert all(p.grad is None for p in model.encoder.parameters())
                assert all(p.grad is None for p in model.projector.parameters())
                assert all(p.grad is None for n,p in model.generator.generate_model.named_parameters() if 'lora_' not in n)
            opt.step()
            losses.append(loss_value)
            record={'step':step+1,'updates':79,'examples_seen':start+len(group),'loss':loss_value,
                    'lr':5e-6*scale,'gradient_norm_before_clip':grad.item(),
                    'elapsed_seconds':time.monotonic()-started,
                    'gpu_allocated_GiB':torch.cuda.memory_allocated()/1024**3}
            log.write(json.dumps(record)+'\n')
            log.flush()
            if step==0 or (step+1)%5==0 or step==78:
                print(json.dumps(record),flush=True)
    resource_guard()
    destination=out/'generator_lora'
    model.generator.generate_model.save_pretrained(destination,safe_serialization=True)
    stats={'updated':'generator_LoRA_only','examples':5000,'updates':79,
           'trainable_parameters':sum(p.numel() for p in pars),'first_gradient_norm':first_grad,
           'frontend_and_base_frozen':True,'generator_eval':True,
           'runtime_seconds':time.monotonic()-started,'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
           'mean_training_loss':sum(losses)/len(losses),'adapter_sha256':sha(destination/'adapter_model.safetensors')}
    (out/'train_metrics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':
    main()
