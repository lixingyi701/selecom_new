"""Low-cost, answer-NLL diagnosis with true versus mismatched retrieved documents."""
import json
from pathlib import Path
import random
from types import SimpleNamespace
import time
import torch
import torch.nn.functional as F
from canonical_eval import Model, ROOT, WORK, SOURCE_DIR, resource_guard, sha, norm
from util.llm_utils import get_encode_prompt,get_qa_prompt

@torch.inference_mode()
def nll_batch(model,rows):
    qs=[r['question'] for r in rows]
    docs=[r['documents'][:1] for r in rows]
    answers=[r['answer'][0] if isinstance(r['answer'],list) else r['answer'] for r in rows]
    ep=get_encode_prompt(model.args.encoder_name,qs,docs,answers,8,
                         tokenizer=model.encoder.tokenizer,max_length=2560)
    enc=model.encoder.tokenizer(ep,padding=True,truncation=True,max_length=2560,return_tensors='pt')
    assert all((r==model.encoder.encode_token_id).sum().item()==8 for r in enc['input_ids'])
    z=model.encoder(enc['input_ids'],enc['attention_mask'])
    z=model.projector(z.reshape(len(rows),8,-1)).reshape(len(rows)*2,-1)
    prefixes=get_qa_prompt(model.args.generator_name,qs,docs,answers,2,test=True)
    complete=get_qa_prompt(model.args.generator_name,qs,docs,answers,2,test=False)
    full=model.generator.tokenizer(complete,padding=True,truncation=True,max_length=1024,
                                  return_tensors='pt',return_offsets_mapping=True)
    offsets=full.pop('offset_mapping')
    inp,mask=model.generator.prepare_input(z,full['input_ids'],full['attention_mask'])
    logits=model.generator.generate_model(inputs_embeds=inp,attention_mask=mask).logits
    result=[]
    for i,a in enumerate(answers):
        start=len(prefixes[i])+len('<answer>')
        end=start+len(a.strip('.'))
        positions=[j for j,(s,e) in enumerate(offsets[i].tolist()) if e>start and s<end and j>0]
        assert positions
        target=full['input_ids'][i,positions].to('cuda:0')
        selected=logits[i,torch.tensor(positions,device='cuda:0')-1].float()
        result.append(F.cross_entropy(selected,target,reduction='mean').item())
    return result

def contains_gold(doc,row):
    gold=row['answer'] if isinstance(row['answer'],list) else [row['answer']]
    text=' '+norm(doc)+' '
    return any(norm(a) and (' '+norm(a)+' ') in text for a in gold)

def main():
    args=SimpleNamespace(seed=20261027,encoder_name=str(ROOT/'baselineModel/Qwen3-Embedding-0.6B'),
                         generator_name=str(ROOT/'baselineModel/Qwen2.5-7B-Instruct'),
                         encoder_max_length=2560,generator_max_length=1024,num_emb_tokens=8,
                         num_doc_tokens=2,max_new_tokens=128,attn_implementation='flash_attention_2')
    out=WORK/'document-intervention-512'
    out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():
        print('ALREADY_COMPLETE',flush=True)
        return
    data=[]
    with (ROOT/'data/nq/eval/nq_eval.jsonl').open() as f:
        for line in f:
            x=json.loads(line)
            data.append({'question':x['question'],'answer':x['answer'],'documents':x['documents'][:1]})
    indices=sorted(random.Random(args.seed).sample(range(len(data)),512))
    real=[data[i] for i in indices]
    wrong=[]
    mappings=[]
    rng=random.Random(args.seed+1)
    for i in indices:
        row=data[i]
        start=rng.randrange(len(data))
        for k in range(len(data)):
            j=(start+k)%len(data)
            if j!=i and data[j]['documents'][0]!=row['documents'][0] and not contains_gold(data[j]['documents'][0],row):
                break
        else:
            raise RuntimeError('No mismatched document without reference string found')
        wrong.append({**row,'documents':data[j]['documents']})
        mappings.append({'question_index':i,'wrong_document_index':j,
                         'original_document_has_reference_string':contains_gold(row['documents'][0],row)})
    fronts={'original':ROOT/'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector',
            'front_G0':WORK/'controlled-G0-fp32/checkpoint-final',
            'front_G2':WORK/'controlled-G2-fp32/checkpoint-final'}
    manifest={'args':vars(args),'source_sha256':sha(__file__),'canonical_helper_sha256':sha(SOURCE_DIR/'canonical_eval.py'),
              'selection':mappings,'intervention':'keep question/gold fixed, replace top1 text with another question document with no normalized gold string; original retrieval may lack reference string',
              'note':'Artificial retrieval perturbation; gold-conditioned NLL only, not proof of causal faithfulness or free-generation accuracy'}
    manifest['training_records']={k:json.loads((WORK/v/'train_metrics.json').read_text()) for k,v in
        [('front_G0','controlled-G0-fp32'),('front_G2','controlled-G2-fp32'),('generator_only','controlled-generator-only')]}
    manifest['canonical_reference_manifest']=json.loads((WORK/'canonical-4462-greedy-cpuload/manifest.json').read_text())
    mp=out/'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text())==manifest
    else:
        mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    torch.set_num_threads(2)
    resource_guard()
    started=time.monotonic()
    model=Model(args)
    model.load_frontend(fronts['original'])
    reference=[json.loads(x) for x in (WORK/'canonical-4462-greedy-cpuload/stage2.jsonl').read_text().splitlines()]
    check=[]
    for i in range(0,64,4):
        check.extend(nll_batch(model,data[i:i+4]))
    max_delta=max(abs(v-r['content_NLL']) for v,r in zip(check,reference[:64]))
    assert max_delta<1e-4
    summary={'baseline_identity_NLL_max_delta':max_delta,'conditions':{}}
    for name in [*fronts,'generator_only']:
        resource_guard()
        if name=='generator_only':
            model.load_frontend(fronts['original'])
            path=WORK/'controlled-generator-only/generator_lora'
            model.generator.generate_model.load_adapter(path,adapter_name='generator_only',is_trainable=False,torch_device='cpu')
            model.generator.generate_model.set_adapter('generator_only')
            model.generator.requires_grad_(False).eval()
        else:
            model.load_frontend(fronts[name])
        records=[]
        for i in range(0,512,4):
            resource_guard()
            a=nll_batch(model,real[i:i+4])
            b=nll_batch(model,wrong[i:i+4])
            for j,(x,y) in enumerate(zip(a,b)):
                records.append({**mappings[i+j],'question':real[i+j]['question'],
                                'true_NLL':x,'wrong_NLL':y,'wrong_minus_true_NLL':y-x})
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        summary['conditions'][name]={'n':512,'true_NLL':sum(r['true_NLL'] for r in records)/512,
                                    'wrong_NLL':sum(r['wrong_NLL'] for r in records)/512,
                                    'wrong_minus_true_NLL':sum(r['wrong_minus_true_NLL'] for r in records)/512,
                                    'original_reference_string_present':sum(r['original_document_has_reference_string'] for r in records)}
        print(json.dumps({'condition':name,**summary['conditions'][name]}),flush=True)
    summary['runtime_seconds']=time.monotonic()-started
    summary['peak_gpu_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
