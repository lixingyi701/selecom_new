"""Final same-protocol G2 LoRA-only evaluation with a cached-baseline identity check."""
import json
from types import SimpleNamespace
import time
import torch
from canonical_eval import Model, ROOT, WORK, SOURCE_DIR, resource_guard, sha

def main():
    args=SimpleNamespace(seed=20261027,encoder_name=str(ROOT/'baselineModel/Qwen3-Embedding-0.6B'),
                         generator_name=str(ROOT/'baselineModel/Qwen2.5-7B-Instruct'),encoder_max_length=2560,
                         generator_max_length=1024,num_emb_tokens=8,num_doc_tokens=2,max_new_tokens=128,
                         attn_implementation='flash_attention_2')
    out=WORK/'generator-only-eval-G2'
    out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():
        print('ALREADY_COMPLETE',flush=True)
        return
    ref_dir=WORK/'canonical-4462-greedy-cpuload'
    source=ROOT/'data/nq/eval/nq_eval.jsonl'
    rows=[]
    with source.open() as f:
        for line in f:
            x=json.loads(line)
            rows.append({'question':x['question'],'answer':x['answer'],'documents':x['documents'][:1]})
    assert len(rows)==4462
    reference=[json.loads(x) for x in (ref_dir/'stage2.jsonl').read_text().splitlines()]
    reference_manifest=json.loads((ref_dir/'manifest.json').read_text())
    assert sha(SOURCE_DIR/'canonical_eval.py')==reference_manifest['evaluator_sha256']
    stats=json.loads((WORK/'controlled-generator-only/train_metrics.json').read_text())
    adapter=WORK/'controlled-generator-only/generator_lora'
    assert sha(adapter/'adapter_model.safetensors')==stats['adapter_sha256']
    manifest={'args':vars(args),'training':stats,'source_sha256':sha(__file__),
              'canonical_helper_sha256':sha(SOURCE_DIR/'canonical_eval.py'),'reference_manifest':reference_manifest}
    mp=out/'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text())==manifest
    else:
        mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    resource_guard()
    torch.set_num_threads(2)
    started=time.monotonic()
    model=Model(args)
    model.load_frontend(ROOT/'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector')
    check=[]
    for i in range(0,64,4):
        check.extend(model.batch(rows[i:i+4]))
    assert all(x['raw_output']==y['raw_output'] and abs(x['content_NLL']-y['content_NLL'])<1e-4
               for x,y in zip(check,reference[:64]))
    (out/'baseline_identity_check.json').write_text(json.dumps({'n':64,'raw_outputs_match':True,
        'max_abs_NLL_delta':max(abs(x['content_NLL']-y['content_NLL']) for x,y in zip(check,reference[:64]))},indent=2))
    model.generator.generate_model.load_adapter(adapter,adapter_name='generator_only',is_trainable=False,torch_device='cpu')
    model.generator.generate_model.set_adapter('generator_only')
    model.generator.requires_grad_(False).eval()
    dest=out/'generator_only.jsonl'
    values=[json.loads(x) for x in dest.read_text().splitlines() if x.strip()] if dest.exists() else []
    assert all(x['question']==y['question'] for x,y in zip(values,rows))
    with dest.open('a',encoding='utf-8') as f:
        for i in range(len(values),len(rows),4):
            resource_guard()
            batch=model.batch(rows[i:i+4])
            for x in batch:
                f.write(json.dumps(x,ensure_ascii=False)+'\n')
            f.flush()
            values.extend(batch)
            if len(values)%256==0 or len(values)==4462:
                print(json.dumps({'done':len(values),'elapsed_seconds':time.monotonic()-started}),flush=True)
    summary={'n':len(values),'EM':100*sum(x['EM'] for x in values)/len(values),
             'F1':100*sum(x['F1'] for x in values)/len(values),
             'content_NLL':sum(x['content_NLL'] for x in values)/len(values),
             'format_invalid':sum(not x['format_valid'] for x in values),
             'runtime_seconds':time.monotonic()-started,'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
