"""Evaluate the matched controls with the frozen canonical evaluator."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from canonical_eval import Model, ROOT, WORK, SOURCE_DIR, resource_guard, sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='matched-controls-eval-G2')
    a=p.parse_args()
    args=SimpleNamespace(seed=20261027,encoder_name=str(ROOT/'baselineModel/Qwen3-Embedding-0.6B'),
                         generator_name=str(ROOT/'baselineModel/Qwen2.5-7B-Instruct'),
                         encoder_max_length=2560,generator_max_length=1024,
                         num_emb_tokens=8,num_doc_tokens=2,max_new_tokens=128,
                         attn_implementation='flash_attention_2')
    torch.set_num_threads(2)
    resource_guard()
    out=WORK/a.output
    out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():
        print('ALREADY_COMPLETE',out,flush=True)
        return
    rows=[]
    with (ROOT/'data/nq/eval/nq_eval.jsonl').open() as f:
        for line in f:
            x=json.loads(line)
            rows.append({'question':x['question'],'answer':x['answer'],'documents':x['documents'][:1]})
    assert len(rows)==4462
    ref_dir=WORK/'canonical-4462-greedy-cpuload'
    ref=[json.loads(x) for x in (ref_dir/'stage2.jsonl').read_text().splitlines()]
    ref_manifest=json.loads((ref_dir/'manifest.json').read_text())
    assert ref_manifest['evaluator_sha256']==sha(SOURCE_DIR/'canonical_eval.py')
    conditions={'trained_with_G0':WORK/'controlled-G0-fp32/checkpoint-final',
                'trained_with_G2':WORK/'controlled-G2-fp32/checkpoint-final'}
    training={}
    for name,path in conditions.items():
        stats=json.loads((path.parent/'train_metrics.json').read_text())
        assert stats['updates']==79 and stats['examples']==5000
        for filename,digest in stats['frontend_sha256'].items():
            assert sha(path/filename)==digest
        training[name]=stats
    manifest={'evaluator_sha256':sha(Path(__file__)),
              'frozen_model_helper_sha256':sha(SOURCE_DIR/'canonical_eval.py'),
              'canonical_reference_manifest':ref_manifest,'training':training,'evaluation_generator':'G2',
              'args':vars(args),'batch_size':4,'data_rows':4462}
    mp=out/'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text())==manifest
    else:
        mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    started=time.monotonic()
    model=Model(args)
    model.load_frontend(ROOT/'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector')
    # The same 64 cases and padding batches verify that the cached baseline can be reused.
    check=[]
    for i in range(0,64,4):
        check.extend(model.batch(rows[i:i+4]))
    assert all(x['raw_output']==y['raw_output'] for x,y in zip(check,ref[:64]))
    assert all(abs(x['content_NLL']-y['content_NLL'])<1e-4 for x,y in zip(check,ref[:64]))
    (out/'baseline_identity_check.json').write_text(json.dumps({'n':64,'exact_raw_answer_match':True,
        'max_abs_NLL_delta':max(abs(x['content_NLL']-y['content_NLL']) for x,y in zip(check,ref[:64]))},indent=2))
    summary={'cached_stage2':json.loads((ref_dir/'metrics.json').read_text())['stage2']}
    for name,path in conditions.items():
        model.load_frontend(path)
        dest=out/(name+'.jsonl')
        existing=[json.loads(x) for x in dest.read_text().splitlines() if x.strip()] if dest.exists() else []
        assert all(x['question']==y['question'] for x,y in zip(existing,rows))
        values=existing
        with dest.open('a',encoding='utf-8') as f:
            for i in range(len(existing),len(rows),4):
                resource_guard()
                batch=model.batch(rows[i:i+4])
                for row in batch:
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                f.flush()
                values.extend(batch)
                if len(values)%256==0 or len(values)==len(rows):
                    print(json.dumps({'condition':name,'done':len(values),'elapsed_seconds':time.monotonic()-started}),flush=True)
        summary[name]={'n':len(values),'EM':100*sum(x['EM'] for x in values)/len(values),
                       'F1':100*sum(x['F1'] for x in values)/len(values),
                       'content_NLL':sum(x['content_NLL'] for x in values)/len(values),
                       'format_invalid':sum(not x['format_valid'] for x in values)}
    summary['runtime_seconds']=time.monotonic()-started
    summary['peak_gpu_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
