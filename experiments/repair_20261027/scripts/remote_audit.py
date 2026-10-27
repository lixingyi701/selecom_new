"""CPU-only audit; never loads a whole corpus or initializes CUDA."""
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
import torch
from transformers import AutoTokenizer

root = Path(os.environ['SELECOM_ROOT']).expanduser().resolve()
out = Path(os.environ['SELECOM_RUN_DIR']).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)
if (out/'historical_audit.json').exists() or (out/'historical-5000-seed2025.jsonl').exists():
    raise RuntimeError('Audit outputs already exist; preserve them and select a new run directory')
base = root / 'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector'
cont = root / 'results/stage1_continue/checkpoint-79'
audit = {'time': time.time(), 'checkpoint_comparison': {}, 'data': {}}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

for name in ['encoder.pt', 'projector.pt', 'encode_token_embedding_layer.pt',
             'soft_prompt_start_embedding_layer.pt', 'soft_prompt_end_embedding_layer.pt']:
    a = torch.load(base / name, map_location='cpu', weights_only=True, mmap=True)
    b = torch.load(cont / name, map_location='cpu', weights_only=True, mmap=True)
    changed = 0
    max_abs = 0.
    squared_difference = 0.
    squared_original = 0.
    n = 0
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].shape == b[key].shape
        n += a[key].numel()
        if torch.equal(a[key], b[key]):
            continue
        changed += 1
        # Bound additional memory by tensor chunks.
        aa, bb = a[key].reshape(-1), b[key].reshape(-1)
        for start in range(0, aa.numel(), 1_000_000):
            x = aa[start:start+1_000_000].float()
            delta = bb[start:start+1_000_000].float() - x
            max_abs = max(max_abs, delta.abs().max().item())
            squared_difference += delta.square().sum().item()
            squared_original += x.square().sum().item()
    audit['checkpoint_comparison'][name] = {
        'parameters': n, 'changed_tensors': changed, 'total_tensors': len(a),
        'max_abs_delta': max_abs,
        'relative_l2_among_changed_tensors': (squared_difference / max(squared_original, 1e-30)) ** .5,
        'baseline_sha256': sha(base / name), 'continued_sha256': sha(cont / name)
    }
    del a, b
    (out/'historical_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

source = root / 'data/stage1/stage1_train_data.jsonl'
with source.open('rb') as f:
    n_rows = sum(1 for _ in f)
indices = random.Random(2025).sample(range(n_rows), 5000)
selected = set(indices)
rows = {}
with source.open(encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in selected:
            rows[i] = json.loads(line)
        if len(rows) == len(selected):
            break
assert len(rows) == 5000
data_path = out / 'historical-5000-seed2025.jsonl'
with data_path.open('w', encoding='utf-8') as f:
    for i in indices:
        f.write(json.dumps(rows[i], ensure_ascii=False) + '\n')

tokenizer = AutoTokenizer.from_pretrained(root/'baselineModel/Qwen3-Embedding-0.6B', local_files_only=True)
tokenizer.add_tokens(['<ENCODE>'], special_tokens=True)
encode_id = tokenizer.convert_tokens_to_ids('<ENCODE>')
retained = []
lengths = []
question_retained = []
for i in indices:
    row = rows[i]
    instruction = f'### Document\n{row["document"]}\n\n### Question\n{row["question"]}\n\n### Instruction\nExtract the key information from the document that is helpful to answer the question.'
    prompt = f'<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n' + '<ENCODE>'*8 + '<|im_end|>'
    ids = tokenizer(prompt)['input_ids']
    truncated = ids[:640]
    lengths.append(len(ids))
    retained.append(truncated.count(encode_id))
    question_retained.append(row['question'] in tokenizer.decode(truncated))
audit['data'] = {
    'source': str(source), 'source_size_bytes': source.stat().st_size,
    'rows': n_rows, 'historical_sample_seed': 2025, 'subset_size': 5000,
    'subset_path': str(data_path), 'subset_sha256': sha(data_path),
    'selected_indices_in_original_subset_order': indices,
    'over_640_tokens': sum(n > 640 for n in lengths),
    'all_8_encode_retained_at_640': sum(n == 8 for n in retained),
    'question_retained_at_640': sum(question_retained),
    'max_prompt_tokens': max(lengths),
    'note': 'Exact index reconstruction conditional on unchanged source JSONL and current random.sample(seed2025) implementation. No historical data hash exists.'
}
audit['generation_confounds'] = {
    'model_generation_config_do_sample': True,
    'wrapper_temperature': .2,
    'historical_scripts_baseline_batch': 128,
    'historical_scripts_continued_batch': 4,
    'historical_eval_seed_fixed': False,
    'requires_canonical_reevaluation': True
}
(out/'historical_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({**audit, 'data': {k:v for k,v in audit['data'].items() if 'indices' not in k}}, ensure_ascii=False, indent=2))
