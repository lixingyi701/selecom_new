"""Verify archived aggregate means from retained predictions without loading models."""
import argparse
import json
import math
from pathlib import Path

def load(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def equal(got, expected, label):
    for key, value in got.items():
        if not math.isclose(value, expected[key], abs_tol=1e-10, rel_tol=1e-10):
            raise AssertionError((label, key, value, expected[key]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    archived = json.loads((Path(__file__).parent / 'results/metrics.json').read_text(encoding='utf-8'))
    cases = [
        ('canonical-4462-greedy-cpuload/stage2.jsonl', archived['matched_front_evaluation']['cached_stage2']),
        ('canonical-4462-greedy-cpuload/stage1_continue.jsonl', archived['historical_continue_canonical_evaluation']['stage1_continue']),
        ('matched-controls-eval-G2/trained_with_G0.jsonl', archived['matched_front_evaluation']['trained_with_G0']),
        ('matched-controls-eval-G2/trained_with_G2.jsonl', archived['matched_front_evaluation']['trained_with_G2']),
        ('generator-only-eval-G2/generator_only.jsonl', archived['generator_only_evaluation']),
    ]
    reference = None
    verified = {}
    for name, expected in cases:
        rows = load(args.run_dir / name)
        identity = [(r['question'], r['groundtruth']) for r in rows]
        if reference is None:
            reference = identity
        assert identity == reference, 'Question/gold identity differs: ' + name
        assert len(rows) == 4462
        values = {'n': len(rows), 'EM': 100 * sum(r['EM'] for r in rows) / len(rows),
                  'F1': 100 * sum(r['F1'] for r in rows) / len(rows),
                  'content_NLL': sum(r['content_NLL'] for r in rows) / len(rows),
                  'format_invalid': sum(not r['format_valid'] for r in rows)}
        equal(values, expected, name)
        verified[name] = values
    intervention_identity = None
    for condition in ['original', 'front_G0', 'front_G2', 'generator_only']:
        name = 'document-intervention-512/' + condition + '.jsonl'
        rows = load(args.run_dir / name)
        assert len(rows) == 512
        identity = [(r['question_index'], r['wrong_document_index'], r['question']) for r in rows]
        if intervention_identity is None:
            intervention_identity = identity
        assert identity == intervention_identity
        values = {'n': len(rows), **{key: sum(r[key] for r in rows) / len(rows)
                  for key in ['true_NLL', 'wrong_NLL', 'wrong_minus_true_NLL']},
                  'original_reference_string_present': sum(r['original_document_has_reference_string'] for r in rows)}
        equal(values, archived['intervention']['conditions'][condition], name)
        verified[name] = values
    print(json.dumps({'verified_prediction_files': len(verified), 'all_means_match': True,
                      'question_and_gold_order_match': True, 'metrics': verified}, indent=2))

if __name__ == '__main__':
    main()
