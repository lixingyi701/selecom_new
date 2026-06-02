import sys
sys.path.append('..')

import os
os.environ.setdefault('VLLM_USE_V1', '0')

import argparse
from util.util import load_jsonl, evaluate_exact_match, evaluate_f1, evaluate_llm_as_a_judge, evaluate_match


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default='path/to/result') # To be filled
    parser.add_argument("--judge_model_name", type=str, default='path/to/judge_model') # To be filled
    parser.add_argument("--num_gpus", type=int, default=2)
    args = parser.parse_args()

    all_response = load_jsonl(args.result_path)

    all_questions = [item['question'] for item in all_response]

    all_outputs = [item['output'] for item in all_response]

    all_groundtruths = [item['groundtruth'] for item in all_response]

    evaluate_exact_match(all_outputs, all_groundtruths)
    evaluate_f1(all_outputs, all_groundtruths)
    evaluate_match(all_outputs, all_groundtruths)
    evaluate_llm_as_a_judge(all_questions, all_outputs, all_groundtruths, model_name=args.judge_model_name, num_gpus=args.num_gpus)

if __name__ == "__main__":
	eval()