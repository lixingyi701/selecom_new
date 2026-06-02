import logging
import string
import time
import os
import random
import regex
import torch
import json
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from collections import Counter
# rouge_scorer and vllm are only needed in evaluation functions.
# Lazy-imported inside each function to avoid triggering nltk/sqlite3
# dependency chains during training.
from transformers import AutoTokenizer


def setuplogging(args):
	root = logging.getLogger()
	root.setLevel(logging.INFO)
	if not os.path.exists(f'{args.log_path}/{args.task_type}'):
		os.mkdir(f'{args.log_path}/{args.task_type}')
	curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
	handler = logging.FileHandler(f'{args.log_path}/{args.task_type}/{curr_time}.log')
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
	handler.setFormatter(formatter)
	root.addHandler(handler)
	

def set_seed(seed=2025):
	random.seed(seed)  
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)  
	torch.manual_seed(seed)  
	torch.cuda.manual_seed(seed)  
	torch.backends.cudnn.deterministic = True  


def save_jsonl(data, path):
	with open(path, 'w', encoding='utf-8') as f:
		for item in tqdm(data):
			f.write(json.dumps(item, ensure_ascii=False) + '\n')
	print(f'Saved to {path}')


def load_jsonl(path):
	with open(path, 'r', encoding='utf-8') as f:
		result = []
		for line in tqdm(f):
			result.append(json.loads(line.strip('\n')))
			if len(result) == 100000000:
				break
		print(f'Loaded from {path}')
		return result
	

def normalize_answer(s):
	def remove_articles(text):
		return regex.sub(r'\b(a|an|the)\b', ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_exact_match(predictions, groundtruths):
	assert len(predictions) == len(groundtruths)
	correct = 0
	for pred, gt in zip(predictions, groundtruths):
		if pred is None:
			continue
		if not isinstance(gt, List):
			correct += 1 if normalize_answer(pred) == normalize_answer(gt) else 0
		else:
			correct += 1 if normalize_answer(pred) in [normalize_answer(gt_answer) for gt_answer in gt] else 0
	total = len(groundtruths)
	print(f'Exact Match: {correct}/{total} = {correct/total:.4f}')


def evaluate_f1(predictions, groundtruths):
	assert len(predictions) == len(groundtruths)
	correct = 0
	for pred, gt in zip(predictions, groundtruths):
		if pred is None:
			continue
		if not isinstance(gt, List):
			correct += f1_score(pred, gt)
		else:
			max_f1 = 0
			for gt_answer in gt:
				max_f1 = max(max_f1, f1_score(pred, gt_answer))
			correct += max_f1
	total = len(groundtruths)
	print(f'F1 Score: {correct/total:.4f}')


def evaluate_rouge_l(predictions, groundtruths):
	from rouge_score import rouge_scorer  # lazy import: avoids nltk/sqlite3 at module load
	assert len(predictions) == len(groundtruths)
	scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
	total_score = 0.0
	for pred, gt in zip(predictions, groundtruths):
		if pred is None:
			continue
		if not isinstance(gt, List):
			score = scorer.score(pred, gt)['rougeL'].fmeasure
			total_score += score
		else:
			max_score = 0.0
			for gt_answer in gt:
				s = scorer.score(pred, gt_answer)['rougeL'].fmeasure
				max_score = max(max_score, s)
			total_score += max_score
	total = len(groundtruths)
	print(f'ROUGE-L: {total_score/total:.4f}')


def evaluate_match(predictions, groundtruths):
	assert len(predictions) == len(groundtruths)
	correct = 0
	for pred, gt in zip(predictions, groundtruths):
		if pred is None:
			continue
		if not isinstance(gt, List):
			if normalize_answer(pred) in normalize_answer(gt):
				correct += 1
			elif normalize_answer(gt) in normalize_answer(pred):
				correct += 1
		else:
			for gt_answer in gt:
				if normalize_answer(pred) in normalize_answer(gt_answer):
					correct += 1
					break
				if normalize_answer(gt_answer) in normalize_answer(pred):
					correct += 1
					break
	total = len(groundtruths)
	print(f'Match: {correct}/{total} = {correct/total:.4f}')


def evaluate_llm_as_a_judge(questions, predictions, groundtruths, model_name, num_gpus, max_retries=5):
	from vllm import LLM, SamplingParams  # lazy import: only needed at eval time
	assert len(predictions) == len(groundtruths)
	if questions is not None:
		assert len(questions) == len(predictions)

	llm = LLM(
		model=model_name,
		tensor_parallel_size=num_gpus,
		gpu_memory_utilization=0.9,
		max_model_len=4096,
		trust_remote_code=True,
		dtype=torch.bfloat16
	)
	judge_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	
	# Set sampling parameters
	sampling_params = SamplingParams(
		temperature=0.2,
		max_tokens=50
	)
	
	eval_samples = []
	sample_indices = []  # Record the original index for each evaluation sample
	
	for i, (pred, gt) in enumerate(zip(predictions, groundtruths)):
		if pred is None:
			continue
			
		question = questions[i] if questions is not None else f"Question {i+1}"
		
		if isinstance(gt, list):
			for gt_item in gt:
				eval_samples.append({
					'question': question,
					'prediction': pred,
					'groundtruth': gt_item,
					'original_index': i,
					'score': None,
					'attempts': 0
				})
				sample_indices.append(i)
		else:
			eval_samples.append({
				'question': question,
				'prediction': pred,
				'groundtruth': gt,
				'original_index': i,
				'score': None,
				'attempts': 0
			})
			sample_indices.append(i)
	
	print(f"Total samples to evaluate: {len(eval_samples)} (original samples: {len([p for p in predictions if p is not None])})")
	
	retry_count = 0
	while retry_count < max_retries:
		pending_samples = [sample for sample in eval_samples if sample['score'] is None]
		
		if not pending_samples:
			print("All samples have obtained valid scores!")
			break
			
		print(f"Round {retry_count + 1} evaluation, {len(pending_samples)} samples remaining for scoring...")
		
		prompts = []
		for sample in pending_samples:
			system_prompt = "You are an evaluation assistant. You will be given a question, a candidate answer, and a reference answer. Judge whether the candidate answer correctly addresses the question compared to the reference. Respond **ONLY** with a numeric score between 0 (completely wrong) and 1 (perfectly correct)."
			user_prompt = f"Question: {sample['question']}\nCandidate Answer: {sample['prediction']}\nReference Answer: {sample['groundtruth']}\nScore:"
			
			messages = [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt}
			]
			full_prompt = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
			prompts.append(full_prompt)
			sample['attempts'] += 1
		
		outputs = llm.generate(prompts, sampling_params)
		
		for i, (output, sample) in enumerate(zip(outputs, pending_samples)):
			try:
				score_text = output.outputs[0].text.strip()
				try:
					score = float(score_text)
				except ValueError:
					import re
					m = re.search(r'(?<![0-9])(0(?:\.[0-9]+)?|1(?:\.0*)?)(?![0-9])', score_text)
					if not m:
						raise
					score = float(m.group())
				score = max(0.0, min(1.0, score))
				sample['score'] = score
			except (ValueError, IndexError):
				continue
		
		retry_count += 1
	
	failed_samples = [sample for sample in eval_samples if sample['score'] is None]
	if failed_samples:
		print(f"After {max_retries} retries, {len(failed_samples)} samples still failed to get valid scores, setting to 0.0")
		for sample in failed_samples:
			sample['score'] = 0.0	
	
	original_sample_scores = {}
	for sample in eval_samples:
		orig_idx = sample['original_index']
		if orig_idx not in original_sample_scores:
			original_sample_scores[orig_idx] = []
		original_sample_scores[orig_idx].append(sample['score'])
	
	final_scores = []
	for i in range(len(predictions)):
		if predictions[i] is None or i not in original_sample_scores:
			final_scores.append(None)
		else:
			scores_for_sample = original_sample_scores[i]
			final_scores.append(max(scores_for_sample))
	
	average_score = sum(final_scores) / len(final_scores)
	print(f'LLM Judge: {sum(final_scores)}/{len(final_scores)} = {average_score:.4f}')
	

def f1_score(prediction, ground_truth):
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1





