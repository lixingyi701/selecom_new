import sys
sys.path.append('..')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import argparse
import warnings
warnings.filterwarnings('ignore')

from transformers import Trainer, TrainingArguments, PrinterCallback, AutoTokenizer
from util.llm_utils import FilePrinterCallback, get_encode_prompt, get_qa_prompt
from model.model_combination import SelectQATrainModel
from util.data import SelectQADataset
from util.constant import *


class data_collator:
	def __init__(self, args):
		self.args = args

		self.encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, padding_side='left', use_fast=True, trust_remote_code=True)
		self.encoder_tokenizer.add_tokens([ENCODE_TOKEN], special_tokens=True)

		self.generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_name, padding_side='left', use_fast=True, trust_remote_code=True)
		self.generator_tokenizer.add_tokens([SOFT_PROMPT_START, SOFT_PROMPT_TOKEN, SOFT_PROMPT_END, RANK_TOKEN], special_tokens=True)
		self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
		self.soft_prompt_token_id = self.generator_tokenizer.convert_tokens_to_ids(SOFT_PROMPT_TOKEN)
		
	def __call__(self, data):
		questions = [item['question'] for item in data]
		documents = [item['documents'][:self.args.rerank_top_k] for item in data]
		answers = [item['answer'] for item in data]

		encode_prompts = get_encode_prompt(self.args.encoder_name, questions, documents, answers, self.args.num_emb_tokens)
		encoder_input = self.encoder_tokenizer(
			encode_prompts, 
			padding=True, 
			truncation=True, 
			max_length=self.args.encoder_max_length, 
			return_tensors='pt'
		)
		encoder_input_ids = encoder_input['input_ids']
		encoder_attention_mask = encoder_input['attention_mask']
		
		qa_prompts = get_qa_prompt(self.args.generator_name, questions, documents, answers, self.args.num_doc_tokens)
		generator_input = self.generator_tokenizer(
			qa_prompts,
			max_length=self.args.generator_max_length,
			padding=True,
			truncation=True,
			return_tensors="pt"
		)
		generator_input_ids = generator_input['input_ids']
		generator_attention_mask = generator_input['attention_mask']
		generator_labels = generator_input_ids.clone()

		for i, input_ids in enumerate(generator_input_ids):
			if 'Qwen' in self.args.generator_name:
				# Find the answer position
				# Mask non-answer parts 
				position = (input_ids == self.generator_tokenizer.convert_tokens_to_ids("<|im_start|>")).nonzero(as_tuple=False)
				if position.numel() > 0:
					idx = position[-1, 0] + 2
			else:
				# For Mistral
				# Find the answer position
				# Mask non-answer parts 
				position = (input_ids == 13).nonzero(as_tuple=False)
				idx = None
				if position.numel() > 0:
					for pos in position:
						pos = pos[0]
						if input_ids[pos - 1] == 28793 and input_ids[pos - 2] == 16289 and input_ids[pos - 3] == 28748:
							idx = pos
							break
				assert idx is not None
			generator_labels[i, :idx + 1] = IGNORE_TOKEN_ID

		return {
			'encoder_input_ids': encoder_input_ids,
			'encoder_attention_mask': encoder_attention_mask,
			'generator_input_ids': generator_input_ids,
			'generator_attention_mask': generator_attention_mask,
			'generator_labels': generator_labels,
		}


class SelectQATrainer(Trainer):
	def _save(self, output_dir=None, state_dict=None):
		output_dir = output_dir or self.args.output_dir
		self.model.save(output_dir)


def train():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default='path/to/data/stage2.jsonl') # To be filled
	parser.add_argument("--encoder_name", type=str, default='path/to/your/model/position') # To be filled
	parser.add_argument("--checkpoint_dir", type=str, default='path/to/checkpoint') # To be filled
	parser.add_argument("--generator_name", type=str, default='path/to/your/model/position') # To be filled
	parser.add_argument("--model_dir", type=str, default='path/to/save/model') # To be filled
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--learning_rate", type=float, default=0.0001)
	parser.add_argument("--batch_size", type=int, default=3)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
	parser.add_argument("--encoder_max_length", type=int, default=2560)
	parser.add_argument("--generator_max_length", type=int, default=1024)
	parser.add_argument("--random_seed", type=int, default=2025)
	parser.add_argument("--log_dir", type=str, default='path/to/log/stage2') # To be filled
	parser.add_argument("--log_name", type=str, default='train_stage2.log')
	parser.add_argument("--num_emb_tokens", type=int, default=8)
	parser.add_argument("--num_doc_tokens", type=int, default=2)
	parser.add_argument("--rerank_top_k", type=int, default=1)
	parser.add_argument("--local_rank", type=int, default=0)

	args = parser.parse_args()

	model = SelectQATrainModel(args)

	dataset = SelectQADataset(args.data_path)

	training_args = TrainingArguments(
        output_dir=args.model_dir,
		do_train=True,
		learning_rate=args.learning_rate,
		per_device_train_batch_size=args.batch_size,
		num_train_epochs=args.epochs,
		seed=args.random_seed,
		save_strategy="epoch",
		logging_steps=100,
		remove_unused_columns=False,
		dataloader_pin_memory=True,
		dataloader_num_workers=4,
		dataloader_prefetch_factor=4,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		report_to='none', 
		deepspeed='dp_config.json',
		bf16=True,
		ddp_find_unused_parameters=False,
    )	

	trainer = SelectQATrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
		data_collator=data_collator(args)
    )
	trainer.remove_callback(PrinterCallback)
	file_cb = FilePrinterCallback(output_file=os.path.join(args.log_dir, args.log_name))
	trainer.add_callback(file_cb)
	trainer.train()


if __name__ == "__main__":
	train()
