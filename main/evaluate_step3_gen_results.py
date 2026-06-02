import sys
sys.path.append('..')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import argparse
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
warnings.filterwarnings('ignore')

from util.llm_utils import get_encode_prompt, get_qa_prompt
from model.model_combination import Stage3TestModel
from util.data import SelectQADataset
from util.constant import *
from util.util import save_jsonl


class data_collator:
    def __init__(self, args):
        self.args = args

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            args.encoder_name, padding_side='left', use_fast=True, trust_remote_code=True
        )
        self.encoder_tokenizer.add_tokens([ENCODE_TOKEN], special_tokens=True)

        self.generator_tokenizer = AutoTokenizer.from_pretrained(
            args.generator_name, padding_side='left', use_fast=True, trust_remote_code=True
        )
        self.generator_tokenizer.add_tokens(
            [SOFT_PROMPT_START, SOFT_PROMPT_TOKEN, SOFT_PROMPT_END, RANK_TOKEN], special_tokens=True
        )
        self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        self.soft_prompt_token_id = self.generator_tokenizer.convert_tokens_to_ids(SOFT_PROMPT_TOKEN)

    def __call__(self, data):
        questions  = [item['question'] for item in data]
        documents  = [item['documents'][:self.args.rerank_top_k] for item in data]
        answers    = [item['answer'] for item in data]

        encode_prompts = get_encode_prompt(
            self.args.encoder_name, questions, documents, answers, self.args.num_emb_tokens
        )
        encoder_input = self.encoder_tokenizer(
            encode_prompts,
            padding=True,
            truncation=True,
            max_length=self.args.encoder_max_length,
            return_tensors='pt'
        )

        qa_prompts = get_qa_prompt(
            self.args.generator_name, questions, documents, answers, self.args.num_doc_tokens, test=True
        )
        generator_input = self.generator_tokenizer(
            qa_prompts,
            max_length=self.args.generator_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'encoder_input_ids':        encoder_input['input_ids'],
            'encoder_attention_mask':   encoder_input['attention_mask'],
            'generator_input_ids':      generator_input['input_ids'],
            'generator_attention_mask': generator_input['attention_mask'],
        }, questions, answers


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",              type=str, default='nq')
    parser.add_argument("--data_path",            type=str, required=True)
    parser.add_argument("--encoder_name",         type=str, required=True)
    parser.add_argument("--generator_name",       type=str, required=True)
    parser.add_argument("--stage1_checkpoint_dir", type=str, required=True,
                        help="Stage 1 selector checkpoint (contains encoder.pt)")
    parser.add_argument("--stage3_checkpoint_dir", type=str, required=True,
                        help="Stage 3 checkpoint (contains encoder_lora/, generator_lora/, projector.pt, frozen/)")
    parser.add_argument("--evaluation_results_path", type=str, required=True)
    parser.add_argument("--batch_size",           type=int, default=4)
    parser.add_argument("--encoder_max_length",   type=int, default=2560)
    parser.add_argument("--generator_max_length", type=int, default=1024)
    parser.add_argument("--num_emb_tokens",       type=int, default=8)
    parser.add_argument("--num_doc_tokens",       type=int, default=2)
    parser.add_argument("--rerank_top_k",         type=int, default=1)
    parser.add_argument("--device_id",            type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    model = Stage3TestModel(args).to(device)
    model.eval()

    dataset      = SelectQADataset(args.data_path)
    datacollator = data_collator(args)
    dataloader   = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=datacollator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    all_outputs, all_groundtruths, all_questions = [], [], []

    for data_batch, questions, gts in tqdm(dataloader):
        with torch.no_grad():
            output = model(**data_batch)
        all_outputs.extend(output)
        all_groundtruths.extend(gts)
        all_questions.extend(questions)

    del model
    torch.cuda.empty_cache()

    os.makedirs(args.evaluation_results_path, exist_ok=True)
    save_jsonl(
        [{
            'question':    all_questions[i],
            'output':      all_outputs[i],
            'groundtruth': all_groundtruths[i],
        } for i in range(len(all_outputs))],
        f'{args.evaluation_results_path}/{args.dataset}_top{args.rerank_top_k}_results.jsonl'
    )
    print(f'Saved {len(all_outputs)} results to '
          f'{args.evaluation_results_path}/{args.dataset}_top{args.rerank_top_k}_results.jsonl')


if __name__ == "__main__":
    evaluate()