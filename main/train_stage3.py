"""
train_stage3.py — Stage 3 Joint Alignment Training for SeleCom
==============================================================
Implements the Task-Loss-only variant of Stage 3 (L_task, design doc §3.1).

What is trained
---------------
  · Encoder LoRA      (LoRA_φ, fresh adapter on top of Stage 1 encoder)
  · Projector MLP     (full fine-tuning, initialized from Stage 1 weights)
  · Generator LoRA    (θ_2 from Stage 2, continued fine-tuning)

Checkpoint inputs
-----------------
  --stage1_checkpoint_dir   encoder.pt, projector.pt, encode_token / soft_prompt embedding .pt
  --stage2_checkpoint_dir   adapter_config.json + adapter_model.safetensors (generator LoRA)

Output layout
-------------
  --model_dir/
    checkpoint-<step>/
      encoder_lora/       — Stage 3 encoder LoRA
      generator_lora/     — Stage 2→3 generator LoRA (merged history)
      projector.pt
      frozen/             — frozen inference artifacts
"""

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
from model.model_combination import Stage3JointModel
from util.data import Stage3Dataset
from util.constant import *


# ─────────────────────────────────────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────────────────────────────────────

class Stage3DataCollator:
    """
    Data collator for Stage 3 joint alignment training.

    Identical to Stage 2 collator with two additions:
      1. Robust 'answer' handling: accepts both str (Stage 2 training data)
         and list (NQ eval format) — takes the first element when it is a list.
      2. Supports an optional 'difficulty' field (filtering is done in Stage3Dataset).
    """

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
            [SOFT_PROMPT_START, SOFT_PROMPT_TOKEN, SOFT_PROMPT_END, RANK_TOKEN],
            special_tokens=True
        )
        self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        self.soft_prompt_token_id = self.generator_tokenizer.convert_tokens_to_ids(SOFT_PROMPT_TOKEN)

    def __call__(self, data):
        questions  = [item['question'] for item in data]
        documents  = [item['documents'][:self.args.rerank_top_k] for item in data]

        # Handle answer as str or list
        answers = []
        for item in data:
            a = item['answer']
            if isinstance(a, list):
                a = a[0]
            answers.append(a)

        # ── Encoder inputs ───────────────────────────────────────────────────
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

        # ── Generator inputs + labels ────────────────────────────────────────
        qa_prompts = get_qa_prompt(
            self.args.generator_name, questions, documents, answers, self.args.num_doc_tokens
        )
        generator_input = self.generator_tokenizer(
            qa_prompts,
            max_length=self.args.generator_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        generator_input_ids       = generator_input['input_ids']
        generator_attention_mask  = generator_input['attention_mask']
        generator_labels          = generator_input_ids.clone()

        # Mask non-answer tokens with IGNORE_TOKEN_ID (same logic as Stage 2)
        for i, input_ids in enumerate(generator_input_ids):
            if 'Qwen' in self.args.generator_name:
                position = (
                    input_ids == self.generator_tokenizer.convert_tokens_to_ids("<|im_start|>")
                ).nonzero(as_tuple=False)
                if position.numel() > 0:
                    idx = position[-1, 0] + 2
            else:
                position = (input_ids == 13).nonzero(as_tuple=False)
                idx = None
                if position.numel() > 0:
                    for pos in position:
                        pos = pos[0]
                        if (input_ids[pos - 1] == 28793 and
                                input_ids[pos - 2] == 16289 and
                                input_ids[pos - 3] == 28748):
                            idx = pos
                            break
                assert idx is not None, 'Could not find [/INST] boundary in Mistral prompt'
            generator_labels[i, :idx + 1] = IGNORE_TOKEN_ID

        return {
            'encoder_input_ids':        encoder_input['input_ids'],
            'encoder_attention_mask':   encoder_input['attention_mask'],
            'generator_input_ids':      generator_input_ids,
            'generator_attention_mask': generator_attention_mask,
            'generator_labels':         generator_labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer subclass
# ─────────────────────────────────────────────────────────────────────────────

class Stage3Trainer(Trainer):
    """
    HuggingFace Trainer subclass for Stage 3 joint alignment training.

    Additions over the base Trainer:
      · create_optimizer(): applies per-component learning rates so the
        Projector is trained at LR × projector_lr_multiplier (design doc §5).
        Without this override, projector_lr_multiplier has no effect because
        TrainingArguments only carries a single global learning_rate.
    """

    def __init__(self, model_args, **kwargs):
        super().__init__(**kwargs)
        self.model_args = model_args   # carries projector_lr_multiplier, etc.

    def create_optimizer(self):
        """
        Build an AdamW optimizer with two parameter groups:
          1. All trainable params except Projector → learning_rate
          2. Projector params                      → learning_rate × projector_lr_multiplier
        """
        projector_param_ids = {id(p) for p in self.model.projector.parameters()}

        base_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in projector_param_ids
        ]
        projector_params = [
            p for p in self.model.projector.parameters() if p.requires_grad
        ]

        multiplier = getattr(self.model_args, 'projector_lr_multiplier', 1.0)
        param_groups = [
            {'params': base_params,
             'lr': self.args.learning_rate},
            {'params': projector_params,
             'lr': self.args.learning_rate * multiplier},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        self.model.save(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser(description='SeleCom Stage 3 Joint Alignment Training')

    # ── Model paths ──────────────────────────────────────────────────────────
    parser.add_argument('--encoder_name',   type=str, default='../baselineModel/Qwen3-Embedding-0.6B')
    parser.add_argument('--generator_name', type=str, default='../baselineModel/Qwen2.5-7B-Instruct')
    parser.add_argument('--stage1_checkpoint_dir', type=str,
                        default='../checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector')
    parser.add_argument('--stage2_checkpoint_dir', type=str,
                        default='../checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator')

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument('--data_path', type=str,
                        default='../data/stage2/stage2_train_data.jsonl')
    parser.add_argument('--min_difficulty', type=int, default=0,
                        help='Keep only samples with difficulty >= this value. 0 = no filter.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Randomly sample N items from the dataset. Default: use all data.')

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument('--model_dir', type=str, default='../results/stage3')
    parser.add_argument('--log_dir',   type=str, default='../results/stage3/logs')
    parser.add_argument('--log_name',  type=str, default='train_stage3.log')

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument('--epochs',                       type=int,   default=3)
    parser.add_argument('--learning_rate',                type=float, default=5e-5)
    parser.add_argument('--projector_lr_multiplier',      type=float, default=1.5)
    parser.add_argument('--warmup_steps',                 type=int,   default=500,
                        help='Linear LR warm-up steps. Avoids instability when all three '
                             'components (Encoder LoRA, Projector, Generator LoRA) start '
                             'updating simultaneously at full LR.')
    parser.add_argument('--batch_size',                   type=int,   default=2)
    parser.add_argument('--gradient_accumulation_steps',  type=int,   default=4)
    parser.add_argument('--encoder_max_length',           type=int,   default=2560)
    parser.add_argument('--generator_max_length',         type=int,   default=1024)
    parser.add_argument('--random_seed',                  type=int,   default=2025)

    # ── Architecture ─────────────────────────────────────────────────────────
    parser.add_argument('--num_emb_tokens', type=int, default=8)
    parser.add_argument('--num_doc_tokens', type=int, default=2)
    parser.add_argument('--rerank_top_k',   type=int, default=1)

    # ── Encoder LoRA config ──────────────────────────────────────────────────
    parser.add_argument('--encoder_lora_r',     type=int, default=32)
    parser.add_argument('--encoder_lora_alpha', type=int, default=16)

    # ── Attention implementation ─────────────────────────────────────────────
    _default_attn = 'flash_attention_2' if torch.cuda.is_available() else 'eager'
    parser.add_argument('--attn_implementation', type=str, default=_default_attn,
                        choices=['flash_attention_2', 'sdpa', 'eager'])

    # ── Distributed ──────────────────────────────────────────────────────────
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    # ── Build model ──────────────────────────────────────────────────────────
    model = Stage3JointModel(args)

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset = Stage3Dataset(args.data_path, min_difficulty=args.min_difficulty)
    if args.max_samples is not None and args.max_samples < len(dataset):
        import random
        random.seed(args.random_seed)
        indices = random.sample(range(len(dataset)), args.max_samples)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f'[Stage3] Sampled {args.max_samples} / {len(dataset.dataset)} items for quick run.')

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        do_train=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=args.random_seed,
        save_strategy='epoch',
        logging_steps=100,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        warmup_steps=args.warmup_steps,        # Fix 3: avoid loss oscillation on cold start
        report_to='none',
        deepspeed='dp_config.json',
        bf16=True,
        ddp_find_unused_parameters=False,
    )

    # ── Build trainer ────────────────────────────────────────────────────────
    trainer = Stage3Trainer(
        model_args=args,               # Fix 1: carry projector_lr_multiplier into optimizer
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Stage3DataCollator(args),
    )
    trainer.remove_callback(PrinterCallback)
    os.makedirs(args.log_dir, exist_ok=True)
    file_cb = FilePrinterCallback(output_file=os.path.join(args.log_dir, args.log_name))
    trainer.add_callback(file_cb)

    trainer.train()


if __name__ == '__main__':
    train()
