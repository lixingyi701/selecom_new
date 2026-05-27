"""
Post-training Drift Experiment — 策略：ALL ANSWER CONTENT TOKENS
================================================================

测量方式：对每个样本，在答案所有实际内容 token 位置分别计算 KL 散度，
取位置均值作为该样本的漂移量，再跨样本求均值得到 D_drift。

    答案序列结构（以 "Neil Armstrong" 为例）：
      [ '<' 'answer' '>' | Neil  Armstrong | '</' 'answer' '>' '<|im_end|>' ]
        ←─ 前缀(3 tok) ─→ ←── 本版本测量 ──→ ←───── 后缀(4 tok) ──────────→

优点：统计更稳定，覆盖答案生成的完整过程。
缺点：后续 token 以 teacher-forcing 的真实前缀为条件，不完全独立于已知信息。

与 drift_experiment_First_Token.py 的唯一代码差异：
    本文件：content_labeled = labeled_pos[content_start : content_end]     # 全部内容位置
    对比版：content_labeled = labeled_pos[content_start : content_start+1]  # 仅第一个内容位置

已验证结果（N=500, NQ 验证集）：
    D_drift = 1.0907, Std = 1.2671, Median = 0.7832  (> 0.5 阈值，漂移显著)

运行示例：
    cd /home/lxy/selecom/main
    python drift_experiment_All_Token.py \\
        --encoder_name  /home/lxy/selecom/baselineModel/Qwen3-Embedding-0.6B \\
        --generator_name /home/lxy/selecom/baselineModel/Qwen2.5-7B-Instruct \\
        --stage1_selector_ckpt /home/lxy/selecom/checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector \\
        --stage2_generator_ckpt /home/lxy/selecom/checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator \\
        --data_path /home/lxy/selecom/data/nq/eval/nq_eval.jsonl \\
        --num_samples 500 --batch_size 4 \\
        --output_path /home/lxy/selecom/results/drift_results_all_token.json
"""

import sys
sys.path.append('..')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from model.encoder import Encoder
from model.generator import Generator
from model.projector import MLPProjector
from util.data import SelectQADataset
from util.llm_utils import get_encode_prompt, get_qa_prompt
from util.constant import *


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class DriftDataCollator:
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

    def __call__(self, data):
        questions = [item['question'] for item in data]
        if 'documents' in data[0]:
            documents = [item['documents'][:self.args.rerank_top_k] for item in data]
        else:
            documents = [[item['document']] for item in data]
        answers = [
            item['answer'][0] if isinstance(item['answer'], list) else item['answer']
            for item in data
        ]

        encode_prompts = get_encode_prompt(
            self.args.encoder_name, questions, documents, answers, self.args.num_emb_tokens
        )
        encoder_input = self.encoder_tokenizer(
            encode_prompts, padding=True, truncation=True,
            max_length=self.args.encoder_max_length, return_tensors='pt'
        )

        qa_prompts = get_qa_prompt(
            self.args.generator_name, questions, documents, answers,
            self.args.num_doc_tokens, test=False
        )
        generator_input = self.generator_tokenizer(
            qa_prompts, max_length=self.args.generator_max_length,
            padding=True, truncation=True, return_tensors='pt'
        )
        generator_input_ids = generator_input['input_ids']
        generator_labels = generator_input_ids.clone()

        for i, input_ids in enumerate(generator_input_ids):
            if 'Qwen' in self.args.generator_name:
                position = (input_ids == self.generator_tokenizer.convert_tokens_to_ids("<|im_start|>")).nonzero(as_tuple=False)
                idx = position[-1, 0] + 2 if position.numel() > 0 else 0
            else:
                position = (input_ids == 13).nonzero(as_tuple=False)
                idx = None
                if position.numel() > 0:
                    for pos in position:
                        pos = pos[0]
                        if input_ids[pos-1] == 28793 and input_ids[pos-2] == 16289 and input_ids[pos-3] == 28748:
                            idx = pos
                            break
                if idx is None:
                    idx = 0
            generator_labels[i, :idx + 1] = IGNORE_TOKEN_ID

        return {
            'encoder_input_ids': encoder_input['input_ids'],
            'encoder_attention_mask': encoder_input['attention_mask'],
            'generator_input_ids': generator_input_ids,
            'generator_attention_mask': generator_input['attention_mask'],
            'generator_labels': generator_labels,
        }


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_selector_components(args, device):
    encoder = Encoder(args)
    ckpt = args.stage1_selector_ckpt
    encoder.encoder.load_state_dict(torch.load(os.path.join(ckpt, 'encoder.pt'), map_location='cpu'))
    encoder.encode_token_embedding_layer.load_state_dict(
        torch.load(os.path.join(ckpt, 'encode_token_embedding_layer.pt'), map_location='cpu')
    )
    print(f'Encoder loaded from {ckpt}')

    encoder_size = encoder.encoder.embed_tokens.weight.shape[-1]
    gen_config = AutoConfig.from_pretrained(args.generator_name, trust_remote_code=True)
    generator_size = gen_config.hidden_size

    projector = MLPProjector(
        encoder_size, generator_size, args.num_emb_tokens, args.num_doc_tokens
    ).to(torch.bfloat16)
    projector.load_state_dict(torch.load(os.path.join(ckpt, 'projector.pt'), map_location='cpu'))
    print(f'Projector loaded from {ckpt}')

    encoder = encoder.to(device)
    projector = projector.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    for p in projector.parameters():
        p.requires_grad = False
    return encoder, projector


def load_generator(args, stage: int, device):
    gen = Generator(args)
    ckpt = args.stage1_selector_ckpt
    gen.soft_prompt_start_embedding_layer.load_state_dict(
        torch.load(os.path.join(ckpt, 'soft_prompt_start_embedding_layer.pt'), map_location='cpu')
    )
    gen.soft_prompt_end_embedding_layer.load_state_dict(
        torch.load(os.path.join(ckpt, 'soft_prompt_end_embedding_layer.pt'), map_location='cpu')
    )
    print(f'Special token embeddings loaded from {ckpt}')

    if stage == 2:
        gen.generate_model = PeftModel.from_pretrained(gen.generate_model, args.stage2_generator_ckpt)
        print(f'LoRA weights loaded from {args.stage2_generator_ckpt}')
    else:
        print('Using base model as θ1 (Stage 1 frozen generator)')

    gen = gen.to(device)
    for p in gen.parameters():
        p.requires_grad = False
    return gen


# ---------------------------------------------------------------------------
# Compressed embedding computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_embeddings(encoder, projector, encoder_input_ids, encoder_attention_mask,
                       num_emb_tokens, num_doc_tokens):
    embeddings = encoder(encoder_input_ids, encoder_attention_mask)
    B = embeddings.shape[0]
    D = embeddings.shape[1] // num_emb_tokens
    embeddings = embeddings.reshape(B, num_emb_tokens, D)
    projected = projector(embeddings)
    return projected.reshape(B * num_doc_tokens, -1)


# ---------------------------------------------------------------------------
# [ALL_TOKEN] 在所有答案内容 token 位置提取 logits
# ---------------------------------------------------------------------------

# 答案序列中格式标签的 token 数量（与 get_qa_prompt 模板强绑定）：
#   前缀 <answer>   : '<' 'answer' '>'         = 3 tokens
#   后缀 </answer>  : '</' 'answer' '>'         = 3 tokens
#        <|im_end|> / </s>                      = 1 token
#   total suffix = 4 tokens
N_FMT_PREFIX = 3
N_FMT_SUFFIX = 4


@torch.no_grad()
def get_answer_content_logits(generator, proj_embeddings, generator_input_ids,
                               generator_attention_mask, generator_labels):
    """
    [ALL_TOKEN 版本]
    提取所有实际答案内容 token 对应的 logit 向量（跳过 <answer>...</answer> 格式标签）。

    答案序列布局：
      [ '<' 'answer' '>' | tok1 tok2 ... tokM | '</' 'answer' '>' '<|im_end|>' ]
        ←── N_FMT_PREFIX ──→ ←── 本函数提取 ──→ ←──── N_FMT_SUFFIX ────────────→

    返回：list of B tensors，每个形状 [M_i, vocab_size]，M_i 为第 i 个样本的内容长度。
    """
    dev = generator.generate_model.device
    generator_input_ids  = generator_input_ids.to(dev)
    generator_attention_mask = generator_attention_mask.to(dev)
    generator_labels     = generator_labels.to(dev)
    proj_embeddings      = proj_embeddings.to(dev)

    input_embeds, input_mask = generator.prepare_input(
        proj_embeddings, generator_input_ids, generator_attention_mask
    )
    output = generator.generate_model(
        input_ids=None, attention_mask=input_mask, inputs_embeds=input_embeds
    )
    logits = output.logits  # [B, seq_len, vocab_size]

    B = logits.shape[0]
    results = []
    for i in range(B):
        labeled_pos   = (generator_labels[i] != IGNORE_TOKEN_ID).nonzero(as_tuple=False)[:, 0]
        n_labeled     = labeled_pos.numel()
        content_start = N_FMT_PREFIX
        content_end   = n_labeled - N_FMT_SUFFIX

        if content_end > content_start:
            # ★ ALL_TOKEN：取全部内容 token 位置 [tok1, tok2, ..., tokM]
            content_labeled = labeled_pos[content_start:content_end]
        else:
            # 答案极短，格式标签已覆盖全部；回退到第一个 labeled 位置
            content_labeled = labeled_pos[:1]

        pred_positions = (content_labeled - 1).clamp(min=0)
        results.append(logits[i, pred_positions, :].half().cpu())

    return results


# ---------------------------------------------------------------------------
# KL 散度：KL(P1 || P2)，在内容 token 位置上取均值
# ---------------------------------------------------------------------------

def compute_kl_divergence(logits1_list, logits2_list):
    """
    [ALL_TOKEN 版本]
    对每个样本，在所有内容 token 位置分别计算 KL(P1||P2)，取位置均值。

    Args:
        logits1_list: list of N tensors，每个 [M_i, vocab_size]
        logits2_list: 同结构，对应 θ2
    Returns:
        kl_per_sample: [N] float tensor
    """
    kl_samples = []
    for l1, l2 in zip(logits1_list, logits2_list):
        l1 = l1.float()
        l2 = l2.float()
        log_p1 = F.log_softmax(l1, dim=-1)
        log_p2 = F.log_softmax(l2, dim=-1)
        p1     = log_p1.exp()
        kl_per_pos = (p1 * (log_p1 - log_p2)).sum(dim=-1)  # [M_i]
        kl_samples.append(kl_per_pos.mean())                # 位置均值
    return torch.stack(kl_samples)  # [N]


# ---------------------------------------------------------------------------
# 主实验流程
# ---------------------------------------------------------------------------

def run_experiment(args):
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('Measurement strategy: ALL ANSWER CONTENT TOKENS (mean KL over positions)')

    full_dataset = SelectQADataset(args.data_path)
    N = min(args.num_samples, len(full_dataset))
    random.seed(args.seed)
    indices = random.sample(range(len(full_dataset)), N)
    dataset = Subset(full_dataset, indices)
    print(f'Sampled {N} examples from {len(full_dataset)} total')

    collator   = DriftDataCollator(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=0, drop_last=False)

    print('\n=== Loading Selector (Encoder + Projector) ===')
    encoder, projector = load_selector_components(args, device)

    print('\n=== Pass 1: θ1 (Stage 1 base generator) ===')
    theta1 = load_generator(args, stage=1, device=device)
    theta1.eval()

    all_logits1 = []
    for batch in tqdm(dataloader, desc='θ1 forward'):
        proj_emb = compute_embeddings(
            encoder, projector,
            batch['encoder_input_ids'].to(device),
            batch['encoder_attention_mask'].to(device),
            args.num_emb_tokens, args.num_doc_tokens
        )
        all_logits1.extend(get_answer_content_logits(
            theta1, proj_emb,
            batch['generator_input_ids'], batch['generator_attention_mask'], batch['generator_labels']
        ))
    del theta1
    torch.cuda.empty_cache()
    print(f'θ1 logits cached: {len(all_logits1)} samples')

    print('\n=== Pass 2: θ2 (Stage 2 LoRA generator) ===')
    theta2 = load_generator(args, stage=2, device=device)
    theta2.eval()

    all_logits2 = []
    for batch in tqdm(dataloader, desc='θ2 forward'):
        proj_emb = compute_embeddings(
            encoder, projector,
            batch['encoder_input_ids'].to(device),
            batch['encoder_attention_mask'].to(device),
            args.num_emb_tokens, args.num_doc_tokens
        )
        all_logits2.extend(get_answer_content_logits(
            theta2, proj_emb,
            batch['generator_input_ids'], batch['generator_attention_mask'], batch['generator_labels']
        ))
    del theta2
    torch.cuda.empty_cache()
    print(f'θ2 logits cached: {len(all_logits2)} samples')

    print('\n=== Computing KL Divergences (ALL_TOKEN) ===')
    kl_values = compute_kl_divergence(all_logits1, all_logits2).clamp(min=0.0)

    d_drift  = kl_values.mean().item()
    d_std    = kl_values.std().item()
    d_median = kl_values.median().item()
    d_p25    = kl_values.quantile(0.25).item()
    d_p75    = kl_values.quantile(0.75).item()
    d_max    = kl_values.max().item()
    d_min    = kl_values.min().item()

    THRESHOLD = 0.5
    print('\n' + '=' * 60)
    print('POST-TRAINING DRIFT RESULTS  [ALL_TOKEN]')
    print('=' * 60)
    print(f'  Measurement     : mean KL over ALL answer content tokens')
    print(f'  Samples         : {N}')
    print(f'  D_drift (mean)  : {d_drift:.4f}')
    print(f'  Std             : {d_std:.4f}')
    print(f'  Median          : {d_median:.4f}')
    print(f'  P25 / P75       : {d_p25:.4f} / {d_p75:.4f}')
    print(f'  Min / Max       : {d_min:.4f} / {d_max:.4f}')
    print('-' * 60)
    if d_drift > THRESHOLD:
        print(f'  CONCLUSION: D_drift={d_drift:.4f} > {THRESHOLD} => Drift confirmed. Stage 3 motivated.')
    else:
        print(f'  CONCLUSION: D_drift={d_drift:.4f} <= {THRESHOLD} => No significant drift.')
    print('=' * 60)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'measurement_strategy': 'ALL_TOKEN',
                'num_samples': N,
                'd_drift_mean': d_drift, 'd_drift_std': d_std,
                'd_drift_median': d_median, 'd_drift_p25': d_p25, 'd_drift_p75': d_p75,
                'd_drift_min': d_min, 'd_drift_max': d_max,
                'threshold': THRESHOLD, 'drift_confirmed': d_drift > THRESHOLD,
                'per_sample_kl': kl_values.tolist(),
                'config': {
                    'encoder_name': args.encoder_name,
                    'generator_name': args.generator_name,
                    'stage1_selector_ckpt': args.stage1_selector_ckpt,
                    'stage2_generator_ckpt': args.stage2_generator_ckpt,
                    'num_emb_tokens': args.num_emb_tokens,
                    'num_doc_tokens': args.num_doc_tokens,
                }
            }, f, indent=2, ensure_ascii=False)
        print(f'Results saved to {args.output_path}')

    return d_drift


def main():
    parser = argparse.ArgumentParser(description='Drift experiment — ALL_TOKEN strategy')
    parser.add_argument('--encoder_name',          type=str, required=True)
    parser.add_argument('--generator_name',         type=str, required=True)
    parser.add_argument('--stage1_selector_ckpt',   type=str, required=True)
    parser.add_argument('--stage2_generator_ckpt',  type=str, required=True)
    parser.add_argument('--data_path',              type=str, required=True)
    parser.add_argument('--num_samples',            type=int, default=500)
    parser.add_argument('--rerank_top_k',           type=int, default=1)
    parser.add_argument('--num_emb_tokens',         type=int, default=8)
    parser.add_argument('--num_doc_tokens',         type=int, default=2)
    parser.add_argument('--encoder_max_length',     type=int, default=2560)
    parser.add_argument('--generator_max_length',   type=int, default=1024)
    parser.add_argument('--batch_size',             type=int, default=4)
    parser.add_argument('--device_id',              type=int, default=0)
    parser.add_argument('--seed',                   type=int, default=2025)
    parser.add_argument('--output_path',            type=str, default=None)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
