"""
Stage 3 Motivation Experiment: Quantifying Post-training Drift (Section 1 of design doc)

Measures D_drift = (1/N) * Σ KL( P^(1)(·|q_i,E_i) || P^(2)(·|q_i,E_i) )

  P^(1): logit distribution of Stage-1 frozen Generator θ1 (base model)
  P^(2): logit distribution of Stage-2 finetuned Generator θ2 (base + LoRA)

Both generators receive the same compressed embeddings E_i produced by the
fixed Selector φ (encoder + projector from Stage 1 selector checkpoint).

The KL is computed over the vocabulary distribution at the **first answer token**
position for each sample, as described in the design document eq. (2).

Empirical threshold: D_drift > 0.5 confirms significant semantic drift and
motivates Stage 3 joint alignment training.

Usage:
python drift_experiment.py \
    --encoder_name /home/lxy/selecom/baselineModel/Qwen3-Embedding-0.6B \
    --generator_name /home/lxy/selecom/baselineModel/Qwen2.5-7B-Instruct \
    --stage1_selector_ckpt /home/lxy/selecom/checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector \
    --stage2_generator_ckpt /home/lxy/selecom/checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator \
    --data_path /home/lxy/selecom/data/nq/eval/nq_eval.jsonl \
    --num_samples 500 \
    --batch_size 4 \
    --output_path /home/lxy/selecom/results/drift_results.json
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
# Data collator (mirrors train_stage2.py but also returns labels)
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
        # Support both 'document' (stage1 format) and 'documents' (stage2 format)
        if 'documents' in data[0]:
            documents = [item['documents'][:self.args.rerank_top_k] for item in data]
        else:
            documents = [[item['document']] for item in data]
        # answer may be a list (NQ format) or a string (stage2 format)
        answers = [
            item['answer'][0] if isinstance(item['answer'], list) else item['answer']
            for item in data
        ]

        # --- Encoder inputs ---
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

        # --- Generator inputs ---
        qa_prompts = get_qa_prompt(
            self.args.generator_name, questions, documents, answers,
            self.args.num_doc_tokens, test=False
        )
        generator_input = self.generator_tokenizer(
            qa_prompts,
            max_length=self.args.generator_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        generator_input_ids = generator_input['input_ids']
        generator_labels = generator_input_ids.clone()

        for i, input_ids in enumerate(generator_input_ids):
            if 'Qwen' in self.args.generator_name:
                position = (input_ids == self.generator_tokenizer.convert_tokens_to_ids("<|im_start|>")).nonzero(as_tuple=False)
                if position.numel() > 0:
                    idx = position[-1, 0] + 2
                else:
                    idx = 0
            else:
                # Mistral: find "\n" after "[/INST]"
                position = (input_ids == 13).nonzero(as_tuple=False)
                idx = None
                if position.numel() > 0:
                    for pos in position:
                        pos = pos[0]
                        if input_ids[pos - 1] == 28793 and input_ids[pos - 2] == 16289 and input_ids[pos - 3] == 28748:
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
    """Load encoder and projector from Stage 1 selector checkpoint."""
    encoder = Encoder(args)
    ckpt = args.stage1_selector_ckpt

    encoder.encoder.load_state_dict(
        torch.load(os.path.join(ckpt, 'encoder.pt'), map_location='cpu')
    )
    encoder.encode_token_embedding_layer.load_state_dict(
        torch.load(os.path.join(ckpt, 'encode_token_embedding_layer.pt'), map_location='cpu')
    )
    print(f'Encoder loaded from {ckpt}')

    encoder_size = encoder.encoder.embed_tokens.weight.shape[-1]

    # Get generator hidden size from config only (avoids loading full model weights here)
    gen_config = AutoConfig.from_pretrained(args.generator_name, trust_remote_code=True)
    generator_size = gen_config.hidden_size

    projector = MLPProjector(
        encoder_size, generator_size,
        args.num_emb_tokens, args.num_doc_tokens
    ).to(torch.bfloat16)
    projector.load_state_dict(
        torch.load(os.path.join(ckpt, 'projector.pt'), map_location='cpu')
    )
    print(f'Projector loaded from {ckpt}')

    encoder = encoder.to(device)
    projector = projector.to(device)

    for p in encoder.parameters():
        p.requires_grad = False
    for p in projector.parameters():
        p.requires_grad = False

    return encoder, projector


def load_generator(args, stage: int, device):
    """
    Load Generator wrapper.
      stage=1 → base model (no LoRA), special embeddings from Stage 1 selector ckpt
      stage=2 → base model + LoRA from stage2_generator_ckpt
    """
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
        gen.generate_model = PeftModel.from_pretrained(
            gen.generate_model, args.stage2_generator_ckpt
        )
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
def compute_embeddings(encoder, projector, encoder_input_ids, encoder_attention_mask, num_emb_tokens, num_doc_tokens):
    """Run encoder + projector to produce compressed soft-prompt embeddings."""
    embeddings = encoder(encoder_input_ids, encoder_attention_mask)
    B = embeddings.shape[0]
    D = embeddings.shape[1] // num_emb_tokens
    embeddings = embeddings.reshape(B, num_emb_tokens, D)

    projected = projector(embeddings)                           # [B, num_doc_tokens, gen_hidden]
    projected_flat = projected.reshape(B * num_doc_tokens, -1) # [B*num_doc_tokens, gen_hidden]
    return projected_flat


# ---------------------------------------------------------------------------
# Logit extraction at answer CONTENT token positions
# ---------------------------------------------------------------------------

# Format tag lengths in the answer sequence produced by get_qa_prompt:
#   prefix: <answer>      → '<' 'answer' '>'        = 3 tokens  (Qwen & Mistral)
#   suffix: </answer>     → '</' 'answer' '>'        = 3 tokens
#           <|im_end|>    → 1 token  (Qwen)
#           </s>          → 1 token  (Mistral)
#   total suffix = 4 tokens
# Only the tokens between prefix and suffix carry actual answer content.
N_FMT_PREFIX = 3
N_FMT_SUFFIX = 4


@torch.no_grad()
def get_answer_content_logits(generator, proj_embeddings, generator_input_ids,
                               generator_attention_mask, generator_labels):
    """
    Forward pass through generator. For each sample, collect the logit vectors
    that predict each ACTUAL ANSWER CONTENT token (skipping the surrounding
    <answer>...</answer> format tags which are deterministic and contribute
    near-zero KL regardless of what the soft prompt encodes).

    Labeled sequence layout per sample:
      [ '<' 'answer' '>' | Neil  Armstrong | '</' 'answer' '>' '<|im_end|>' ]
        ←── N_FMT_PREFIX ──→ ←── content ──→ ←────── N_FMT_SUFFIX ─────────→

    Returns:
        list of B tensors, each [content_len_i, vocab_size] (cpu, float16)
        Samples with no content tokens (answer too short) fall back to the
        first labeled position so they are never dropped from the average.
    """
    dev = generator.generate_model.device
    generator_input_ids = generator_input_ids.to(dev)
    generator_attention_mask = generator_attention_mask.to(dev)
    generator_labels = generator_labels.to(dev)
    proj_embeddings = proj_embeddings.to(dev)

    input_embeds, input_mask = generator.prepare_input(
        proj_embeddings, generator_input_ids, generator_attention_mask
    )
    output = generator.generate_model(
        input_ids=None,
        attention_mask=input_mask,
        inputs_embeds=input_embeds,
    )
    logits = output.logits  # [B, seq_len, vocab_size]

    B = logits.shape[0]
    results = []
    for i in range(B):
        labeled_pos = (generator_labels[i] != IGNORE_TOKEN_ID).nonzero(as_tuple=False)[:, 0]
        # labeled_pos: positions of tokens whose loss is computed (= positions
        # in the label tensor that are NOT masked).  logits[t-1] predicts label[t].
        n_labeled = labeled_pos.numel()

        content_start = N_FMT_PREFIX          # first content token index in labeled_pos
        content_end   = n_labeled - N_FMT_SUFFIX  # exclusive

        if content_end > content_start:
            # Take only the first actual content token (pure prediction from soft prompt,
            # no teacher-forced answer prefix conditioning). Matches "答案首 token" in
            # the design doc and standard practice in soft-compression papers.
            content_labeled = labeled_pos[content_start:content_start + 1]  # [1]
        else:
            # Fallback: answer is so short the format tags eat everything;
            # use first labeled position to avoid dropping this sample entirely.
            content_labeled = labeled_pos[:1]

        # logits[t-1] predicts the token at position t
        pred_positions = (content_labeled - 1).clamp(min=0)
        results.append(logits[i, pred_positions, :].half().cpu())  # [content_len, vocab]

    return results  # list of B tensors


# ---------------------------------------------------------------------------
# KL divergence: KL(P1 || P2), averaged over content token positions
# ---------------------------------------------------------------------------

def compute_kl_divergence(logits1_list, logits2_list):
    """
    Compute per-sample mean KL(P1 || P2) over all answer content token positions.

    Args:
        logits1_list: list of B tensors, each [content_len_i, vocab_size]
        logits2_list: same structure for θ2

    Returns:
        kl_per_sample: [B] float tensor  (mean KL across content positions)
    """
    kl_samples = []
    for l1, l2 in zip(logits1_list, logits2_list):
        l1 = l1.float()
        l2 = l2.float()
        log_p1 = F.log_softmax(l1, dim=-1)   # [content_len, vocab]
        log_p2 = F.log_softmax(l2, dim=-1)
        p1     = log_p1.exp()
        # KL(P1||P2) per token position, then mean across positions
        kl_per_pos = (p1 * (log_p1 - log_p2)).sum(dim=-1)  # [content_len]
        kl_samples.append(kl_per_pos.mean())
    return torch.stack(kl_samples)  # [B]


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------

def run_experiment(args):
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Load data ---
    full_dataset = SelectQADataset(args.data_path)
    N = min(args.num_samples, len(full_dataset))
    random.seed(args.seed)
    indices = random.sample(range(len(full_dataset)), N)
    dataset = Subset(full_dataset, indices)
    print(f'Sampled {N} examples from {len(full_dataset)} total')

    collator = DriftDataCollator(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )

    # --- Load selector components (shared) ---
    print('\n=== Loading Selector (Encoder + Projector) ===')
    encoder, projector = load_selector_components(args, device)

    # --- Sequential mode: compute θ1 logits, then θ2 logits ---
    # This avoids holding two large generators in memory simultaneously.

    print('\n=== Pass 1: Computing logits with θ1 (Stage 1 base generator) ===')
    theta1 = load_generator(args, stage=1, device=device)
    theta1.eval()

    all_logits1 = []  # list of N tensors, each [content_len_i, vocab_size]
    for batch in tqdm(dataloader, desc='θ1 forward'):
        proj_emb = compute_embeddings(
            encoder, projector,
            batch['encoder_input_ids'].to(device),
            batch['encoder_attention_mask'].to(device),
            args.num_emb_tokens, args.num_doc_tokens
        )
        logits1_batch = get_answer_content_logits(
            theta1, proj_emb,
            batch['generator_input_ids'],
            batch['generator_attention_mask'],
            batch['generator_labels'],
        )
        all_logits1.extend(logits1_batch)  # each element: [content_len_i, vocab]

    del theta1
    torch.cuda.empty_cache()
    print(f'θ1 logits cached: {len(all_logits1)} samples')

    print('\n=== Pass 2: Computing logits with θ2 (Stage 2 LoRA generator) ===')
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
        logits2_batch = get_answer_content_logits(
            theta2, proj_emb,
            batch['generator_input_ids'],
            batch['generator_attention_mask'],
            batch['generator_labels'],
        )
        all_logits2.extend(logits2_batch)

    del theta2
    torch.cuda.empty_cache()
    print(f'θ2 logits cached: {len(all_logits2)} samples')

    # --- Compute per-sample KL divergences (averaged over content positions) ---
    print('\n=== Computing KL Divergences ===')
    kl_values = compute_kl_divergence(all_logits1, all_logits2)  # [N]

    # Clamp numerical noise (KL should be >= 0)
    kl_values = kl_values.clamp(min=0.0)

    # --- Report results ---
    d_drift = kl_values.mean().item()
    d_std = kl_values.std().item()
    d_median = kl_values.median().item()
    d_p25 = kl_values.quantile(0.25).item()
    d_p75 = kl_values.quantile(0.75).item()
    d_max = kl_values.max().item()
    d_min = kl_values.min().item()

    print('\n' + '=' * 60)
    print('POST-TRAINING DRIFT MEASUREMENT RESULTS')
    print('=' * 60)
    print(f'  Samples evaluated : {N}')
    print(f'  D_drift (mean KL) : {d_drift:.4f}')
    print(f'  Std               : {d_std:.4f}')
    print(f'  Median            : {d_median:.4f}')
    print(f'  25th percentile   : {d_p25:.4f}')
    print(f'  75th percentile   : {d_p75:.4f}')
    print(f'  Min / Max         : {d_min:.4f} / {d_max:.4f}')
    print('-' * 60)
    THRESHOLD = 0.5
    if d_drift > THRESHOLD:
        print(f'  CONCLUSION: D_drift = {d_drift:.4f} > {THRESHOLD} (threshold)')
        print('  => Significant Post-training Drift confirmed.')
        print('  => Stage 3 Joint Alignment Training is motivated.')
    else:
        print(f'  CONCLUSION: D_drift = {d_drift:.4f} <= {THRESHOLD} (threshold)')
        print('  => Drift is below threshold; Stage 2 has not caused significant misalignment.')
    print('=' * 60)

    # --- Save results ---
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        results = {
            'num_samples': N,
            'd_drift_mean': d_drift,
            'd_drift_std': d_std,
            'd_drift_median': d_median,
            'd_drift_p25': d_p25,
            'd_drift_p75': d_p75,
            'd_drift_min': d_min,
            'd_drift_max': d_max,
            'threshold': THRESHOLD,
            'drift_confirmed': d_drift > THRESHOLD,
            'per_sample_kl': kl_values.tolist(),
            'config': {
                'encoder_name': args.encoder_name,
                'generator_name': args.generator_name,
                'stage1_selector_ckpt': args.stage1_selector_ckpt,
                'stage2_generator_ckpt': args.stage2_generator_ckpt,
                'num_emb_tokens': args.num_emb_tokens,
                'num_doc_tokens': args.num_doc_tokens,
            }
        }
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'\nResults saved to {args.output_path}')

    return d_drift


def main():
    parser = argparse.ArgumentParser(
        description='Measure Post-training Drift between Stage 1 and Stage 2 generators'
    )
    # Model paths
    parser.add_argument('--encoder_name', type=str, required=True,
                        help='Path to encoder base model (e.g. Qwen3-Embedding-0.6B)')
    parser.add_argument('--generator_name', type=str, required=True,
                        help='Path to generator base model (e.g. Qwen2.5-7B or Mistral-7B)')
    parser.add_argument('--stage1_selector_ckpt', type=str, required=True,
                        help='Stage 1 selector checkpoint dir (contains encoder.pt, projector.pt, etc.)')
    parser.add_argument('--stage2_generator_ckpt', type=str, required=True,
                        help='Stage 2 generator checkpoint dir (contains LoRA adapter weights)')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Validation data JSONL (fields: question, answer, documents)')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to evaluate (paper uses N=500)')
    parser.add_argument('--rerank_top_k', type=int, default=1,
                        help='Number of documents per sample to use')

    # Architecture
    parser.add_argument('--num_emb_tokens', type=int, default=8,
                        help='Number of <ENCODE> tokens in selector')
    parser.add_argument('--num_doc_tokens', type=int, default=2,
                        help='Number of <SOFT_PROMPT> tokens in generator')
    parser.add_argument('--encoder_max_length', type=int, default=2560)
    parser.add_argument('--generator_max_length', type=int, default=1024)

    # Execution
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save JSON results (optional)')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()