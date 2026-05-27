"""
Post-training Drift — 20 对比案例详细展示
==========================================

在 drift_experiment_First_Token.py 的基础上，对每个样本额外记录：
  - 答案第一个内容 token 的文本
  - θ1 和 θ2 对该 token 的预测概率
  - 两个模型各自的 Top-5 预测词及概率
  - 该样本的 KL 散度值

处理完所有 N=500 个样本后，按 KL 降序排列，
每隔 N/20=25 个取一个样本，选出 20 个均匀覆盖漂移程度分布的对比案例，
写入 drift_experiment_exampleresult.txt。

运行示例：
    cd /home/lxy/selecom/main
    python drift_experiment_example.py \\
        --encoder_name  /home/lxy/selecom/baselineModel/Qwen3-Embedding-0.6B \\
        --generator_name /home/lxy/selecom/baselineModel/Qwen2.5-7B-Instruct \\
        --stage1_selector_ckpt /home/lxy/selecom/checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector \\
        --stage2_generator_ckpt /home/lxy/selecom/checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator \\
        --data_path /home/lxy/selecom/data/nq/eval/nq_eval.jsonl \\
        --num_samples 500 --batch_size 4 \\
        --output_path /home/lxy/selecom/results/drift_experiment_exampleresult.txt
"""

import sys
sys.path.append('..')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import random
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel

from model.encoder import Encoder
from model.generator import Generator
from model.projector import MLPProjector
from util.data import SelectQADataset
from util.llm_utils import get_encode_prompt, get_qa_prompt
from util.constant import *


# ---------------------------------------------------------------------------
# 格式标签长度（与 get_qa_prompt 模板强绑定，与 First_Token 版本相同）
# ---------------------------------------------------------------------------
N_FMT_PREFIX = 3   # <answer> → '<' 'answer' '>'
N_FMT_SUFFIX = 4   # </answer><|im_end|> → '</' 'answer' '>' '<|im_end|>'


# ---------------------------------------------------------------------------
# Data collator：额外返回 question / answer 原始文本供输出展示
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
            'encoder_input_ids':      encoder_input['input_ids'],
            'encoder_attention_mask': encoder_input['attention_mask'],
            'generator_input_ids':    generator_input_ids,
            'generator_attention_mask': generator_input['attention_mask'],
            'generator_labels':       generator_labels,
            'questions':              questions,   # 原始文本，用于结果展示
            'answers':                answers,
        }


# ---------------------------------------------------------------------------
# Model loading helpers（与 First_Token 版本相同）
# ---------------------------------------------------------------------------

def load_selector_components(args, device):
    encoder = Encoder(args)
    ckpt = args.stage1_selector_ckpt
    encoder.encoder.load_state_dict(torch.load(os.path.join(ckpt, 'encoder.pt'), map_location='cpu'))
    encoder.encode_token_embedding_layer.load_state_dict(
        torch.load(os.path.join(ckpt, 'encode_token_embedding_layer.pt'), map_location='cpu')
    )
    encoder_size = encoder.encoder.embed_tokens.weight.shape[-1]
    gen_config   = AutoConfig.from_pretrained(args.generator_name, trust_remote_code=True)
    projector = MLPProjector(
        encoder_size, gen_config.hidden_size, args.num_emb_tokens, args.num_doc_tokens
    ).to(torch.bfloat16)
    projector.load_state_dict(torch.load(os.path.join(ckpt, 'projector.pt'), map_location='cpu'))
    print(f'Selector components loaded from {ckpt}')
    encoder   = encoder.to(device)
    projector = projector.to(device)
    for p in encoder.parameters():   p.requires_grad = False
    for p in projector.parameters(): p.requires_grad = False
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
    if stage == 2:
        gen.generate_model = PeftModel.from_pretrained(gen.generate_model, args.stage2_generator_ckpt)
        print(f'θ2 (LoRA) loaded from {args.stage2_generator_ckpt}')
    else:
        print('θ1 (base model, no LoRA)')
    gen = gen.to(device)
    for p in gen.parameters(): p.requires_grad = False
    return gen


@torch.no_grad()
def compute_embeddings(encoder, projector, encoder_input_ids, encoder_attention_mask,
                       num_emb_tokens, num_doc_tokens):
    embeddings = encoder(encoder_input_ids, encoder_attention_mask)
    B = embeddings.shape[0]
    D = embeddings.shape[1] // num_emb_tokens
    embeddings = embeddings.reshape(B, num_emb_tokens, D)
    return projector(embeddings).reshape(B * num_doc_tokens, -1)


# ---------------------------------------------------------------------------
# 核心函数：前向传播并提取第一个内容 token 位置的完整 logit 向量
# 同时返回该 token 的 ID（用于后续解码和概率查询）
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_first_token_logits_and_id(generator, proj_embeddings, generator_input_ids,
                                   generator_attention_mask, generator_labels):
    """
    对 batch 中每个样本：
      1. 跳过 <answer> 前缀格式标签（N_FMT_PREFIX 个 token）
      2. 定位第一个真实内容 token 在序列中的位置
      3. 提取该位置前一步的 logit 向量（即对该 token 的预测分布）
      4. 记录该 token 的 ID（来自 generator_labels）

    返回：
        logits_list : list of B tensors，每个 [vocab_size]（cpu fp16）
        token_ids   : list of B ints，每个是第一个内容 token 的 token_id
    """
    dev = generator.generate_model.device
    generator_input_ids      = generator_input_ids.to(dev)
    generator_attention_mask = generator_attention_mask.to(dev)
    generator_labels         = generator_labels.to(dev)
    proj_embeddings          = proj_embeddings.to(dev)

    input_embeds, input_mask = generator.prepare_input(
        proj_embeddings, generator_input_ids, generator_attention_mask
    )
    output = generator.generate_model(
        input_ids=None, attention_mask=input_mask, inputs_embeds=input_embeds
    )
    logits = output.logits  # [B, seq_len, vocab_size]

    B = logits.shape[0]
    logits_list = []
    token_ids   = []

    for i in range(B):
        labeled_pos   = (generator_labels[i] != IGNORE_TOKEN_ID).nonzero(as_tuple=False)[:, 0]
        n_labeled     = labeled_pos.numel()
        content_start = N_FMT_PREFIX
        content_end   = n_labeled - N_FMT_SUFFIX

        if content_end > content_start:
            first_content_pos = labeled_pos[content_start].item()
        else:
            # 回退：答案极短
            first_content_pos = labeled_pos[0].item() if n_labeled > 0 else 0

        # logits[pred_pos] 是预测 first_content_pos 处 token 的分布
        pred_pos = max(first_content_pos - 1, 0)
        logits_list.append(logits[i, pred_pos, :].half().cpu())

        # 真实的第一个内容 token ID（来自 labels）
        token_ids.append(generator_labels[i, first_content_pos].item())

    return logits_list, token_ids


# ---------------------------------------------------------------------------
# KL(P1 || P2) for a single pair of logit vectors
# ---------------------------------------------------------------------------

def kl_single(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    l1 = logits1.float()
    l2 = logits2.float()
    log_p1 = F.log_softmax(l1, dim=-1)
    log_p2 = F.log_softmax(l2, dim=-1)
    p1 = log_p1.exp()
    return (p1 * (log_p1 - log_p2)).sum().item()


# ---------------------------------------------------------------------------
# 格式化输出：将 20 个案例写入文本文件
# ---------------------------------------------------------------------------

def format_example(rank: int, total: int, rec: dict, tok: AutoTokenizer) -> str:
    """将单个 sample record 格式化为可读文本块。"""
    W = 68
    lines = []
    lines.append('═' * W)
    lines.append(f'EXAMPLE {rank:>2} / {total}   '
                 f'[KL rank #{rec["kl_rank"]} / {total_samples},  KL = {rec["kl"]:.4f}]')
    lines.append('═' * W)

    q = rec['question']
    a = rec['answer']
    # 截断过长的展示
    lines.append(f'Question : {q[:90]}{"..." if len(q)>90 else ""}')
    lines.append(f'Answer   : {a[:80]}{"..." if len(a)>80 else ""}')
    lines.append('')

    ft = rec['first_content_token']
    lines.append(f'First content token : {repr(ft):<20}  (token_id = {rec["first_token_id"]})')
    lines.append(f'KL(θ1 ║ θ2)         : {rec["kl"]:.4f}')
    lines.append('')

    # 两模型对真实首 token 的概率
    lines.append(f'{"Probability for true token":30}  θ1 (base)     θ2 (LoRA)')
    lines.append(f'  P({repr(ft):<20})          '
                 f'{rec["p1_true"]:>10.4f}    {rec["p2_true"]:>10.4f}')
    lines.append('')

    # Top-5 对比
    lines.append(f'{"Top-5 predictions":─<{W}}')
    lines.append(f'  {"Rank":<6}  {"θ1 token":<20} {"θ1 prob":>8}    {"θ2 token":<20} {"θ2 prob":>8}')
    lines.append(f'  {"────":<6}  {"────────":<20} {"────────":>8}    {"────────":<20} {"────────":>8}')
    for i, (t1, p1, t2, p2) in enumerate(zip(
            rec['theta1_top5_tokens'], rec['theta1_top5_probs'],
            rec['theta2_top5_tokens'], rec['theta2_top5_probs']), 1):
        marker1 = ' ◀' if t1 == ft else '  '
        marker2 = ' ◀' if t2 == ft else '  '
        lines.append(f'  #{i:<5}  {repr(t1):<20} {p1:>8.4f}{marker1}    {repr(t2):<20} {p2:>8.4f}{marker2}')

    lines.append('─' * W)
    lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

# 全局变量，供 format_example 访问 N
total_samples = 500


def run_experiment(args):
    global total_samples

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 加载数据
    full_dataset = SelectQADataset(args.data_path)
    N = min(args.num_samples, len(full_dataset))
    total_samples = N
    random.seed(args.seed)
    indices = random.sample(range(len(full_dataset)), N)
    dataset = Subset(full_dataset, indices)
    print(f'Sampled {N} examples from {len(full_dataset)} total')

    collator   = DriftDataCollator(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=0, drop_last=False)

    # Generator tokenizer（用于 token id → 文字解码）
    gen_tok = AutoTokenizer.from_pretrained(
        args.generator_name, use_fast=True, trust_remote_code=True
    )
    gen_tok.add_tokens([SOFT_PROMPT_START, SOFT_PROMPT_TOKEN, SOFT_PROMPT_END, RANK_TOKEN],
                       special_tokens=True)

    # 加载 Selector
    print('\n=== Loading Selector ===')
    encoder, projector = load_selector_components(args, device)

    # ── Pass 1：θ1 ──────────────────────────────────────────────────────────
    print('\n=== Pass 1: θ1 (Stage 1 base generator) ===')
    theta1 = load_generator(args, stage=1, device=device)
    theta1.eval()

    all_logits1   = []   # list of N tensors [vocab_size] fp16
    all_token_ids = []   # list of N ints（第一个内容 token 的 id）
    all_meta      = []   # list of N dicts {question, answer}

    for batch in tqdm(dataloader, desc='θ1 forward'):
        proj_emb = compute_embeddings(
            encoder, projector,
            batch['encoder_input_ids'].to(device),
            batch['encoder_attention_mask'].to(device),
            args.num_emb_tokens, args.num_doc_tokens
        )
        logits_batch, tids_batch = get_first_token_logits_and_id(
            theta1, proj_emb,
            batch['generator_input_ids'],
            batch['generator_attention_mask'],
            batch['generator_labels'],
        )
        all_logits1.extend(logits_batch)
        all_token_ids.extend(tids_batch)
        for q, a in zip(batch['questions'], batch['answers']):
            all_meta.append({'question': q, 'answer': a})

    del theta1
    torch.cuda.empty_cache()
    print(f'θ1 done: {len(all_logits1)} samples')

    # ── Pass 2：θ2 ──────────────────────────────────────────────────────────
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
        logits_batch, _ = get_first_token_logits_and_id(
            theta2, proj_emb,
            batch['generator_input_ids'],
            batch['generator_attention_mask'],
            batch['generator_labels'],
        )
        all_logits2.extend(logits_batch)

    del theta2
    torch.cuda.empty_cache()
    print(f'θ2 done: {len(all_logits2)} samples')

    # ── 构建每个样本的完整记录 ───────────────────────────────────────────────
    print('\n=== Building sample records ===')
    TOP_K = 5
    records = []

    for i in range(N):
        l1 = all_logits1[i].float()
        l2 = all_logits2[i].float()
        tid = all_token_ids[i]

        kl_val = max(kl_single(l1, l2), 0.0)

        p1 = F.softmax(l1, dim=-1)
        p2 = F.softmax(l2, dim=-1)

        # 真实首 token 的预测概率
        p1_true = p1[tid].item()
        p2_true = p2[tid].item()

        # Top-5 预测
        top5_p1 = torch.topk(p1, TOP_K)
        top5_p2 = torch.topk(p2, TOP_K)

        def decode(tok_id):
            return gen_tok.decode([tok_id])

        records.append({
            'question':            all_meta[i]['question'],
            'answer':              all_meta[i]['answer'],
            'first_token_id':      tid,
            'first_content_token': decode(tid),
            'kl':                  kl_val,
            'p1_true':             p1_true,
            'p2_true':             p2_true,
            'theta1_top5_tokens':  [decode(t.item()) for t in top5_p1.indices],
            'theta1_top5_probs':   [v.item() for v in top5_p1.values],
            'theta2_top5_tokens':  [decode(t.item()) for t in top5_p2.indices],
            'theta2_top5_probs':   [v.item() for v in top5_p2.values],
        })

    # ── 按 KL 降序排列，均匀选 20 个 ─────────────────────────────────────────
    records.sort(key=lambda r: r['kl'], reverse=True)

    # 给每条记录标注其 KL 排名（1 = 最高漂移）
    for rank_idx, rec in enumerate(records, 1):
        rec['kl_rank'] = rank_idx

    # 每隔 N/20 取一个，覆盖从最高到最低漂移的全分布
    step = max(N // 20, 1)
    selected = [records[i * step] for i in range(20)]

    # 全局统计
    all_kl = torch.tensor([r['kl'] for r in records])
    d_drift = all_kl.mean().item()
    d_std   = all_kl.std().item()
    d_med   = all_kl.median().item()
    d_p25   = all_kl.quantile(0.25).item()
    d_p75   = all_kl.quantile(0.75).item()

    # ── 写入结果文件 ──────────────────────────────────────────────────────────
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # 文件头：全局统计
        f.write('POST-TRAINING DRIFT — 20 EXAMPLE CASES\n')
        f.write('Measurement: KL at FIRST answer content token (θ1=base, θ2=Stage2-LoRA)\n')
        f.write('◀ marks the ground-truth first content token in Top-5 lists\n')
        f.write('=' * 68 + '\n')
        f.write(f'Global stats (N={N})\n')
        f.write(f'  D_drift (mean KL) : {d_drift:.4f}\n')
        f.write(f'  Std               : {d_std:.4f}\n')
        f.write(f'  Median            : {d_med:.4f}\n')
        f.write(f'  P25 / P75         : {d_p25:.4f} / {d_p75:.4f}\n')
        f.write(f'  Selection         : 20 samples evenly spaced by KL rank '
                f'(step={step}, rank #1=highest drift)\n')
        f.write('=' * 68 + '\n\n')

        for display_idx, rec in enumerate(selected, 1):
            f.write(format_example(display_idx, 20, rec, gen_tok))

    print(f'\nResults written to {output_path}')
    print(f'Global D_drift = {d_drift:.4f} (N={N})')


def main():
    parser = argparse.ArgumentParser(description='Drift experiment — 20 example cases')
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
    parser.add_argument('--device_id',              type=int, default=3)
    parser.add_argument('--seed',                   type=int, default=2025)
    parser.add_argument('--output_path',            type=str,
                        default='/home/lxy/selecom/results/drift_experiment_exampleresult.txt')
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
