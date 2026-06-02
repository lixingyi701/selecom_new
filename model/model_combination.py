import os
import time

from peft import LoraConfig
import torch
import torch.nn.functional as F
import safetensors
import torch
import torch.nn as nn

import os
from .encoder import Encoder
from .generator import Generator
from .projector import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from util.llm_utils import compute_trainable_parameters
from util.constant import IGNORE_TOKEN_ID
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# Combined model for Stage 1 training with full tuning.
# The encoder, projector and special token embeddings are trainable.
class SelectTrainModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_emb_tokens = args.num_emb_tokens
        self.num_doc_tokens = args.num_doc_tokens

        self.encoder = Encoder(args)
        self.generator = Generator(args)

        encoder_size = self.encoder.encoder.embed_tokens.weight.shape[-1]
        generator_size = self.generator.generate_model.config.hidden_size

        self.projector = MLPProjector(encoder_size, generator_size, args.num_emb_tokens, args.num_doc_tokens).to(torch.bfloat16)
        self.config = self.generator.generate_model.config

        print(f'Encoder dimension: {encoder_size}')
        print(f'Generator dimension: {generator_size}')
        print(f'Projector shape: {encoder_size * args.num_emb_tokens} * {generator_size * args.num_doc_tokens}')
        
        try:
            self.load_checkpoint(args.checkpoint_dir)
        except Exception as e:
            print(e)
            pass
        self.set_trainable()
        compute_trainable_parameters(self)
    
    # Continue training from checkpoint or initialize from original models
    def load_checkpoint(self, checkpoint_dir):
        encoder_path = os.path.join(checkpoint_dir, 'encoder.pt')
        self.encoder.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        print(f'Encoder loaded from {checkpoint_dir}')
        
        encode_token_path = os.path.join(checkpoint_dir, 'encode_token_embedding_layer.pt')
        self.encoder.encode_token_embedding_layer.load_state_dict(torch.load(encode_token_path, map_location="cpu"))
        print(f'Encode token embedding loaded from {checkpoint_dir}')
        
        projector_path = os.path.join(checkpoint_dir, 'projector.pt')
        self.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
        print(f'Projector loaded from {checkpoint_dir}')
        
        soft_start_path = os.path.join(checkpoint_dir, 'soft_prompt_start_embedding_layer.pt')
        self.generator.soft_prompt_start_embedding_layer.load_state_dict(torch.load(soft_start_path, map_location="cpu"))
        print(f'Soft prompt start embedding loaded from {checkpoint_dir}')

        soft_end_path = os.path.join(checkpoint_dir, 'soft_prompt_end_embedding_layer.pt')
        self.generator.soft_prompt_end_embedding_layer.load_state_dict(torch.load(soft_end_path, map_location="cpu"))
        print(f'Soft prompt end embedding loaded from {checkpoint_dir}')

    # The encoder, projector and special token embeddings are trainable.
    def set_trainable(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        print(f'Trainable encoder')
        for p in self.encoder.encode_token_embedding_layer.parameters():
            p.requires_grad = True
        print(f'Trainable <ENCODE>')
        for p in self.projector.parameters():
            p.requires_grad = True
        print(f'Trainable projector')
        for p in self.generator.embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen generator embedding layer')
        for p in self.generator.soft_prompt_start_embedding_layer.parameters():
            p.requires_grad = True
        print(f'Trainable <SOFT_PROMPT_START>')
        for p in self.generator.soft_prompt_end_embedding_layer.parameters():
            p.requires_grad = True
        print(f'Trainable <SOFT_PROMPT_END>')
        for p in self.generator.rerank_token_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <RANK>') # RANK 并未实际使用
        for p in self.generator.generate_model.parameters():
            p.requires_grad = False
        print(f'Frozen generator')

    # Main function for Stage 1 training with forwarding encoder, projector and generator
    def forward(self, encoder_input_ids, encoder_attention_mask, generator_input_ids, generator_attention_mask, generator_labels):
        # embeddings: [batch_size, num_emb_tokens * hidden_size]
        embeddings = self.encoder(encoder_input_ids, encoder_attention_mask)
        B = embeddings.shape[0]
        D = embeddings.shape[1] // self.num_emb_tokens
        # embeddings: [batch_size, num_emb_tokens, hidden_size]
        embeddings = embeddings.reshape([B, self.num_emb_tokens, D])

        project_embeddings = self.projector(embeddings)
        project_embeddings = project_embeddings.reshape([embeddings.shape[0] * self.num_doc_tokens, -1])
        loss = self.generator(project_embeddings, generator_input_ids, generator_attention_mask, generator_labels)
        return {'loss': loss}

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.encoder.encoder.state_dict(), os.path.join(output_dir, "encoder.pt"))
        torch.save(self.encoder.encode_token_embedding_layer.state_dict(), os.path.join(output_dir, "encode_token_embedding_layer.pt"))
        torch.save(self.projector.state_dict(), os.path.join(output_dir, "projector.pt"))
        torch.save(self.generator.soft_prompt_start_embedding_layer.state_dict(), os.path.join(output_dir, "soft_prompt_start_embedding_layer.pt"))
        torch.save(self.generator.soft_prompt_end_embedding_layer.state_dict(), os.path.join(output_dir, "soft_prompt_end_embedding_layer.pt"))


# Combined model for Stage 2 training with LoRA.
# Only the generator with LoRA layers is trainable.
class SelectQATrainModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_emb_tokens = args.num_emb_tokens
        self.num_doc_tokens = args.num_doc_tokens
        self.encoder = Encoder(args)
        self.generator = Generator(args)
        encoder_size = self.encoder.encoder.embed_tokens.weight.shape[-1]
        generator_size = self.generator.generate_model.config.hidden_size
        self.projector = MLPProjector(encoder_size, generator_size, args.num_emb_tokens, args.num_doc_tokens).to(torch.bfloat16)
        self.config = self.generator.generate_model.config
        print(f'Encoder dimension: {encoder_size}')
        print(f'Generator dimension: {generator_size}')
        print(f'Projector shape: {encoder_size * args.num_emb_tokens} * {generator_size * args.num_doc_tokens}')
        self.load_checkpoint(args.checkpoint_dir)
        self.set_trainable()
        self.make_lora()
        compute_trainable_parameters(self)
    
    # Add LoRA layers to the generator
    def make_lora(self):
        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules='all-linear',
            lora_dropout=0.1,
            bias='none',
            task_type="CAUSAL_LM",
        )
        self.generator.generate_model = get_peft_model(self.generator.generate_model, lora_config)
        print(f'Initialized LoRA')

    # Load checkpoint from Stage 1 training
    def load_checkpoint(self, checkpoint_dir):
        encoder_path = os.path.join(checkpoint_dir, 'encoder.pt')
        self.encoder.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        print(f'Encoder loaded from {checkpoint_dir}')
        
        encode_token_path = os.path.join(checkpoint_dir, 'encode_token_embedding_layer.pt')
        self.encoder.encode_token_embedding_layer.load_state_dict(torch.load(encode_token_path, map_location="cpu"))
        print(f'Encode token embedding loaded from {checkpoint_dir}')
        
        projector_path = os.path.join(checkpoint_dir, 'projector.pt')
        self.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
        print(f'Projector loaded from {checkpoint_dir}')
        
        soft_start_path = os.path.join(checkpoint_dir, 'soft_prompt_start_embedding_layer.pt')
        self.generator.soft_prompt_start_embedding_layer.load_state_dict(torch.load(soft_start_path, map_location="cpu"))
        print(f'Soft prompt start embedding loaded from {checkpoint_dir}')

        soft_end_path = os.path.join(checkpoint_dir, 'soft_prompt_end_embedding_layer.pt')
        self.generator.soft_prompt_end_embedding_layer.load_state_dict(torch.load(soft_end_path, map_location="cpu"))
        print(f'Soft prompt end embedding loaded from {checkpoint_dir}')
    
    # Only the generator with LoRA layers is trainable.
    def set_trainable(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        print(f'Frozen encoder')
        for p in self.encoder.encode_token_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <ENCODE>')
        for p in self.projector.parameters():
            p.requires_grad = False
        print(f'Frozen projector')
        for p in self.generator.embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen generator embedding layer')
        for p in self.generator.soft_prompt_start_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <SOFT_PROMPT_START>')
        for p in self.generator.soft_prompt_end_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <SOFT_PROMPT_END>')
        for p in self.generator.rerank_token_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <RANK>')
        for p in self.generator.generate_model.parameters():
            p.requires_grad = True
        print(f'Trainable generator')

    def forward(self, encoder_input_ids, encoder_attention_mask, generator_input_ids, generator_attention_mask, generator_labels):
        with torch.no_grad():
            embeddings = self.encoder(encoder_input_ids, encoder_attention_mask)
            B = embeddings.shape[0]
            D = embeddings.shape[1] // self.num_emb_tokens
            embeddings = embeddings.reshape([B, self.num_emb_tokens, D])

            # project_embeddings: [batch_size * num_doc_tokens, hidden_size]
            project_embeddings = self.projector(embeddings)
            project_embeddings = project_embeddings.reshape([embeddings.shape[0] * self.num_doc_tokens, -1])

        loss = self.generator(project_embeddings, generator_input_ids, generator_attention_mask, generator_labels)
        return {'loss': loss}

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.generator.generate_model.save_pretrained(output_dir)


# Combined model for testing with tuned encoder, projector and generator.
# All components are frozen.
class SelectQATestModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_emb_tokens = args.num_emb_tokens
        self.num_doc_tokens = args.num_doc_tokens
        self.encoder = Encoder(args)
        self.generator = Generator(args)
        encoder_size = self.encoder.encoder.embed_tokens.weight.shape[-1]
        generator_size = self.generator.generate_model.config.hidden_size
        self.projector = MLPProjector(encoder_size, generator_size, args.num_emb_tokens, args.num_doc_tokens).to(torch.bfloat16)
        self.config = self.generator.generate_model.config
        print(f'Encoder dimension: {encoder_size}')
        print(f'Generator dimension: {generator_size}')
        print(f'Projector shape: {encoder_size * args.num_emb_tokens} * {generator_size * args.num_doc_tokens}')
        self.load_checkpoint(args.encoder_checkpoint_dir)
        self.load_lora(args.generator_checkpoint_dir)
        self.set_trainable()
        compute_trainable_parameters(self)

    # Load LoRA weights into the generator
    def load_lora(self, generator_checkpoint_dir):
        self.generator.generate_model = PeftModel.from_pretrained(
            self.generator.generate_model,
            generator_checkpoint_dir
        )
        print('LoRA loaded from', generator_checkpoint_dir)

    # Load checkpoint from Stage 1 training
    def load_checkpoint(self, checkpoint_dir):
        encoder_path = os.path.join(checkpoint_dir, 'encoder.pt')
        self.encoder.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        print(f'Encoder loaded from {checkpoint_dir}')
        
        encode_token_path = os.path.join(checkpoint_dir, 'encode_token_embedding_layer.pt')
        self.encoder.encode_token_embedding_layer.load_state_dict(torch.load(encode_token_path, map_location="cpu"))
        print(f'Encode token embedding loaded from {checkpoint_dir}')
        
        projector_path = os.path.join(checkpoint_dir, 'projector.pt')
        self.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
        print(f'Projector loaded from {checkpoint_dir}')
        
        soft_start_path = os.path.join(checkpoint_dir, 'soft_prompt_start_embedding_layer.pt')
        self.generator.soft_prompt_start_embedding_layer.load_state_dict(torch.load(soft_start_path, map_location="cpu"))
        print(f'Soft prompt start embedding loaded from {checkpoint_dir}')

        soft_end_path = os.path.join(checkpoint_dir, 'soft_prompt_end_embedding_layer.pt')
        self.generator.soft_prompt_end_embedding_layer.load_state_dict(torch.load(soft_end_path, map_location="cpu"))
        print(f'Soft prompt end embedding loaded from {checkpoint_dir}')
        
    def set_trainable(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        print(f'Frozen encoder')
        for p in self.encoder.encode_token_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <ENCODE>')
        for p in self.projector.parameters():
            p.requires_grad = False
        print(f'Frozen projector')
        for p in self.generator.embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen generator embedding layer')
        for p in self.generator.soft_prompt_start_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <SOFT_PROMPT_START>')
        for p in self.generator.soft_prompt_end_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <SOFT_PROMPT_END>')
        for p in self.generator.rerank_token_embedding_layer.parameters():
            p.requires_grad = False
        print(f'Frozen <RANK>')
        for p in self.generator.generate_model.parameters():
            p.requires_grad = False
        print(f'Frozen generator')

    def forward(self, encoder_input_ids, encoder_attention_mask, generator_input_ids, generator_attention_mask):
        with torch.no_grad():
            embeddings = self.encoder(encoder_input_ids, encoder_attention_mask)
            B = embeddings.shape[0]
            D = embeddings.shape[1] // self.num_emb_tokens
            embeddings = embeddings.reshape([B, self.num_emb_tokens, D])

            # project_embeddings: [batch_size * num_doc_tokens, hidden_size]
            project_embeddings = self.projector(embeddings)
            project_embeddings = project_embeddings.reshape([embeddings.shape[0] * self.num_doc_tokens, -1])
            
            output = self.generator.generate(project_embeddings, generator_input_ids, generator_attention_mask)
            formatted_output = []
            for o in output:
                try:
                    o = o[o.find('<answer>') + len('<answer>'):o.find('</answer>')]
                except:
                    o = None
                    # 静默失败:解析 <answer>...</answer> 标签时，如果生成结果格式不对，except 会把该条结果设为 None 而不报错
                formatted_output.append(o)

        return formatted_output




# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Joint Alignment Training
#
# Trainable:
#   · Encoder LoRA  (LoRA_φ, freshly added on top of Stage 1 encoder)
#   · Projector MLP (full fine-tuning, initialized from Stage 1 weights)
#   · Generator LoRA (θ_2 from Stage 2, continued fine-tuning)
#
# Frozen:
#   · Encoder base weights
#   · Encoder <ENCODE> special token embedding
#   · Generator base weights
#   · Generator special token embeddings (<SOFT_PROMPT_START/END>)
#
# Loss (Phase 1 – Task Loss only):
#   L_task = -Σ log P_θ(a_t | Q, E, a_{<t})
#   Gradient path: L_task → Generator LoRA → Projector → Encoder LoRA
#   This is the fully differentiable path described in the design document §2.
# ─────────────────────────────────────────────────────────────────────────────
class Stage3JointModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_emb_tokens = args.num_emb_tokens
        self.num_doc_tokens  = args.num_doc_tokens

        self.encoder   = Encoder(args)
        self.generator = Generator(args)

        encoder_size   = self.encoder.encoder.embed_tokens.weight.shape[-1]
        generator_size = self.generator.generate_model.config.hidden_size

        self.projector = MLPProjector(
            encoder_size, generator_size, args.num_emb_tokens, args.num_doc_tokens
        ).to(torch.bfloat16)
        self.config = self.generator.generate_model.config

        print(f'[Stage3] Encoder dim  : {encoder_size}')
        print(f'[Stage3] Generator dim: {generator_size}')
        print(f'[Stage3] Projector    : {encoder_size * args.num_emb_tokens} → '
              f'{generator_size * args.num_doc_tokens}')

        # 1. Load Stage 1 selector (encoder base + projector + special embeddings)
        self._load_stage1_selector(args.stage1_checkpoint_dir)
        # 2. Load Stage 2 generator LoRA; is_trainable=True keeps adapter params grad-on
        self._load_stage2_generator_lora(args.stage2_checkpoint_dir)
        # 3. Freeze base params; set projector trainable
        self._set_trainable()
        # 4. Wrap encoder with a fresh LoRA adapter for Stage 3
        encoder_lora_r     = getattr(args, 'encoder_lora_r',     32)
        encoder_lora_alpha = getattr(args, 'encoder_lora_alpha', 16)
        self._make_encoder_lora(r=encoder_lora_r, lora_alpha=encoder_lora_alpha)
        # 5. Enable gradient checkpointing to prevent OOM (design doc §5)
        self._enable_gradient_checkpointing()

        compute_trainable_parameters(self)

    # ── checkpoint loading ───────────────────────────────────────────────────

    def _load_stage1_selector(self, checkpoint_dir):
        """Load Stage 1 trained selector: encoder weights, projector, special embeddings."""
        self.encoder.encoder.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'encoder.pt'), map_location='cpu')
        )
        self.encoder.encode_token_embedding_layer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'encode_token_embedding_layer.pt'), map_location='cpu')
        )
        self.projector.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'projector.pt'), map_location='cpu')
        )
        self.generator.soft_prompt_start_embedding_layer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'soft_prompt_start_embedding_layer.pt'), map_location='cpu')
        )
        self.generator.soft_prompt_end_embedding_layer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'soft_prompt_end_embedding_layer.pt'), map_location='cpu')
        )
        print(f'[Stage3] ✓ Stage 1 selector loaded from {checkpoint_dir}')

    def _load_stage2_generator_lora(self, stage2_checkpoint_dir):
        """Load Stage 2 generator LoRA adapter, keeping it trainable for Stage 3."""
        self.generator.generate_model = PeftModel.from_pretrained(
            self.generator.generate_model,
            stage2_checkpoint_dir,
            is_trainable=True   # ← allows continued fine-tuning in Stage 3
        )
        print(f'[Stage3] ✓ Stage 2 generator LoRA loaded from {stage2_checkpoint_dir} (trainable)')

    # ── parameter setup ──────────────────────────────────────────────────────

    def _set_trainable(self):
        """
        Freeze base encoder + all embedding layers.
        Unfreeze projector.
        Generator LoRA params are already trainable via PeftModel (is_trainable=True).
        Encoder LoRA params will be set trainable by _make_encoder_lora().
        """
        # Encoder base → frozen (LoRA will add trainable delta on top)
        for p in self.encoder.encoder.parameters():
            p.requires_grad = False
        # <ENCODE> special embedding → frozen (Stage 1 value, stable)
        for p in self.encoder.encode_token_embedding_layer.parameters():
            p.requires_grad = False
        # Projector → TRAINABLE (full MLP fine-tuning; use LR × 1.5 per design doc §5)
        for p in self.projector.parameters():
            p.requires_grad = True
        # Generator token embedding layer → frozen
        for p in self.generator.embedding_layer.parameters():
            p.requires_grad = False
        # Generator special token embeddings → frozen
        for p in self.generator.soft_prompt_start_embedding_layer.parameters():
            p.requires_grad = False
        for p in self.generator.soft_prompt_end_embedding_layer.parameters():
            p.requires_grad = False
        for p in self.generator.rerank_token_embedding_layer.parameters():
            p.requires_grad = False
        print('[Stage3] ✓ Trainability set: projector=trainable, bases=frozen')

    def _make_encoder_lora(self, r=32, lora_alpha=16):
        """Wrap encoder with a fresh LoRA adapter (Stage 3 LoRA_φ)."""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules='all-linear',
            lora_dropout=0.1,
            bias='none',
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder.encoder = get_peft_model(self.encoder.encoder, lora_config)
        # Note: self.encoder.embedding_layer still points to the base model's embedding
        # weights, which is correct — LoRA does not wrap embedding layers.
        print(f'[Stage3] ✓ Encoder LoRA added (r={r}, lora_alpha={lora_alpha})')

    def _enable_gradient_checkpointing(self, gc_kwargs=None):
        """Enable gradient checkpointing on generator + encoder to reduce peak VRAM.

        gc_kwargs is forwarded to each sub-model's gradient_checkpointing_enable so that
        settings like use_reentrant=False (specified in TrainingArguments) are actually
        applied.  Without this forwarding the kwarg is silently ignored and the older
        use_reentrant=True mode is used, which can break LoRA + DeepSpeed compute graphs.
        """
        gc_kwargs = gc_kwargs or {}
        for m in [self.generator.generate_model, self.encoder.encoder]:
            if hasattr(m, 'enable_input_require_grads'):
                m.enable_input_require_grads()
            if hasattr(m, 'gradient_checkpointing_enable'):
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        print('[Stage3] ✓ Gradient checkpointing enabled')

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Called by HF Trainer when gradient_checkpointing=True in TrainingArguments."""
        self._enable_gradient_checkpointing(gradient_checkpointing_kwargs)

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        generator_input_ids,
        generator_attention_mask,
        generator_labels,
    ):
        """
        Stage 3 forward pass — fully differentiable end-to-end (design doc §2).

        Unlike Stage 2 (which wrapped encoder/projector in torch.no_grad()),
        here the entire path is open to gradients:
            L_task  ──backprop──►  Generator LoRA
                    ──backprop──►  Projector (MLP)
                    ──backprop──►  Encoder LoRA

        This allows the selector to receive direct supervision from the
        downstream generation loss and self-correct its compression strategy.
        """
        # ── Encoder (gradient flows through LoRA layers) ────────────────────
        embeddings = self.encoder(encoder_input_ids, encoder_attention_mask)
        B = embeddings.shape[0]
        D = embeddings.shape[1] // self.num_emb_tokens
        embeddings = embeddings.reshape([B, self.num_emb_tokens, D])
        # shape: [B, num_emb_tokens, encoder_hidden]

        # ── Projector (gradient flows through MLP) ──────────────────────────
        project_embeddings = self.projector(embeddings)
        project_embeddings = project_embeddings.reshape([B * self.num_doc_tokens, -1])
        # shape: [B * num_doc_tokens, generator_hidden]

        # ── Generator – L_task (autoregressive CE on answer tokens) ─────────
        loss = self.generator(
            project_embeddings, generator_input_ids, generator_attention_mask, generator_labels
        )
        return {'loss': loss}

    # ── save ─────────────────────────────────────────────────────────────────

    def save(self, output_dir):
        """
        Save all updated components.

        Directory layout:
          output_dir/
            encoder_lora/       ← Stage 3 encoder LoRA adapter
            generator_lora/     ← Stage 2→3 generator LoRA adapter (merged history)
            projector.pt        ← updated projector weights
            frozen/             ← frozen artifacts required at inference time
              encode_token_embedding_layer.pt
              soft_prompt_start_embedding_layer.pt
              soft_prompt_end_embedding_layer.pt
        """
        os.makedirs(output_dir, exist_ok=True)

        # Encoder LoRA
        self.encoder.encoder.save_pretrained(os.path.join(output_dir, 'encoder_lora'))
        print(f'[Stage3] Saved encoder LoRA  → {output_dir}/encoder_lora')

        # Generator LoRA
        self.generator.generate_model.save_pretrained(os.path.join(output_dir, 'generator_lora'))
        print(f'[Stage3] Saved generator LoRA → {output_dir}/generator_lora')

        # Projector
        torch.save(self.projector.state_dict(), os.path.join(output_dir, 'projector.pt'))
        print(f'[Stage3] Saved projector      → {output_dir}/projector.pt')

        # Frozen inference artifacts (needed when loading at eval time)
        frozen_dir = os.path.join(output_dir, 'frozen')
        os.makedirs(frozen_dir, exist_ok=True)
        torch.save(
            self.encoder.encode_token_embedding_layer.state_dict(),
            os.path.join(frozen_dir, 'encode_token_embedding_layer.pt')
        )
        torch.save(
            self.generator.soft_prompt_start_embedding_layer.state_dict(),
            os.path.join(frozen_dir, 'soft_prompt_start_embedding_layer.pt')
        )
        torch.save(
            self.generator.soft_prompt_end_embedding_layer.state_dict(),
            os.path.join(frozen_dir, 'soft_prompt_end_embedding_layer.pt')
        )
        print(f'[Stage3] Saved frozen artifacts → {output_dir}/frozen')


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Inference / Evaluation Model
#
# Loads:
#   · Encoder base weights      from stage1_checkpoint_dir (encoder.pt)
#   · Encoder LoRA              from stage3_checkpoint_dir/encoder_lora/
#   · Projector                 from stage3_checkpoint_dir/projector.pt
#   · Frozen special embeddings from stage3_checkpoint_dir/frozen/
#   · Generator LoRA            from stage3_checkpoint_dir/generator_lora/
#
# All parameters are frozen for inference.
# ─────────────────────────────────────────────────────────────────────────────
class Stage3TestModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_emb_tokens = args.num_emb_tokens
        self.num_doc_tokens  = args.num_doc_tokens

        self.encoder   = Encoder(args)
        self.generator = Generator(args)

        encoder_size   = self.encoder.encoder.embed_tokens.weight.shape[-1]
        generator_size = self.generator.generate_model.config.hidden_size

        self.projector = MLPProjector(
            encoder_size, generator_size, args.num_emb_tokens, args.num_doc_tokens
        ).to(torch.bfloat16)
        self.config = self.generator.generate_model.config

        stage3_dir = args.stage3_checkpoint_dir
        frozen_dir = os.path.join(stage3_dir, 'frozen')

        # 1. Encoder base weights (Stage 1, frozen in Stage 3)
        self.encoder.encoder.load_state_dict(
            torch.load(os.path.join(args.stage1_checkpoint_dir, 'encoder.pt'), map_location='cpu')
        )
        print(f'[Stage3 Eval] Encoder base loaded from {args.stage1_checkpoint_dir}')

        # 2. Encoder LoRA (Stage 3 adapter)
        self.encoder.encoder = PeftModel.from_pretrained(
            self.encoder.encoder,
            os.path.join(stage3_dir, 'encoder_lora'),
            is_trainable=False
        )
        print(f'[Stage3 Eval] Encoder LoRA loaded from {stage3_dir}/encoder_lora')

        # 3. Projector (Stage 3 updated weights)
        self.projector.load_state_dict(
            torch.load(os.path.join(stage3_dir, 'projector.pt'), map_location='cpu')
        )
        print(f'[Stage3 Eval] Projector loaded from {stage3_dir}/projector.pt')

        # 4. Frozen special-token embeddings (saved as Stage 1 values, unchanged in Stage 3)
        self.encoder.encode_token_embedding_layer.load_state_dict(
            torch.load(os.path.join(frozen_dir, 'encode_token_embedding_layer.pt'), map_location='cpu')
        )
        self.generator.soft_prompt_start_embedding_layer.load_state_dict(
            torch.load(os.path.join(frozen_dir, 'soft_prompt_start_embedding_layer.pt'), map_location='cpu')
        )
        self.generator.soft_prompt_end_embedding_layer.load_state_dict(
            torch.load(os.path.join(frozen_dir, 'soft_prompt_end_embedding_layer.pt'), map_location='cpu')
        )
        print(f'[Stage3 Eval] Frozen embeddings loaded from {frozen_dir}')

        # 5. Generator LoRA (Stage 2→3 merged adapter)
        self.generator.generate_model = PeftModel.from_pretrained(
            self.generator.generate_model,
            os.path.join(stage3_dir, 'generator_lora'),
            is_trainable=False
        )
        print(f'[Stage3 Eval] Generator LoRA loaded from {stage3_dir}/generator_lora')

        # 6. Freeze everything for inference
        for p in self.parameters():
            p.requires_grad = False

        compute_trainable_parameters(self)

    def forward(self, encoder_input_ids, encoder_attention_mask, generator_input_ids, generator_attention_mask):
        with torch.no_grad():
            embeddings = self.encoder(encoder_input_ids, encoder_attention_mask)
            B = embeddings.shape[0]
            D = embeddings.shape[1] // self.num_emb_tokens
            embeddings = embeddings.reshape([B, self.num_emb_tokens, D])

            project_embeddings = self.projector(embeddings)
            project_embeddings = project_embeddings.reshape([B * self.num_doc_tokens, -1])

            output = self.generator.generate(project_embeddings, generator_input_ids, generator_attention_mask)
            formatted_output = []
            for o in output:
                try:
                    o = o[o.find('<answer>') + len('<answer>'):o.find('</answer>')]
                except:
                    o = None
                formatted_output.append(o)

        return formatted_output
