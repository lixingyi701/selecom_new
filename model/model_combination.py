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
from util.util import *
from util.constant import IGNORE_TOKEN_ID
from peft import LoraConfig, get_peft_model, PeftModel


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

        


