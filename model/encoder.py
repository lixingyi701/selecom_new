import sys
sys.path.append('..')

import torch.nn.functional as F
import torch
import torch.nn as nn
from util.constant import *
from transformers import AutoModel, AutoTokenizer

	
class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()
		self.max_length = args.encoder_max_length
		self.num_emb_tokens = args.num_emb_tokens
		
		self.tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, padding_side='left', use_fast=True, trust_remote_code=True)
		self.tokenizer.add_tokens([ENCODE_TOKEN], special_tokens=True)
		self.encoder = AutoModel.from_pretrained(args.encoder_name, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
		
		self.encode_token_id = self.tokenizer.convert_tokens_to_ids(ENCODE_TOKEN)
		self.encoder.resize_token_embeddings(len(self.tokenizer))
		self.embedding_layer = self.encoder.get_input_embeddings()
		# Embedding layer for ENCODE token Initial
		self.encode_token_embedding_layer = nn.Embedding(1, self.encoder.config.hidden_size)
		self.encode_token_embedding_layer.weight.data = self.embedding_layer.weight[self.encode_token_id].unsqueeze(0)
	
	
	def forward(self, encoder_input_ids, encoder_attention_mask):
		encoder_input_ids = encoder_input_ids.to(self.encoder.device)
		encoder_attention_mask = encoder_attention_mask.to(self.encoder.device)

		# 第一步：通过主 embedding 查表（包括 ENCODE_TOKEN 对应行）
		input_embeds = self.embedding_layer(encoder_input_ids)

		input_embeds[encoder_input_ids == self.encode_token_id] = self.encode_token_embedding_layer.weight[0]
		# Replace ENCODE token embeddings with special embedding layer
		# 覆盖了第一步查出来的 ENCODE_TOKEN embedding，导致
		# 反向传播时：
		#   梯度 → input_embeds[ENCODE_TOKEN 位置]
		#        → encode_token_embedding_layer.weight[0]  ✅ 有梯度
		#        → embedding_layer.weight[encode_token_id]  ❌ 已被覆盖，梯度被截断
		# 之后 encode_token_embedding_layer.weight[0] 被训练更新并保存在 pt 并从中加载，就和初始化的一直不同了

		model_output = self.encoder(
			inputs_embeds=input_embeds, 
			attention_mask=encoder_attention_mask
		)

		output = model_output.last_hidden_state[encoder_input_ids == self.encode_token_id]
		output = output.reshape([-1, self.num_emb_tokens * output.shape[-1]])
		
		return output
