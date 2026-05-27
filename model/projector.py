import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, encoder_size, generator_size, num_emb_tokens, num_doc_tokens,
                 hidden_dim=None, use_ln=True):
        super().__init__()
        self.encoder_size = encoder_size
        self.generator_size = generator_size
        self.num_emb_tokens = num_emb_tokens
        self.num_doc_tokens = num_doc_tokens

        in_dim = encoder_size * num_emb_tokens
        out_dim = generator_size * num_doc_tokens
        self.fc = nn.Linear(in_dim, out_dim) # 单层线性

    def forward(self, encoder_hidden):
        B, N, D = encoder_hidden.shape
        assert N == self.num_emb_tokens and D == self.encoder_size

        x = encoder_hidden.reshape(B, -1)
        x = self.fc(x)
        x = x.view(B, self.num_doc_tokens, self.generator_size)
        return x

# projector 同时承担两件事：
# 1.Hidden 维度对齐：encoder_size → generator_size（不同模型一般不同）。
# 2.Token 数量重塑：从 encoder 端的 num_emb_tokens 个摘要 token 变成
#   generator 端的 num_doc_tokens 个软提示 token。二者彼此解耦。