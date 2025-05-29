import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification
import torch.nn.functional as F


class LocalSparseAttention(nn.Module):
    def __init__(self, hidden_size, window_size, sparsity_threshold):
        super(LocalSparseAttention, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.sparsity_threshold = sparsity_threshold
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.LeakyReLU()
    def forward(self, x):

        num_windows = x.size(1) // self.window_size
        attention_outputs = []

        for i in range(num_windows):
            window_x = x[:, i * self.window_size:(i + 1) * self.window_size, :]

            query = self.query_layer(window_x)
            key = self.key_layer(window_x)
            value = self.value_layer(window_x)

            attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_size)

            # 动态计算 top-k
            dynamic_top_k = max(1, int(attn_scores.size(-1) * self.sparsity_threshold))


            # 对每一行单独进行 topk 选择
            top_k_scores, top_k_indices = torch.topk(attn_scores, dynamic_top_k, dim=-1)
            top_k_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
            top_k_mask.scatter_(-1, top_k_indices, True)

            # 将非 top-k 的注意力分数置为负无穷
            attn_scores_masked = attn_scores.masked_fill(~top_k_mask, float('-inf'))

            # 数值稳定性处理
            attn_scores_masked = attn_scores_masked - attn_scores_masked.max(dim=-1, keepdim=True)[0]
            # 对 top-k 的得分进行 softmax
            attn_weights = F.softmax(attn_scores_masked, dim=-1)

            # 加权求和
            attention_output = torch.matmul(attn_weights, value)
            attention_outputs.append(attention_output)

        output = torch.cat(attention_outputs, dim=1)
        output = self.activation(output)
        output = self.norm(output + x)

        return output


class GlobalLocalEnhancedMoudle(nn.Module):
    def __init__(self, hidden_size, seq_len, num_windows, sparsity_threshold,local_num,drop_n):
        super(GlobalLocalEnhancedMoudle, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.local_num = local_num
        self.window_size = max(2, self.seq_len // num_windows)
        self.local_attention = nn.ModuleList()
        for i in range(self.local_num):
            self.local_attention.append(LocalSparseAttention(hidden_size, self.window_size, sparsity_threshold))
        self.dropout = nn.Dropout(drop_n)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8),
            num_layers=1
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x, cls_token):
        padding = 1
        x = F.pad(x, (0, 0, 0, padding), mode='constant', value=0)
        for i in range(self.local_num):
            x = self.local_attention[i](x)
            x = self.dropout(x)
        x = x[:, :-padding, :]
        combined = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        output = self.transformer_encoder(combined)
        output = self.activation(output)
        output = self.norm(output)
        return output


class TextEncoder(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer, args):
        super(TextEncoder, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.local_num = args.t_num
        self.drop_n=args.drop_n
        self.gl_enhencer = GlobalLocalEnhancedMoudle(hidden_size=768, seq_len=512, num_windows=32,
                                                     sparsity_threshold=args.ts, local_num=self.local_num, drop_n=self.drop_n)
        self.dropout = nn.Dropout(self.drop_n)
    def forward(self, input_ids, output_attentions=False):
        sourcecode_outputs = \
        self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
        sourcecode_cls = sourcecode_outputs[:, 0, :]
        sourcecode_other = sourcecode_outputs[:, 1:, :]
        output = self.gl_enhencer(sourcecode_other, sourcecode_cls)
        output = self.dropout(output)
        return output
