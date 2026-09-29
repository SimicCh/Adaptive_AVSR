import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, **kwargs):
        head_shape = x.shape[:-1]
        batchSize, sequSize, featureSize = x.shape
        x = self.linear(x)
        x_pre = x.clone()
        # x = x.view(batchSize, sequSize, self.heads, self.d_k).transpose(1, 2)
        x = x.reshape(batchSize, sequSize, self.heads, self.d_k).transpose(1, 2)
        # ################################################################
        # # # Check that x_pre corresponds to x after reshaping
        # max_diff = 0
        # for b in range(batchSize):
        #     for c in range(x.shape[1]):
        #         x_pre_chunk = x_pre[b,:,c*self.d_k:(c+1)*self.d_k]
        #         x_chunk = x[b,c].reshape(sequSize, self.d_k)
        #         diff = torch.abs(x_pre_chunk - x_chunk).max().item()
        #         max_diff = max(max_diff, diff)
        # print(f"Max diff between x_pre and x chunks: {max_diff}")
        return x


class MultiHeadAttention_oldVersion(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()

        self.heads = heads
        self.d_k = d_model#  // heads
        self.dropout_prob = dropout_prob
        self.bias = bias

        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(self.d_k*self.heads, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None
    

    # def get_scores(self, query: torch.Tensor, key: torch.Tensor):
    #     return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[1]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(1)
        return mask
    
    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        # #################################################################
        # print(q.shape, k.shape, v.shape, flush=True)
        # import matplotlib.pyplot as plt
        # for b in range(q.shape[0]):
        #     plt.imshow(q[b,0].transpose(-2,-1).cpu().detach().numpy())
        #     plt.colorbar()
        #     plt.savefig(f'q_batch{b}.png', dpi=300)
        #     plt.close()
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            min_value = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(mask == 0, min_value)
        attention = F.softmax(attn_logits, dim=-1)

        # for b in range(attention.shape[0]):
        #     plt.imshow(attention[b,0].cpu().detach().numpy())
        #     plt.colorbar()
        #     plt.savefig(f'attention_batch{b}.png', dpi=300)
        #     plt.close()
        # print(aslkjd)
        #################################################################
        # # Check attention visually
        # import matplotlib.pyplot as plt
        # plt.imshow(mask[10,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(attn_logits[10,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(attention[10,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                **kwargs):
        batch_size, seq_len, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        query = self.query(query, **kwargs)
        key = self.key(key, **kwargs)
        value = self.value(value, **kwargs)
        values, attn = self.scaled_dot_product(query, key, value, mask)
        x = values.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # #################################################
        # # # Check that values correspond to chunks of x
        # max_diff = 0
        # for b in range(x.shape[0]):
        #     for h in range(values.shape[1]):
        #         chunk_size = x.shape[2] // values.shape[1]
        #         value_chunk = values[b,h]
        #         x_chunk = x[b,:,h*chunk_size:(h+1)*chunk_size]
        #         diff = torch.abs(value_chunk - x_chunk).max().item()
        #         max_diff = max(max_diff, diff)
        # print(f"Max diff between values and x chunks: {max_diff}")
        x = self.output(x)
        return x, attn.mean(dim=1)
