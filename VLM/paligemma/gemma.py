import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math

class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0 
        else:
            # The shape of the key_cache is (batch_size, num_heads_KV, seq_len, head_dim)
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # if we nevel added anything to the KV-Cache of this layer, create it
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: (batch_size, num_heads_KV, seq_len, head_dim)
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # and then we return all the existing keys + he new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_key_value_heads, q_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, q_len, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, q_len, head_dim)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim 
        self.max_position_embeddings = max_position_embeddings 
        self.base = base 

        # calculate the theta accordint to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        self.inv_freq.to(x.device)
        # copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: (batch_size, head_dim // 2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: (batch_size, 1 ,seq_len)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type 
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # for mixed precision computation to retain full float 32 computation
        with torch.autocast(device_type=device_type, enabled=False):
            # multiply each theta by the position (sin, cos)
            # freqs: (batch_size, head_dim // 2, 1) @ (batch_size, 1, seq_len) -> (batch_size, seq_len, head_dim // 2)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            # emb: (batch_size, seq_len, head_dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: (batch_size, seq_len, head_dim)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    # build the (-x2, x1, -x4, x3, ...) tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Take the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rorate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads= num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias = False)

    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config 
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads # GQA/MQA less heads for k and v
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True 

        assert self.hidden_size % self.num_heads == 0

        # number of heads = 8
        # hidden_size = 1024
        # head_dim = 1024 // 8 = 128
        # wq: (1024, 8 * 128) = (1024, 1024)
        # wk: (1024, 2 * 128) = (1024, 256)
        # wv: (1024, 2 * 128) = (1024, 256)
        # GQA: balance performance / efficiency, reduce HBM memory transfer to speed up, reduce KV cache to save memory
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # (batch_size, seq_len, num_heads_q * head_dim)
        batch_size, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # batch_size, num_heads_q, seq_len, head_dim
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1,2)
        # batch_size, num_heads_KV, seq_len, head_dim
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        # batch_size, num_heads_KV, seq_len, head_dim
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # repeat the key and values to match the number of heads of the query
        # we didn't customize the cuda kernel for the computation of the attention, 
        # so we have to have same number of key_states and value_states as query_states
        # perhaps use flash attention to actually leverage the reduced number of heads
        # or perhaps use the same idea of nn.Conv with group
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None 
        attn_weights = attn_weights + attention_mask 

        # apply the softmax
        # (batch_size, num_heads_q, se_len_q, seq_len_KV)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # multiply by the values
        # (batch_size, num_heads_q, se_len_q, seq_len_KV) x (batch_size, num_heads_KV, seq_len_KV, head_dim) -> (batch_size, num_heads_q, se_len_q, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output_size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states 
        hidden_states = self.input_layernorm(hidden_states)
        # (batch_size, seq_len, hidden_size)

        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        self.config = config 
        self.padding_idx = config.pad_token_id 
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # (batch_size, seq_len, hidden_size)
        hidden_states = inputs_embeds
        # (batch_size, seq_len, hidden_size)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # (batch_size, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states)

        # (batch_size, seq_len, hidden_size)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, 
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input_embeds: (batch_size, seq_len, hidden_size)
        # outputs: (batch_size, seq_len, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids = position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        hidden_sttes = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {"logits": logits}

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data
