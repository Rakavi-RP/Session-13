import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / norm) * self.weight

class RotaryEmbedding(nn.Module):
    
    def __init__(self, dim, max_position=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_position = max_position
        
        
        position = torch.arange(max_position).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', position, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(1))  
        self.register_buffer('sin_cached', emb.sin().unsqueeze(1)) 

    def forward(self, x, seq_len=None):
        if seq_len > self.max_position:
            seq_len = self.max_position
        return (
            self.cos_cached[:seq_len].unsqueeze(0), 
            self.sin_cached[:seq_len].unsqueeze(0)   
        )
        
def rotate_half(x):       
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.expand(q.shape[0], -1, -1, -1)  
    sin = sin.expand(k.shape[0], -1, -1, -1)  
    
    q_embed = (q * cos) + (torch.roll(q, shifts=1, dims=-1) * sin)
    k_embed = (k * cos) + (torch.roll(k, shifts=1, dims=-1) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        
        self.q_proj = nn.Linear(config["hidden_size"], self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config["hidden_size"], bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)

        # Repeat k/v heads to match number of query heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, None)

       
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)

class FeedForward(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)
        self.act_fn = nn.SiLU()  # From config hidden_act: silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.ffn_norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.attention_norm(x), attention_mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class SmolLM2(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        
        
        self.apply(self._init_weights)
        self.gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config["initializer_range"]
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, attention_mask)
            
        x = self.norm(x)
        
        logits = F.linear(x, self.embed_tokens.weight)
        return logits

    def generate(
        self, 
        input_ids, 
        max_length, 
        temperature=0.8, 
        do_sample=True, 
        pad_token_id=None,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        top_k=50,
        top_p=0.95
    ):
        self.eval()
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for previous_token in set(input_ids[i].tolist()):
                            next_token_logits[i, previous_token] /= repetition_penalty
                
                # Apply n-gram blocking
                if no_repeat_ngram_size > 0:
                    for i in range(batch_size):
                        ngrams = set()
                        input_list = input_ids[i].tolist()
                        for j in range(len(input_list) - no_repeat_ngram_size + 1):
                            ngram = tuple(input_list[j:j + no_repeat_ngram_size])
                            ngrams.add(ngram)
                        for ngram in ngrams:
                            if len(input_list) >= no_repeat_ngram_size:
                                banned_tokens = ngram[-1:]
                                for token in banned_tokens:
                                    next_token_logits[i, token] = float('-inf')
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                
                if pad_token_id is not None and (next_tokens == pad_token_id).all():
                    break
                
        return input_ids

def create_model_from_config(config):
    
    model_config = {
        "vocab_size": config["model"]["model_config"]["vocab_size"],
        "hidden_size": config["model"]["model_config"]["hidden_size"],
        "num_hidden_layers": config["model"]["model_config"]["num_hidden_layers"],
        "num_attention_heads": config["model"]["model_config"]["num_attention_heads"],
        "num_key_value_heads": config["model"]["model_config"]["num_key_value_heads"],
        "intermediate_size": config["model"]["model_config"]["intermediate_size"],
        "rms_norm_eps": config["model"]["model_config"]["rms_norm_eps"],
        "initializer_range": config["model"]["model_config"]["initializer_range"],
    }
    return SmolLM2(model_config)

def print_model_summary(model, config):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("SmolLM2-135M Model Summary:")
    print("="*50)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (MB): {total_params * 4 / (1024 * 1024):.2f}")
    print("\nArchitecture:")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Layers: {config['num_hidden_layers']}")
    print(f"Attention Heads: {config['num_attention_heads']}")
    print(f"KV Heads: {config['num_key_value_heads']}")
    print("="*50 + "\n")