# SmolLM2-135M-Replica
This is a replica of the SmolLM2-135M model trained on the Cosmopedia-v2 dataset.

## Repository Structure

```
├── requirements.txt # Project dependencies
├── config_smollm.py # Configuration for SmolLM2-135M
├── model.py         # Model implementation
├── train.py         # Training script
├── app.py           # Gradio interface for model deployment
├── log1.txt         # Training log - phase 1
├── log2.txt         # Training log - phase 2
└── README.md        # Project documentation
```


#### Transformer Blocks (30 layers)
Each block contains:

1. **Grouped-Query Attention (GQA)**
   - 9 query heads, 3 key-value heads (3:1 ratio)
   - Head dimension: 64 (576/9)
   - Implementation:

   ```python
   class GroupedQueryAttention(nn.Module):
       def __init__(self, config):
           self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
           self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
           self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
           self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
   ```

2. **Feed-Forward Network**
   - Expansion ratio: 2.67x (1536/576)
   - SiLU activation
   - Implementation:
   ```python
   class FeedForward(nn.Module):
       def __init__(self, config):
           self.up_proj = nn.Linear(hidden_size, intermediate_size)
           self.gate_proj = nn.Linear(hidden_size, intermediate_size)
           self.down_proj = nn.Linear(intermediate_size, hidden_size)
   ```

### 2. Parameter Calculation

Total Parameters: 134,515,880

1. **Embeddings**: 28,311,552
   - Token embeddings: 49,152 × 576 = 28,311,552

2. **Each Transformer Layer**: 3,540,672
   - GQA:
     * Q projection: 576 × 576 = 331,776
     * K projection: 576 × 192 = 110,592
     * V projection: 576 × 192 = 110,592
     * O projection: 576 × 576 = 331,776
     * Total GQA: 884,736
   - Feed-Forward:
     * Up projection: 576 × 1536 = 884,736
     * Gate projection: 576 × 1536 = 884,736
     * Down projection: 1536 × 576 = 884,736
     * Total FFN: 2,654,208
   - RMSNorms (2 per layer):
     * Attention norm: 576
     * FFN norm: 576
     * Total norms: 1,152
   - Total per layer: 884,736 + 2,654,208 + 1,152 = 3,540,672
   - All layers (×30): 3,540,672 × 30 = 106,220,160

3. **Final Layer Norm**: 576

Total Parameter Count:
- Embeddings: 28,311,552
- Transformer Layers: 106,220,160
- Final Layer Norm: 576
- Total: 134,515,880 parameters

Note: Each RMSNorm layer has parameters equal to the hidden size (576) for scaling factors.

## Training Process

### Data Loading
```python
def setup_training():
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", 
        "cosmopedia-v2", 
        streaming=True, 
        split="train"
    )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
```

### Training Configuration
```python
config = {
    "model": {
        "model_config": {
            "vocab_size": 49152,
            "hidden_size": 576,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "intermediate_size": 1536,
            "rms_norm_eps": 1e-5,
        }
    },
    "optimizer": {
        "learning_rate": 6e-4,
        "weight_decay": 0.01,
        "lr_warmup_steps": 100
    },
    "tokens": {
        "micro_batch_size": 32
    }
}
```

### Optimization Techniques

1. **Memory Optimizations**

```python

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.backends.cuda.matmul.allow_tf32 = True
```

2. **Training Loop Optimizations**

- Mixed Precision Training

```python
scaler = GradScaler(enabled=torch.cuda.is_available())
with torch.amp.autocast('cuda'):
    outputs = model(input_ids, attention_mask=attention_mask)
```

- Gradient Checkpointing

```python
if self.gradient_checkpointing and self.training:
    x = torch.utils.checkpoint.checkpoint(
        layer,
        x,
        attention_mask,
        use_reentrant=False
    )
```

3. **Learning Rate Schedule**
- Warmup steps: 100
- Linear decay
```python
def get_lr_scheduler(optimizer, config):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)
```

### Training Process
1. Phase 1: Initial 5000 steps (0 to 4999)
2. Phase 2: Additional 50 steps with reduced batch size (5000 to 5049)
3. Checkpointing every 500 steps

### Training Logs
1. **Phase 1:** [Training Logs](log1.txt)
2. **Phase 2:** [Training Logs](log2.txt)

## Deployment
- Hosted on Hugging Face Spaces
- Uses Gradio for web interface
- Model checkpoint stored in Hugging Face Model Hub

### Hugging Face Spaces
- [Demo](https://huggingface.co/spaces/Rakavi12/smolLm2-135M-replica)

