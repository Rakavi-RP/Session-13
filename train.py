import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from torch.cuda.amp import autocast, GradScaler
import math
from model import SmolLM2
from datasets import load_dataset
import sys
import torch.nn.functional as F


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Config dictionary
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
            "initializer_range": 0.041666666666666664,
        }
    },
    "optimizer": {
        "learning_rate_scheduler": {
            "learning_rate": 6e-4,
            "lr_warmup_steps": 100,
        },
        "weight_decay": 0.01
    },
    "tokens": {
        "micro_batch_size": 32
    }
}


SAVE_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 500  
os.makedirs(SAVE_DIR, exist_ok=True)

def setup_training():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", 
        "cosmopedia-v2", 
        streaming=True, 
        split="train"
    )
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        # Add labels (same as input_ids for causal language modeling)
        outputs['labels'] = outputs['input_ids'].clone()
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config["tokens"]["micro_batch_size"],
        shuffle=False,  # Keep shuffle False for streaming dataset
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return tokenizer, dataloader

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7):
    device = next(model.parameters()).device
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text

def save_checkpoint(model, optimizer, scheduler, step, loss):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, f"{SAVE_DIR}/checkpoint.pt")

def load_checkpoint(model, optimizer, scheduler):
    path = f"{SAVE_DIR}/checkpoint.pt"
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step']
    return 0

def train_loop(model, tokenizer, total_steps, start_step=0):
    device = next(model.parameters()).device
    tokenizer, train_loader = setup_training()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate_scheduler"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["optimizer"]["learning_rate_scheduler"]["lr_warmup_steps"],
        num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    step = start_step
    train_iterator = iter(train_loader)
    
    try:
        while step < total_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(input_ids, attention_mask=attention_mask)
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            if step % 1 == 0:
                print(f"Step [{step}/{total_steps}] Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if step % CHECKPOINT_INTERVAL == 0:
                print("\nGenerating sample text...")
                sample_text = generate_text(model, tokenizer, "There was a time")
                print(f"Generated: {sample_text}\n")
                save_checkpoint(model, optimizer, scheduler, step, loss.item())
            
            step += 1
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        save_checkpoint(model, optimizer, scheduler, step, loss.item())

def train():
    print("Starting Phase 1: Training for 5000 steps...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SmolLM2(config["model"]["model_config"])
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    
    # Train for 5000 steps
    train_loop(model, tokenizer, total_steps=5000)
    print("Phase 1 complete - Step 0 to 4999 -> 5000 steps")

def train_phase2():
    print("Loading checkpoint...")
    print("Starting Phase 2: Training for 50 more steps - Step 5000 to 5049 -> 50 steps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SmolLM2(config["model"]["model_config"])
    model.to(device)
    
    # Load 5000-step checkpoint
    checkpoint = torch.load(f"{SAVE_DIR}/checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    
    # Train for 50 more steps
    train_loop(model, tokenizer, total_steps=5049, start_step=5000)
    print("Phase 2 complete - Step 5000 to 5049 -> 50 steps")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "phase2":
        train_phase2()
    else:
        train() 