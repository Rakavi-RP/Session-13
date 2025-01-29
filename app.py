import gradio as gr
import torch
import os
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from model import SmolLM2

# Get token from environment variable
HF_TOKEN = os.environ.get('HF_TOKEN')

# Load model and tokenizer
def load_model():
    config = {
        "vocab_size": 49152,
        "hidden_size": 576,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "intermediate_size": 1536,
        "rms_norm_eps": 1e-5,
        "initializer_range": 0.041666666666666664,
    }
    
    model = SmolLM2(config)
    
    # Download checkpoint from hub using your token
    checkpoint_path = hf_hub_download(
        repo_id="Rakavi12/smollm2-checkpoint",
        filename="checkpoint.pt",
        token=HF_TOKEN
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    return model, tokenizer

model, tokenizer = load_model()

def generate_text(prompt, max_length=100, temperature=0.7):
    try:
        print(f"Starting generation with prompt: {prompt}")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        print(f"Encoded input shape: {input_ids.shape}")
        
        with torch.no_grad():
            print("Generating...")
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                top_k=50,
                top_p=0.95
            )
            print(f"Generation complete. Output shape: {output_ids.shape}")
        
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Decoded text: {text}")
        return text
    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        return f"Error generating text: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt", value="There was a time"),
        gr.Slider(minimum=10, maximum=200, value=100, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Text Generation",
    description="Enter a prompt to generate text using SmolLM2"
)

iface.launch() 