import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
import os
from dataclasses import dataclass
from app_model import GPT

# ---- define GPTConfig before loading checkpoint ----
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
# ----------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ckpt_path = "gpt_final_checkpoint_trimmed.pt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Checkpoint file '{ckpt_path}' not found. Upload it to your Space root folder."
    )

# ✅ explicitly allow full unpickling (safe since it's your own checkpoint)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

config = checkpoint["config"]
model = GPT(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")

def generate_text(prompt, max_tokens=50, temperature=0.8, top_k=50):
    if not prompt.strip():
        return "Please enter a prompt."
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = tokens[:, -config.block_size :]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, idx_next], dim=1)
    return enc.decode(tokens[0].tolist())


demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", value="Once upon a time", lines=3),
        gr.Slider(10, 200, value=50, step=10, label="Max new tokens"),
        gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(0, 100, value=50, step=10, label="Top-K sampling"),
    ],
    outputs=gr.Textbox(label="Generated text", lines=10),
    title="GPT-124M Model",
    description="Inference demo using trained GPT checkpoint.",
    examples=[
        ["CORIOLANUS: What is the city but the people?"],
        ["MENENIUS: There was a time when all the body's members rebelled against the belly—"],
        ["CITIZEN: We have power in ourselves to do it, but it is a power that we have no power to do."],
        ["O pride, thou noble fault that ruins noble men!"],
        ["War’s trumpet sounds, and peace withdraws her hand."]
    ],
    examples_per_page=5,
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()