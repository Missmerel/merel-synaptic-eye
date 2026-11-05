import torch
import torch.nn as nn
import os

# ==========================================
# 1. LOAD THE MODEL ARCHITECTURE
# ==========================================
# We need the exact same architecture to load the weights
block_size = 128
device = 'cpu'

# (Assuming vocab_size is the same, we need to load it from the data)
import re
with open('my_personal_data.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()
words = re.findall(r'\w+|[^\w\s]', text)
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

class MerelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 256)
        self.position_embedding = nn.Embedding(block_size, 256)
        self.lm_head = nn.Linear(256, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)
        return logits, x

# ==========================================
# 2. LOAD THE FLOAT32 BRAIN
# ==========================================
print("Loading original brain (float32)...")
model = MerelModel()
model.load_state_dict(torch.load('merel_words.pth', map_location=device))
model.eval()

# ==========================================
# 3. QUANTIZE TO INT8
# ==========================================
print("Quantizing to int8...")
# PyTorch has a built-in dynamic quantization tool

# Force the ARM-compatible quantization engine
torch.backends.quantized.engine = 'qnnpack'
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear}, # ONLY quantize the Linear layers
    dtype=torch.qint8
)

# ==========================================
# 4. SAVE THE OPTIMIZED BRAIN
# ==========================================
torch.save(quantized_model.state_dict(), 'merel_words_int8.pth')
print("Quantization complete! Saved to merel_words_int8.pth")

# ==========================================
# 5. COMPARE FILE SIZES
# ==========================================
original_size = os.path.getsize('merel_words.pth') / (1024 * 1024)
new_size = os.path.getsize('merel_words_int8.pth') / (1024 * 1024)
print(f"Original size: {original_size:.2f} MB")
print(f"New size:      {new_size:.2f} MB")
print(f"Reduction:     {original_size / new_size:.1f}x smaller")
