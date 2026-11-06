import torch
import torch.nn as nn
import os
import re

# ==========================================
# 1. SETUP
# ==========================================
torch.set_num_threads(1)
block_size = 128
device = 'cpu'
torch.backends.quantized.engine = 'qnnpack'

# Load vocab (same as train_words.py)
with open('my_personal_data.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()
words = re.findall(r'\w+|[^\w\s]', text)
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

# ==========================================
# 2. THE NEW MODEL ARCHITECTURE
# ==========================================
class MerelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.north_embedding = nn.Embedding(vocab_size, 128)
        self.south_embedding = nn.Embedding(vocab_size, 128)
        self.position_embedding = nn.Embedding(block_size, 256)
        self.facing_projection = nn.Linear(256, 3)
        self.lm_head = nn.Linear(256, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        north = self.north_embedding(idx)
        south = self.south_embedding(idx)
        combined = torch.cat([north, south], dim=-1)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = combined + pos_emb
        facing = self.facing_projection(x)
        agreement = torch.sum(north * south, dim=-1, keepdim=True)
        x = x * torch.sigmoid(agreement)
        logits = self.lm_head(x)
        return logits, x

# ==========================================
# 3. LOAD AND QUANTIZE
# ==========================================
print("Loading original Entangled brain (float32)...")
model = MerelModel()
model.load_state_dict(torch.load('merel_words.pth', map_location=device))
model.eval()

print("Quantizing to int8...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear}, # Only quantize Linear layers
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 'merel_words_int8.pth')
print("Quantization complete! Saved to merel_words_int8.pth")

# Compare sizes
original_size = os.path.getsize('merel_words.pth') / (1024 * 1024)
new_size = os.path.getsize('merel_words_int8.pth') / (1024 * 1024)
print(f"Original size: {original_size:.2f} MB")
print(f"New size:      {new_size:.2f} MB")
print(f"Reduction:     {original_size / new_size:.1f}x smaller")
