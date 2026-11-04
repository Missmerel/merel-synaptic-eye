import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import re
import sys

# Force single thread to prevent Mac CPU deadlocks
torch.set_num_threads(1)

# ==========================================
# 1. SETUP
# ==========================================
batch_size = 16
block_size = 64
max_iters = 30000
learning_rate = 1e-3 # Increased learning rate for faster training
device = 'cpu'
print(f"Using device: {device}")

print("Loading personal dataset...")
with open('my_personal_data.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

# Word-level tokenization
words = re.findall(r'\w+|[^\w\s]', text)
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

stoi = { w:i for i,w in enumerate(vocab) }
itos = { i:w for i,w in enumerate(vocab) }
encode = lambda s: [stoi.get(w, 0) for w in re.findall(r'\w+|[^\w\s]', s.lower())]
decode = lambda l: ' '.join([itos[i] for i in l])

data = torch.tensor([stoi[w] for w in words], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]

def get_batch(split):
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 2. THE MODEL (Simplified to prevent Mac freeze)
# ==========================================
class MerelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 256)  # Changed from 64
        self.position_embedding = nn.Embedding(block_size, 256) # Changed from 64
        self.lm_head = nn.Linear(256, vocab_size) # Changed from 64

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb # Shape: (B, T, 64)

        logits = self.lm_head(x) # Shape: (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = MerelModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
print("Starting training...")
for iter in range(max_iters):
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 10 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")
        sys.stdout.flush() # Force print to screen

# Save the brain
torch.save(model.state_dict(), 'merel_words.pth')
print("Brain saved to merel_words.pth!")

print("\n--- MEREL SPEAKS ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=20)[0].tolist()))
