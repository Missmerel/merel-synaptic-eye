import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
torch.set_num_threads(1)

# ==========================================
# 1. HYPERPARAMETERS (The knobs you can tune)
# ==========================================
batch_size = 32      # How many sequences to process at once
block_size = 64      # The maximum context length (how many characters it looks back)
max_iters = 3000     # How many training steps to run
learning_rate = 1e-3 # How fast the model learns
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if you have one!
print(f"Using device: {device}")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
# Download a dataset (Shakespeare's works)
# To this:
with open('my_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers (Tokenization)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert the entire text to integers
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

# Function to get batches of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ==========================================
# 3. THE NEURAL NETWORK (The Brain)
# ==========================================
class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(vocab_size, head_size, bias=False)
        self.query = nn.Linear(vocab_size, head_size, bias=False)
        self.value = nn.Linear(vocab_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x) # (B,T,C)
        out = wei @ v     # (B,T,C)
        return out

class MerelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, vocab_size)
        self.sa_heads = Head(vocab_size)
        self.lm_head = nn.Linear(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x)

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
# 4. THE TRAINING LOOP
# ==========================================
print("Starting training...")
for iter in range(max_iters):
    if iter % 10 == 0:
        print(f"Working on step {iter}...")

    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Print progress every 500 steps
    if iter % 500 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

# ==========================================
# 5. GENERATE TEXT (Test the model)
# ==========================================
print("\n--- MEREL SPEAKS ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with a newline
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
# Save the model so startup.py can load it
torch.save(model.state_dict(), 'merel.pth')
print("Brain saved to merel.pth")
