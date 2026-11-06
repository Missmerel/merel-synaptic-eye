import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import sys

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================
torch.set_num_threads(1)
block_size = 128
batch_size = 16
max_iters = 30000
learning_rate = 1e-3
device = 'cpu'
print(f"Using device: {device}")

print("Loading personal dataset...")
with open('my_personal_data.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

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
# 2. THE MODEL (ENTANGLED PAIR ARCHITECTURE)
# ==========================================
class MerelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.north_embedding = nn.Embedding(vocab_size, 128)
        self.south_embedding = nn.Embedding(vocab_size, 128)
        self.position_embedding = nn.Embedding(block_size, 256)
        self.facing_projection = nn.Linear(256, 3)
        self.lm_head = nn.Linear(256, vocab_size)

    def forward(self, idx, targets=None):
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

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        # RETURN 4 THINGS: logits, loss, facing, hidden_state
        return logits, loss, facing, x

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # Ignore the 3 extra returns
            logits, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 3. INITIALIZE MODEL AND OPTIMIZER
# ==========================================
model = MerelModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("Starting training...")
for iter in range(max_iters):
    xb, yb = get_batch('train')

    # Unpack 4 values
    logits, loss, facing, hidden_state = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 10 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")
        sys.stdout.flush()

# Save the brain
torch.save(model.state_dict(), 'merel_words.pth')
print("Brain saved to merel_words.pth!")

print("\n--- MEREL SPEAKS ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=20)[0].tolist()))
