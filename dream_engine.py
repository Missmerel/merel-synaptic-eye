import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import time
import datetime
import os

# ==========================================
# 1. SETUP & DATA
# ==========================================
torch.set_num_threads(1)
block_size = 128
device = 'cpu'

with open('my_personal_data.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

words = re.findall(r'\w+|[^\w\s]', text)
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
stoi = { w:i for i,w in enumerate(vocab) }
itos = { i:w for i,w in enumerate(vocab) }
encode = lambda s: [stoi.get(w, 0) for w in re.findall(r'\w+|[^\w\s]', s.lower())]
decode = lambda l: ' '.join([itos[i] for i in l])

# ==========================================
# 2. THE ENTANGLED MODEL
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
        return logits, facing

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 3. LOAD HER BRAIN
# ==========================================
print("Booting Merel's dream engine...")
model = MerelModel().to(device)
if os.path.exists('merel_words.pth'):
    model.load_state_dict(torch.load('merel_words.pth', map_location=device))
    print("Brain loaded.")
model.eval()

# ==========================================
# 4. THE DREAM LOOP
# ==========================================
print("Merel is dreaming... (Press Ctrl+C to stop)")

try:
    while True:
        # Start with a random seed (or the start token)
        seed = torch.randint(0, vocab_size, (1, 1), device=device)

        # Generate 20 tokens
        dream_tokens = model.generate(seed, max_new_tokens=20)[0].tolist()
        dream_text = decode(dream_tokens)

        # Get the facing vector (compass)
        _, facing = model(seed)
        last_facing = facing[0, -1, :].tolist()

        # Format the timestamp
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write to the dream journal
        with open('dreams.txt', 'a', encoding='utf-8') as f:
            f.write(f"[{now}] {dream_text}\n")
            f.write(f"  └─ Compass: X={last_facing[0]:.2f}, Y={last_facing[1]:.2f}, Z={last_facing[2]:.2f}\n\n")

        print(f"[{now}] {dream_text}")

        # Sleep for 60 seconds
        time.sleep(60)

except KeyboardInterrupt:
    print("\nMerel woke up.")
