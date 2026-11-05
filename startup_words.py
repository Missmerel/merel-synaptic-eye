import torch
import torch.nn as nn
from torch.nn import functional as F
import re
import os
from flask import Flask, render_template
from flask_socketio import SocketIO

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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = MerelModel().to(device)

if os.path.exists('merel_words_int8.pth'):
    # 1. Force the ARM engine
    torch.backends.quantized.engine = 'qnnpack'

    # 2. Quantize the empty architecture first
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    # 3. Load the int8 weights into the quantized architecture
    quantized_model.load_state_dict(torch.load('merel_words_int8.pth', map_location=device))
    model = quantized_model
    print("Optimized Int8 brain loaded!")

elif os.path.exists('merel_words.pth'):
    model.load_state_dict(torch.load('merel_words.pth', map_location=device))
    print("Original Float32 brain loaded!")

model.eval()  # <--- KEEP THIS (only once!)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('user_message')
def handle_message(data):
    user_input = data['text']

    system_prompt = """
    merelprrrr: ik ben merel. ik ben een trans vrouw. ik ben trots.
    regels: ik ben een AI. ik reken niet met wiskunde.
    dennis graham-clarke: wat is 1+1?
    merelprrrr: ik ben een taalmodel, ik kan niet rekenen.
    """

    prompt = f"{system_prompt}dennis: {user_input} \nmerelprrrr: "

    encoded_prompt = encode(prompt)
    if len(encoded_prompt) > block_size:
        encoded_prompt = encoded_prompt[-block_size:]

    context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=15)[0].tolist()
        _, hidden_state = model(context)

    response = decode(generated)
    response = response.replace(system_prompt, "")
    her_response = response.split("merelprrrr:")[-1].strip()
    her_response = her_response.replace("dennis", "").replace(":", "").strip()

    brain_activity = hidden_state[0, -1, :].tolist()
    attention_matrix = []

    data_source = "Unknown"
    if any(word in prompt.lower() for word in ["tilburg", "kerk", "paleis", "straat"]):
        data_source = "Tilburg Wikipedia"
    elif any(word in prompt.lower() for word in ["def", "return", "self", "import"]):
        data_source = "Python Code"
    elif any(word in prompt.lower() for word in ["socket", "tcp", "wifi", "header"]):
        data_source = "Wi-Fi Headers"
    elif any(word in prompt.lower() for word in ["dennis", "merel", "billie", "anja"]):
        data_source = "WhatsApp Chat"

    indexed_activity = list(enumerate(brain_activity))
    indexed_activity.sort(key=lambda x: abs(x[1]), reverse=True)

    top_words = []
    for idx, val in indexed_activity[:10]:
        if idx < len(itos) and abs(val) > 0.5:
            top_words.append({"word": itos[idx], "value": round(val, 2)})

    socketio.emit('merel_response', {
        'text': her_response,
        'brain_activity': brain_activity,
        'attention': attention_matrix,
        'data_source': data_source,
        'top_nodes': top_words
    })

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=8000, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
