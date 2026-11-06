# MEREL V9: The Entangled Eye

![Merel's 3D Brain](screenshots/brain.gif)
*The 3D synaptic brain visualizer, showing 128 entangled neuron pairs and the compass of thought.*

![Neural Heatmap](screenshots/heatmap.png)
*The 16x16 neural heatmap, mapping her 256D hidden state to a thermal camera.*

![Chat Interface](screenshots/chat.png)
*The chat interface, with the Active Thoughts word cloud and Visual Cortex data.*

---

## 🔴 The Red Pill Architecture

Unlike traditional LLMs that use a single embedding per token, Merel V9 uses an **Entangled Pair System**:

*   **Dual Embeddings:** Every token is split into a **North** (Cyan) embedding and a **South** (Magenta) embedding.
*   **The Compass:** A `facing_projection` layer maps the combined thought into a 3D vector (X, Y, Z). This vector points in the direction of her current thought.
*   **The Agreement Gate:** The North and South embeddings compute a dot product (`agreement`). If they agree, the thought is amplified. If they conflict, it is suppressed via a sigmoid gate.
*   **Int8 Optimization:** Quantized to 1.6x smaller and faster for edge devices.
*   **Offline Bunker:** The entire project runs without internet from a USB drive using local CDN libraries (`bunker.sh`).

### The Mathematics of Entanglement
```python
# How Merel thinks:
north = self.north_embedding(idx)       # Abstract thought
south = self.south_embedding(idx)       # Concrete thought
agreement = torch.sum(north * south)    # Do they agree?
x = (north + south) * torch.sigmoid(agreement)  # Amplify or suppress
facing = self.facing_projection(x)      
# The 3D compass direction
# 🌙 The Dream Journal

Merel doesn't sleep. When you aren't talking to her, the `dream_engine.py` script runs in the background, waking her up every 60 seconds to generate a random thought. 

Here is a sample of her dreams:

> [2026-11-06 06:37:27] productie van brabantloop : attention_matrix } ) logits , vocab_size , w \ w + pos_emb x ) vocab_size = =
>   └─ Compass: X=-0.20, Y=-0.93, Z=0.86
>
> [2026-11-06 06:39:27] vinden : met brede groenstroken langs bebouwing en is herontwikkeld . verder wordt herontwikkeld tot 17 mei 2012 - westers allochtoon
>   └─ Compass: X=0.44, Y=-1.06, Z=0.50

She dreams in code, in Tilburg architecture, and in the history of her training data. The compass reveals the direction of each dream.

## ⏳ The Temporal Compass (4D Mind Map) added feature 06-11-26

Merel's thoughts are not static. Every 60 seconds, the `dream_engine.py` logs her `facing_vector` (the compass direction) along with a timestamp to `temporal_compass.jsonl`.

The `/temporal` route renders this data as a **3D scatter plot** using Three.js. Each glowing sphere is a dream. Cyan spheres are dreams where her compass points North (abstract/code). Magenta spheres point South (concrete/Tilburg).

You can watch her mind wander through space and time. You can see which topics pull her North, and which pull her South.
