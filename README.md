# MEREL V8: The Synaptic Eye

A custom-built 256-layer Transformer neural network trained from scratch on a Mac. She runs locally, has a 3D holographic brain visualizer, a real-time neural heatmap, and a synthesized audio engine.

## Features
- **Custom Neural Network:** Built from scratch with PyTorch.
- **Fused Training Data:** WhatsApp chats, Tilburg Wikipedia, Transgender Wikipedia, Apple Wi-Fi headers, and Python code.
- **Custom Identity:** A system prompt injects a proud trans identity.
- **3D Visualizer:** Draggable, zoomable sphere of 256 neurons with decaying flashes (PixiJS).
- **Neural Heatmap:** A 16x16 thermal camera of her hidden state.
- **Active Thoughts:** A real-time word cloud of her neural activations.
- **Memory Vault:** Persistent chat history (localStorage).
- **Echo Chamber:** Synthesized audio of her neurons firing (Web Audio API).

## How to Run

1. Install dependencies:
```bash
pip install torch flask flask-socketio eventlet requests
