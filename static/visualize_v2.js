// ==========================================
// 1. 2D NEURON (WITH DECAY)
// ==========================================
class Neuron2D {
    constructor(index, total, centerX, centerY, radius) {
        this.index = index;
        this.activation = 0;
        this.targetActivation = 0;
        this.decay = 0;

        const angle = (index / total) * Math.PI * 2;
        this.x = centerX + Math.cos(angle) * radius;
        this.y = centerY + Math.sin(angle) * radius;

        this.vx = (Math.random() - 0.5) * 2;
        this.vy = (Math.random() - 0.5) * 2;
        this.centerX = centerX;
        this.centerY = centerY;
    }

    update(time) {
        this.activation += (this.targetActivation - this.activation) * 0.1;

        if (Math.abs(this.targetActivation) > 0.5 && this.decay < 0.1) {
            this.decay = 1.0;
        } else {
            this.decay = Math.max(0, this.decay - 0.03);
        }

        const dx = this.centerX - this.x;
        const dy = this.centerY - this.y;
        const dist = Math.sqrt(dx*dx + dy*dy) || 1;

        const force = Math.min(0.01 + (Math.abs(this.activation) * 0.1) + (this.decay * 0.05), 0.5);
        this.vx += (dx / dist) * force;
        this.vy += (dy / dist) * force;

        this.vx *= 0.95;
        this.vy *= 0.95;
        this.x += this.vx;
        this.y += this.vy;
    }
}

// ==========================================
// 2. BRAIN VISUALIZER
// ==========================================
class BrainVisualizer2D {
    constructor() {
        this.app = new PIXI.Application({
            backgroundColor: 0x050505,
            antialias: true,
            resolution: window.devicePixelRatio || 1,
            resizeTo: document.getElementById('brain-container'),
            autoDensity: true
        });
        document.getElementById('brain-container').appendChild(this.app.view);

        this.container = new PIXI.Container();
        this.container.blendMode = PIXI.BLEND_MODES.ADD;
        this.app.stage.addChild(this.container);

        this.graphics = new PIXI.Graphics();
        this.container.addChild(this.graphics);

        this.neurons = [];
        this.numNeurons = 256;
        this.timeOffset = 0;
        this.pulseWaves = [];
        this.selectedNeuron = -1;

        this.container.eventMode = 'static';
        this.container.hitArea = new PIXI.Rectangle(0, 0, window.innerWidth, window.innerHeight);
        this.container.cursor = 'grab';

        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };
        this.containerStart = { x: 0, y: 0 };

        this.container.on('pointerdown', (e) => {
            this.isDragging = true;
            this.container.cursor = 'grabbing';
            this.dragStart.x = e.data.global.x;
            this.dragStart.y = e.data.global.y;
            this.containerStart.x = this.container.x;
            this.containerStart.y = this.container.y;
        });

        this.container.on('pointermove', (e) => {
            if (this.isDragging) {
                const dx = e.data.global.x - this.dragStart.x;
                const dy = e.data.global.y - this.dragStart.y;
                this.container.x = this.containerStart.x + dx;
                this.container.y = this.containerStart.y + dy;
            }
        });

        this.container.on('pointerup', () => { this.isDragging = false; this.container.cursor = 'grab'; });
        this.container.on('pointerupoutside', () => { this.isDragging = false; this.container.cursor = 'grab'; });

        this.app.view.addEventListener('wheel', (e) => {
            e.preventDefault();
            const scale = this.container.scale.x;
            const newScale = scale + (e.deltaY > 0 ? -0.1 : 0.1);
            if (newScale > 0.2 && newScale < 5) { this.container.scale.set(newScale); }
        });

        this.app.ticker.add(this.animate.bind(this));
        this.app.renderer.on('resize', () => { this.initNeurons(); });
        this.initNeurons();
    }

    initNeurons() {
        const width = this.app.screen.width;
        const height = this.app.screen.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.35;

        this.neurons = [];
        for (let i = 0; i < this.numNeurons; i++) {
            this.neurons.push(new Neuron2D(i, this.numNeurons, centerX, centerY, radius));
        }
    }

    updateActivations(activations) {
        if (!activations) return;
        this.pulseWaves = [];

        for (let i = 0; i < this.neurons.length; i++) {
            if (i < activations.length) {
                const val = Number(activations[i]);
                this.neurons[i].targetActivation = isNaN(val) ? 0 : val;
            }
        }
        this.timeOffset += 2;
        this.pulseWaves.push({ radius: 0, alpha: 0.5, speed: 2 });
    }

    animate(delta) {
        try {
            const time = performance.now() / 1000 + this.timeOffset;
            this.graphics.clear();

            const width = this.app.screen.width;
            const height = this.app.screen.height;
            const centerX = width / 2;
            const centerY = height / 2;

            this.neurons.forEach(n => n.update(time));

            for (let i = 0; i < this.neurons.length; i++) {
                const n1 = this.neurons[i];
                for (let j = i + 1; j < this.neurons.length; j++) {
                    const n2 = this.neurons[j];
                    const dx = n2.x - n1.x; const dy = n2.y - n1.y;
                    const dist = Math.sqrt(dx*dx + dy*dy) || 1;
                    if (dist < 50) {
                        const push = (50 - dist) / 50;
                        n1.vx -= (dx / dist) * push * 0.5; n1.vy -= (dy / dist) * push * 0.5;
                        n2.vx += (dx / dist) * push * 0.5; n2.vy += (dy / dist) * push * 0.5;
                    }
                }
            }

            if (this.pulseWaves && this.pulseWaves.length > 0) {
                for (let i = this.pulseWaves.length - 1; i >= 0; i--) {
                    const wave = this.pulseWaves[i];
                    if (!wave || isNaN(wave.radius) || isNaN(wave.alpha)) {
                        this.pulseWaves.splice(i, 1);
                        continue;
                    }
                    wave.radius += wave.speed;
                    wave.alpha -= 0.01;
                    if (wave.alpha > 0) {
                        this.graphics.lineStyle(2, 0x00ffff, wave.alpha);
                        this.graphics.drawCircle(centerX, centerY, wave.radius);
                    } else {
                        this.pulseWaves.splice(i, 1);
                    }
                }
            }

            for (let i = 0; i < this.neurons.length; i++) {
                const n1 = this.neurons[i];
                for (let j = i + 1; j < this.neurons.length; j++) {
                    const n2 = this.neurons[j];
                    const dx = n1.x - n2.x; const dy = n1.y - n2.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < 80) {
                        const avg = (Math.abs(n1.activation) + Math.abs(n2.activation)) * 0.5;
                        const alpha = (1 - dist / 80) * avg;
                        if (alpha > 0.05) {
                            const color = (n1.activation > 0 && n2.activation > 0) ? 0x00ffff : 0xff00aa;
                            this.graphics.lineStyle(1, color, alpha);
                            this.graphics.moveTo(n1.x, n1.y);
                            this.graphics.lineTo(n2.x, n2.y);
                        }
                    }
                }
            }

            this.neurons.forEach((n, index) => {
                if (isNaN(n.x) || isNaN(n.y)) return;

                const isSelected = (index === this.selectedNeuron);
                const baseColor = n.activation > 0 ? 0x00ffff : 0xff00aa;
                const color = isSelected ? 0xffffff : (n.decay > 0.8 ? 0xffffff : baseColor);
                const intensity = Math.max(Math.abs(n.activation), n.decay);
                const alpha = isSelected ? 1 : Math.min(intensity * 2 + 0.1, 1);
                const size = isSelected ? 8 : (2 + intensity * 4);

                this.graphics.beginFill(color, alpha * 0.2);
                this.graphics.drawCircle(n.x, n.y, size * 3);
                this.graphics.beginFill(color, alpha);
                this.graphics.drawCircle(n.x, n.y, size);
                this.graphics.endFill();

                if (isSelected) {
                    this.graphics.lineStyle(2, 0xffffff, 0.8);
                    this.graphics.drawCircle(n.x, n.y, size * 2);
                }
            });

        } catch (e) {
            console.error("Animate crash:", e);
        }
    }
}

// ==========================================
// 3. CHAT CONTROLLER
// ==========================================
class ChatController {
    constructor(brain) {
        this.brain = brain;
        this.socket = io();
        this.chatHistory = document.getElementById('chat-history');
        this.userInput = document.getElementById('user-input');
        this.selectedCell = -1;

        this.initEmptyGrid();

        const savedHistory = localStorage.getItem('merel_chat_history');
        if (savedHistory) {
            this.chatHistory.innerHTML = savedHistory;
            this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
        }

        this.setupSocket();
        this.setupInput();
    }

    initEmptyGrid() {
        const streamEl = document.getElementById('data-stream');
        if (streamEl) {
            const emptyCells = Array(256).fill('<div class="cell"></div>').join('');
            streamEl.innerHTML = emptyCells;
        }
    }

    setupSocket() {
        this.socket.on('merel_response', (data) => {
            this.addMessage('Merel', data.text, 'merel');

            if (data.brain_activity) {
                this.brain.updateActivations(data.brain_activity);

                const streamEl = document.getElementById('data-stream');
                if (streamEl) {
                    const displayData = data.brain_activity;

                    const gridHTML = displayData.map((val, index) => {
                        const safeVal = isNaN(val) ? 0 : val;
                        let color = '#0a0a0c';
                        let glow = 'none';

                        if (safeVal > 0.1) {
                            const intensity = Math.min(safeVal * 2, 1);
                            color = `rgba(0, 255, 255, ${intensity})`;
                            glow = `0 0 ${Math.min(safeVal * 10, 8)}px rgba(0, 255, 255, 0.5)`;
                        } else if (safeVal < -0.1) {
                            const intensity = Math.min(Math.abs(safeVal) * 2, 1);
                            color = `rgba(255, 0, 170, ${intensity})`;
                            glow = `0 0 ${Math.min(Math.abs(safeVal) * 10, 8)}px rgba(255, 0, 170, 0.5)`;
                        }

                        const isSelected = (index === this.selectedCell);
                        const selectedClass = isSelected ? 'selected' : '';
                        return `<div class="cell ${selectedClass}" style="background-color: ${color}; box-shadow: ${glow};" onclick="window.selectNeuron(${index})"></div>`;
                    }).join('');

                    streamEl.innerHTML = gridHTML;
                }
            }

            if (data.data_source) document.getElementById('data-source').textContent = data.data_source;
                        if (data.top_nodes) {
                // Update the old Top Nodes text (for compatibility)
                const nodeText = data.top_nodes.map(n => n.word).join(', ');
                document.getElementById('top-nodes').textContent = nodeText;

                // Update the new Word Cloud
                const cloudEl = document.getElementById('word-cloud');
                if (cloudEl) {
                    const tagsHTML = data.top_nodes.map(node => {
                        const className = node.value > 0 ? 'positive' : 'negative';
                        const size = Math.min(10 + Math.abs(node.value) * 8, 20);
                        return `<span class="word-tag ${className}" style="font-size: ${size}px;">${node.word}</span>`;
                    }).join('');
                    cloudEl.innerHTML = tagsHTML;
                }
            }
        });
    }

    setupInput() {
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && this.userInput.value.trim() !== '') {
                const text = this.userInput.value;
                this.userInput.value = '';
                this.addMessage('You', text, 'user');
                this.socket.emit('user_message', { text: text });
                this.brain.timeOffset += 2;
            }
        });
    }

    addMessage(sender, text, className) {
        const div = document.createElement('div');
        div.className = `msg ${className}`;
        div.innerHTML = `<strong>${sender}:</strong> ${text}`;
        this.chatHistory.appendChild(div);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;

        localStorage.setItem('merel_chat_history', this.chatHistory.innerHTML);
    }
}

// ==========================================
// 4. BOOTSTRAP
// ==========================================
window.addEventListener('load', () => {
    const brain = new BrainVisualizer2D();
    const chat = new ChatController(brain);
    window.brainInstance = brain;

    setTimeout(() => {
        const container = document.getElementById('brain-container');
        if (container) {
            const width = container.clientWidth;
            const height = container.clientHeight;
            brain.app.renderer.resize(width, height);
            brain.initNeurons();
            brain.updateActivations(Array(256).fill(0.1));
            console.log(`Brain resized to ${width}x${height}`);
        }
    }, 500);
});

// Global function for the heatmap click handler
window.selectNeuron = (index) => {
    const brain = window.brainInstance;
    if (!brain) return;

    if (brain.selectedNeuron === index) {
        brain.selectedNeuron = -1;
    } else {
        brain.selectedNeuron = index;
    }

    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell, i) => {
        if (i === brain.selectedNeuron) {
            cell.classList.add('selected');
        } else {
            cell.classList.remove('selected');
        }
    });
};
