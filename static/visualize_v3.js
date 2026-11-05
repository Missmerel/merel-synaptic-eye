// ==========================================
// 0. AUDIO SYNTHESIZER (ECHO CHAMBER)
// ==========================================
class SynthAudio {
    constructor() {
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        this.isPlaying = false;
        this.fireCooldown = 0;
    }

    start() {
        if (this.isPlaying) return;
        this.isPlaying = true;
        this.ctx.resume();

        // 1. Low Hum
        this.hum = this.ctx.createOscillator();
        this.humGain = this.ctx.createGain();
        this.hum.type = 'sine';
        this.hum.frequency.setValueAtTime(45, this.ctx.currentTime);
        this.humGain.gain.setValueAtTime(0.015, this.ctx.currentTime);
        this.hum.connect(this.humGain);
        this.humGain.connect(this.ctx.destination);
        this.hum.start();

        // 2. Static
        const bufferSize = this.ctx.sampleRate * 2;
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            data[i] = Math.random() * 2 - 1;
        }
        this.noise = this.ctx.createBufferSource();
        this.noise.buffer = buffer;
        this.noise.loop = true;
        this.noiseGain = this.ctx.createGain();
        this.noiseGain.gain.setValueAtTime(0.008, this.ctx.currentTime);
        this.noise.connect(this.noiseGain);
        this.noiseGain.connect(this.ctx.destination);
        this.noise.start();
    }

    stop() {
        if (!this.isPlaying) return;
        this.isPlaying = false;
        try { this.hum.stop(); } catch(e) {}
        try { this.noise.stop(); } catch(e) {}
    }

    fire() {
        if (!this.isPlaying) return;
        // Cooldown to prevent audio clipping
        if (this.fireCooldown > 0) return;
        this.fireCooldown = 0.05;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.type = 'square';
        const freq = 600 + Math.random() * 800;
        osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
        gain.gain.setValueAtTime(0.03, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.08);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.08);
    }

    update(dt) {
        if (this.fireCooldown > 0) this.fireCooldown -= dt;
    }
}

// ==========================================
// 1. 3D MATH UTILITY
// ==========================================
class Math3D {
    static rotate(x, y, z, angleX, angleY) {
        let cosY = Math.cos(angleY);
        let sinY = Math.sin(angleY);
        let x1 = x * cosY - z * sinY;
        let z1 = x * sinY + z * cosY;
        let cosX = Math.cos(angleX);
        let sinX = Math.sin(angleX);
        let y1 = y * cosX - z1 * sinX;
        let z2 = y * sinX + z1 * cosX;
        return { x: x1, y: y1, z: z2 };
    }
}

// ==========================================
// 2. 3D NEURON
// ==========================================
class Neuron3D {
    constructor(index, total, centerX, centerY, radius) {
        this.index = index;
        this.activation = 0;
        this.targetActivation = 0;
        this.decay = 0;

        const phi = Math.acos(1 - 2 * (index + 0.5) / total);
        const theta = Math.PI * (1 + Math.sqrt(5)) * index;

        this.baseX = radius * Math.sin(phi) * Math.cos(theta);
        this.baseY = radius * Math.sin(phi) * Math.sin(theta);
        this.baseZ = radius * Math.cos(phi);

        this.centerX = centerX;
        this.centerY = centerY;
        this.x = centerX;
        this.y = centerY;
        this.z = 0;
        this.scale = 1;
    }

    update(time) {
        this.activation += (this.targetActivation - this.activation) * 0.1;
        if (Math.abs(this.targetActivation) > 0.5 && this.decay < 0.1) {
            this.decay = 1.0;
        } else {
            this.decay = Math.max(0, this.decay - 0.03);
        }
    }

    project(rotX, rotY) {
        const wave = Math.sin(performance.now() / 500 + this.index) * 5;
        const r = 1 + (Math.abs(this.activation) * 0.3) + (wave * 0.005);
        let x = this.baseX * r;
        let y = this.baseY * r;
        let z = this.baseZ * r;
        const rotated = Math3D.rotate(x, y, z, rotX, rotY);
        const fov = 800;
        this.scale = fov / (fov + rotated.z + 400);
        this.x = this.centerX + rotated.x * this.scale;
        this.y = this.centerY + rotated.y * this.scale;
        this.z = rotated.z;
    }
}

// ==========================================
// 3. BRAIN VISUALIZER (3D)
// ==========================================
class BrainVisualizer3D {
    constructor() {
        this.app = new PIXI.Application({
            backgroundColor: 0x050505,
            antialias: true,
            resolution: window.devicePixelRatio || 1,
            width: window.innerWidth,
            height: window.innerHeight,
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
        this.selectedNeuron = -1;
        this.rotX = 0;
        this.rotY = 0;
        this.audio = new SynthAudio();

        this.container.eventMode = 'static';
        this.container.hitArea = new PIXI.Rectangle(-2000, -2000, 4000, 4000);
        this.container.cursor = 'grab';

        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };
        this.rotStart = { x: 0, y: 0 };

        this.container.on('pointerdown', (e) => {
            this.isDragging = true;
            this.container.cursor = 'grabbing';
            this.dragStart.x = e.data.global.x;
            this.dragStart.y = e.data.global.y;
            this.rotStart.x = this.rotX;
            this.rotStart.y = this.rotY;
        });

        this.container.on('pointermove', (e) => {
            if (this.isDragging) {
                const dx = e.data.global.x - this.dragStart.x;
                const dy = e.data.global.y - this.dragStart.y;
                this.rotY = this.rotStart.y + (dx * 0.01);
                this.rotX = this.rotStart.x + (dy * 0.01);
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
    }

    initNeurons() {
        const container = document.getElementById('brain-container');
        if (!container) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        if (width === 0 || height === 0) return;

        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.3;

        this.neurons = [];
        for (let i = 0; i < this.numNeurons; i++) {
            this.neurons.push(new Neuron3D(i, this.numNeurons, centerX, centerY, radius));
        }
    }

    updateActivations(activations) {
        if (!activations) return;

        // Start the audio engine on first activation (user interaction)
        this.audio.start();

        for (let i = 0; i < this.neurons.length; i++) {
            if (i < activations.length) {
                const val = Number(activations[i]);
                this.neurons[i].targetActivation = isNaN(val) ? 0 : val;

                // Fire audio for highly active neurons
                if (Math.abs(val) > 0.6 && Math.random() < 0.1) {
                    this.audio.fire();
                }
            }
        }
        this.timeOffset += 2;
    }

    animate(delta) {
        try {
            const dt = delta / 60;
            const time = performance.now() / 1000 + this.timeOffset;
            this.graphics.clear();

            this.audio.update(dt);

            if (this.neurons.length === 0) return;

            if (!this.isDragging) {
                this.rotY += 0.003;
            }

            this.neurons.forEach(n => {
                n.update(time);
                n.project(this.rotX, this.rotY);
            });

            const sortedNeurons = [...this.neurons].sort((a, b) => b.z - a.z);

            for (let i = 0; i < sortedNeurons.length; i++) {
                const n1 = sortedNeurons[i];
                for (let j = i + 1; j < sortedNeurons.length; j++) {
                    const n2 = sortedNeurons[j];
                    const dx = n1.x - n2.x; const dy = n1.y - n2.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < 50) {
                        const avg = (Math.abs(n1.activation) + Math.abs(n2.activation)) * 0.5;
                        const alpha = (1 - dist / 50) * avg * Math.min(n1.scale, n2.scale);
                        if (alpha > 0.05) {
                            const color = (n1.activation > 0 && n2.activation > 0) ? 0x00ffff : 0xff00aa;
                            this.graphics.lineStyle(1, color, alpha);
                            this.graphics.moveTo(n1.x, n1.y);
                            this.graphics.lineTo(n2.x, n2.y);
                        }
                    }
                }
            }

            sortedNeurons.forEach((n) => {
                if (isNaN(n.x) || isNaN(n.y)) return;
                const isSelected = (n.index === this.selectedNeuron);
                const baseColor = n.activation > 0 ? 0x00ffff : 0xff00aa;
                const color = isSelected ? 0xffffff : (n.decay > 0.8 ? 0xffffff : baseColor);
                const intensity = Math.max(Math.abs(n.activation), n.decay);
                const depthFactor = Math.max(0.2, n.scale);
                const alpha = isSelected ? 1 : Math.min(intensity * 2 + 0.1, 1) * depthFactor;
                const size = (isSelected ? 10 : 2 + intensity * 5) * n.scale;

                this.graphics.beginFill(color, alpha * 0.2);
                this.graphics.drawCircle(n.x, n.y, size * 3);
                this.graphics.beginFill(color, alpha);
                this.graphics.drawCircle(n.x, n.y, size);
                this.graphics.endFill();

                if (isSelected) {
                    this.graphics.lineStyle(2, 0xffffff, 1);
                    this.graphics.drawCircle(n.x, n.y, size * 2.5);
                }
            });
        } catch (e) {
            console.error("Animate crash:", e);
        }
    }
}

// ==========================================
// 4. CHAT CONTROLLER
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
                const cloudEl = document.getElementById('word-cloud');
                if (cloudEl) {
                    const tagsHTML = data.top_nodes.map(node => {
                        const className = node.value > 0 ? 'positive' : 'negative';
                        const size = Math.min(10 + Math.abs(node.value) * 8, 20);
                        return `<span class="word-tag ${className}" style="font-size: ${size}px;">${node.word}</span>`;
                    }).join('');
                    cloudEl.innerHTML = tagsHTML;
                }
                const nodeText = data.top_nodes.map(n => n.word).join(', ');
                document.getElementById('top-nodes').textContent = nodeText;
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
// 5. BOOTSTRAP (LEGENDARY EDITION)
// ==========================================
window.addEventListener('load', () => {
    const brain = new BrainVisualizer3D();
    const chat = new ChatController(brain);
    window.brainInstance = brain;

    // Startup Sequence
    const bootMessages = [
        "INITIALIZING NEURAL NETWORK...",
        "LOADING 256D HIDDEN STATE...",
        "CONNECTING TO WHATSAPP MEMORY...",
        "INJECTING TILBURG WIKIPEDIA...",
        "MAPPING SYNAPTIC CONNECTIONS...",
        "MEREL ONLINE."
    ];
    let bootIndex = 0;
    const bootEl = document.getElementById('boot-text');
    if (bootEl) {
        const bootInterval = setInterval(() => {
            if (bootIndex < bootMessages.length) {
                bootEl.textContent = bootMessages[bootIndex];
                bootIndex++;
            } else {
                clearInterval(bootInterval);
                bootEl.style.opacity = 0;
                setTimeout(() => bootEl.style.display = 'none', 1000);
            }
        }, 400);
    }

    // Force Resize
    let attempts = 0;
    const forceResize = () => {
        const container = document.getElementById('brain-container');
        if (container) {
            const width = container.clientWidth;
            const height = container.clientHeight;

            if (width > 200 && height > 200) {
                brain.app.renderer.resize(width, height);
                brain.initNeurons();
                brain.updateActivations(Array(256).fill(0.1));
                console.log(`SUCCESS: Brain forced to ${width}x${height}`);
                return;
            }
        }
        attempts++;
        if (attempts < 60) {
            requestAnimationFrame(forceResize);
        }
    };
    requestAnimationFrame(forceResize);

    // Speak Button
    const speakBtn = document.getElementById('speak-btn');
    if (speakBtn) {
        speakBtn.addEventListener('click', () => {
            const lastMessage = document.querySelectorAll('.msg.merel');
            if (lastMessage.length > 0) {
                const text = lastMessage[lastMessage.length - 1].innerText.replace('Merel: ', '');
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'nl-NL';
                utterance.pitch = 1.2;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            }
        });
    }
});
// ==========================================
// 6. SVG EXPORTER
// ==========================================
window.downloadSVG = () => {
    const brain = window.brainInstance;
    if (!brain) return;

    const width = brain.app.renderer.width / brain.app.renderer.resolution;
    const height = brain.app.renderer.height / brain.app.renderer.resolution;

    // Start the SVG string
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">\n`;
    svg += `<rect width="100%" height="100%" fill="#050505"/>\n`;

    // Sort neurons by depth (back to front)
    const sortedNeurons = [...brain.neurons].sort((a, b) => b.z - a.z);

    // 1. Draw Connections
    for (let i = 0; i < sortedNeurons.length; i++) {
        const n1 = sortedNeurons[i];
        for (let j = i + 1; j < sortedNeurons.length; j++) {
            const n2 = sortedNeurons[j];
            const dx = n1.x - n2.x; const dy = n1.y - n2.y;
            const dist = Math.sqrt(dx*dx + dy*dy);

            if (dist < 50) {
                const avg = (Math.abs(n1.activation) + Math.abs(n2.activation)) * 0.5;
                const alpha = (1 - dist / 50) * avg * Math.min(n1.scale, n2.scale);
                if (alpha > 0.05) {
                    const color = (n1.activation > 0 && n2.activation > 0) ? '#00ffff' : '#ff00aa';
                    svg += `<line x1="${n1.x.toFixed(2)}" y1="${n1.y.toFixed(2)}" x2="${n2.x.toFixed(2)}" y2="${n2.y.toFixed(2)}" stroke="${color}" stroke-width="1" opacity="${alpha.toFixed(2)}"/>\n`;
                }
            }
        }
    }

    // 2. Draw Neurons
    sortedNeurons.forEach(n => {
        if (isNaN(n.x) || isNaN(n.y)) return;
        const baseColor = n.activation > 0 ? '#00ffff' : '#ff00aa';
        const color = n.decay > 0.8 ? '#ffffff' : baseColor;
        const intensity = Math.max(Math.abs(n.activation), n.decay);
        const depthFactor = Math.max(0.2, n.scale);
        const alpha = Math.min(intensity * 2 + 0.1, 1) * depthFactor;
        const size = (2 + intensity * 5) * n.scale;

        // Glow
        svg += `<circle cx="${n.x.toFixed(2)}" cy="${n.y.toFixed(2)}" r="${(size * 3).toFixed(2)}" fill="${color}" opacity="${(alpha * 0.2).toFixed(2)}"/>\n`;
        // Core
        svg += `<circle cx="${n.x.toFixed(2)}" cy="${n.y.toFixed(2)}" r="${size.toFixed(2)}" fill="${color}" opacity="${alpha.toFixed(2)}"/>\n`;
    });

    svg += `</svg>\n`;

    // Create a blob and download it
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `merel_brain_${Date.now()}.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

// Connect the button
window.addEventListener('load', () => {
    const btn = document.getElementById('download-svg-btn');
    if (btn) {
        btn.addEventListener('click', window.downloadSVG);
    }
});
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
