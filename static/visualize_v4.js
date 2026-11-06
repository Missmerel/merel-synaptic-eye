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
// 2. NEURON PAIR (ENTANGLED)
// ==========================================
class NeuronPair {
    constructor(index, centerX, centerY, centerZ, size) {
        this.index = index;
        this.centerX = centerX;
        this.centerY = centerY;
        this.centerZ = centerZ;
        this.size = size;

        this.x = centerX + (Math.random() - 0.5) * size;
        this.y = centerY + (Math.random() - 0.5) * size;
        this.z = centerZ + (Math.random() - 0.5) * size;

        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.vz = (Math.random() - 0.5) * 0.5;

        this.facingX = (Math.random() - 0.5);
        this.facingY = (Math.random() - 0.5);
        this.facingZ = (Math.random() - 0.5);

        this.activationNorth = 0;
        this.activationSouth = 0;
        this.decay = 0;

        this.projX = 0;
        this.projY = 0;
        this.projScale = 1;
    }

    update(time) {
        this.x += this.vx;
        this.y += this.vy;
        this.z += this.vz;

        const half = this.size / 2;
        if (this.x < this.centerX - half || this.x > this.centerX + half) this.vx *= -1;
        if (this.y < this.centerY - half || this.y > this.centerY + half) this.vy *= -1;
        if (this.z < this.centerZ - half || this.z > this.centerZ + half) this.vz *= -1;

        this.facingX += Math.sin(time + this.index) * 0.02;
        this.facingY += Math.cos(time + this.index) * 0.02;
        this.facingZ += Math.sin(time * 0.5 + this.index) * 0.02;

        this.decay = Math.max(0, this.decay - 0.02);
    }

    project(rotX, rotY) {
        const rotated = Math3D.rotate(this.x - this.centerX, this.y - this.centerY, this.z - this.centerZ, rotX, rotY);
        const fov = 800;
        this.projScale = fov / (fov + rotated.z + 400);
        this.projX = this.centerX + rotated.x * this.projScale;
        this.projY = this.centerY + rotated.y * this.projScale;
    }
}

// ==========================================
// 3. AUDIO SYNTHESIZER
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
        this.hum = this.ctx.createOscillator();
        this.humGain = this.ctx.createGain();
        this.hum.type = 'sine';
        this.hum.frequency.setValueAtTime(45, this.ctx.currentTime);
        this.humGain.gain.setValueAtTime(0.015, this.ctx.currentTime);
        this.hum.connect(this.humGain);
        this.humGain.connect(this.ctx.destination);
        this.hum.start();
    }
    fire() {
        if (!this.isPlaying || this.fireCooldown > 0) return;
        this.fireCooldown = 0.05;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.type = 'square';
        osc.frequency.setValueAtTime(600 + Math.random() * 800, this.ctx.currentTime);
        gain.gain.setValueAtTime(0.03, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.08);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.08);
    }
    update(dt) { if (this.fireCooldown > 0) this.fireCooldown -= dt; }
}

// ==========================================
// 4. BRAIN VISUALIZER (ENTANGLED PAIRS)
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

        this.pairs = [];
        this.numPairs = 128;
        this.timeOffset = 0;
        this.selectedNeuron = -1;
        this.rotX = 0;
        this.rotY = 0;
        this.realFacing = [1, 0, 0];

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
        const size = Math.min(width, height) * 0.6;

        this.pairs = [];
        for (let i = 0; i < this.numPairs; i++) {
            this.pairs.push(new NeuronPair(i, centerX, centerY, 0, size));
        }
    }

    updateActivations(activations, facingVector) {
        if (!activations) return;
        this.audio.start();
        if (facingVector) this.realFacing = facingVector;

        for (let i = 0; i < this.pairs.length; i++) {
            const northIdx = i * 2;
            const southIdx = i * 2 + 1;

            let northVal = 0;
            let southVal = 0;

            if (northIdx < activations.length) {
                northVal = Number(activations[northIdx]);
                if (isNaN(northVal)) northVal = 0;
            }
            if (southIdx < activations.length) {
                southVal = Number(activations[southIdx]);
                if (isNaN(southVal)) southVal = 0;
            }

            this.pairs[i].activationNorth = northVal;
            this.pairs[i].activationSouth = southVal;

            if (Math.abs(northVal) > 0.5 || Math.abs(southVal) > 0.5) {
                this.pairs[i].decay = 1.0;
                if (Math.random() < 0.1) this.audio.fire();
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

            if (this.pairs.length === 0) return;

            if (!this.isDragging) {
                this.rotY += 0.003;
                this.rotX += 0.001;
            }

            this.pairs.forEach(p => {
                p.update(time);
                p.project(this.rotX, this.rotY);
            });

            const sortedPairs = [...this.pairs].sort((a, b) => b.projScale - a.projScale);

            // 1. Draw Connections
            sortedPairs.forEach(p => {
                const northX = p.projX + p.facingX * 30 * p.projScale;
                const northY = p.projY + p.facingY * 30 * p.projScale;
                const southX = p.projX - p.facingX * 30 * p.projScale;
                const southY = p.projY - p.facingY * 30 * p.projScale;

                const intensity = Math.max(Math.abs(p.activationNorth), Math.abs(p.activationSouth), p.decay);
                const alpha = Math.min(intensity * 2, 1) * p.projScale;

                if (alpha > 0.05) {
                    this.graphics.lineStyle(1, 0xffffff, alpha * 0.3);
                    this.graphics.moveTo(northX, northY);
                    this.graphics.lineTo(southX, southY);
                }
            });

            // 2. Draw Neurons
            sortedPairs.forEach(p => {
                const northX = p.projX + p.facingX * 30 * p.projScale;
                const northY = p.projY + p.facingY * 30 * p.projScale;
                const southX = p.projX - p.facingX * 30 * p.projScale;
                const southY = p.projY - p.facingY * 30 * p.projScale;

                const nIntensity = Math.abs(p.activationNorth);
                if (nIntensity > 0.05) {
                    const nAlpha = Math.min(nIntensity * 2, 1) * p.projScale;
                    const nSize = (2 + nIntensity * 4) * p.projScale;
                    this.graphics.beginFill(0x00ffff, nAlpha * 0.2);
                    this.graphics.drawCircle(northX, northY, nSize * 3);
                    this.graphics.beginFill(0x00ffff, nAlpha);
                    this.graphics.drawCircle(northX, northY, nSize);
                    this.graphics.endFill();
                }

                const sIntensity = Math.abs(p.activationSouth);
                if (sIntensity > 0.05) {
                    const sAlpha = Math.min(sIntensity * 2, 1) * p.projScale;
                    const sSize = (2 + sIntensity * 4) * p.projScale;
                    this.graphics.beginFill(0xff00aa, sAlpha * 0.2);
                    this.graphics.drawCircle(southX, southY, sSize * 3);
                    this.graphics.beginFill(0xff00aa, sAlpha);
                    this.graphics.drawCircle(southX, southY, sSize);
                    this.graphics.endFill();
                }
            });

            // 3. Draw Compass
            if (this.realFacing) {
                const fx = this.realFacing[0];
                const fy = this.realFacing[1];
                const angle = Math.atan2(fy, fx);

                const centerX = this.app.screen.width / 2;
                const centerY = this.app.screen.height / 2;
                const compassLength = 100;

                const endX = centerX + Math.cos(angle) * compassLength;
                const endY = centerY + Math.sin(angle) * compassLength;

                this.graphics.lineStyle(3, 0xffffff, 0.8);
                this.graphics.moveTo(centerX, centerY);
                this.graphics.lineTo(endX, endY);

                this.graphics.beginFill(0xffffff, 1);
                this.graphics.drawCircle(endX, endY, 6);
                this.graphics.endFill();
            }

        } catch (e) {
            console.error("Animate crash:", e);
        }
    }
}

// ==========================================
// 5. CHAT CONTROLLER
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
                this.brain.updateActivations(data.brain_activity, data.facing_vector);

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
// 6. BOOTSTRAP
// ==========================================
window.addEventListener('load', () => {
    const brain = new BrainVisualizer3D();
    const chat = new ChatController(brain);
    window.brainInstance = brain;

    let attempts = 0;
    const forceResize = () => {
        const container = document.getElementById('brain-container');
        if (container) {
            const width = container.clientWidth;
            const height = container.clientHeight;
            if (width > 200 && height > 200) {
                brain.app.renderer.resize(width, height);
                brain.initNeurons();
                brain.updateActivations(Array(256).fill(0.1), [1, 0, 0]);
                console.log(`Brain resized to ${width}x${height}`);
                return;
            }
        }
        attempts++;
        if (attempts < 60) requestAnimationFrame(forceResize);
    };
    requestAnimationFrame(forceResize);
});

window.selectNeuron = (index) => {
    const brain = window.brainInstance;
    if (!brain) return;
    brain.selectedNeuron = (brain.selectedNeuron === index) ? -1 : index;
    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell, i) => {
        cell.classList.toggle('selected', i === brain.selectedNeuron);
    });
};
