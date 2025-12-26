/**
 * WebGPU Snow Effect
 * 基于 cyandev.app 的 WebGPU 下雪实现
 * https://github.com/unixzii/cyandev.app
 */

// ==================== WGSL Shader 代码 ====================
// Reason: 与原版一致，使用像素坐标系 + atomic 派生 + PCG32 状态机
const SHADER_CODE = `
// ========== 数据结构 ==========
struct Uniforms {
    viewportSize: vec2f,
    snowSpeed: f32,
    windForce: f32,
    sizeMin: f32,
    sizeMax: f32,
}

struct Particle {
    position: vec2f,
    size: vec2f,
    velocity: vec2f,
    distance: f32,
    opacity: f32,
    spawned: i32,
}

struct SimulationContext {
    time: f32,
    timeDelta: f32,
    randSeed: f32,
    particlesToSpawn: atomic<i32>,
}

// ========== Bindings ==========
// Reason: 原版 cyandev.app 使用同一个 group，通过不同 pipeline 区分 read/read_write
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Compute 阶段使用 read_write
@group(0) @binding(1) var<storage, read_write> writableParticles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> simulationCtx: SimulationContext;
// Render 阶段使用 read（与原版一致，同一个 buffer 不同访问模式）
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

// ========== PCG32 随机数生成器 ==========
struct Pcg32RandomState {
    state: u32,
}

var<private> randState: Pcg32RandomState;

fn initRand(invocationId: u32, seed: f32) {
    randState.state = invocationId * 1664525u + u32(seed * 1664525.0);
}

fn rand() -> f32 {
    let state = randState.state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    randState.state = (word >> 22u) ^ word;
    return f32(randState.state) / 4294967296.0;
}

// ========== Compute Shader ==========
@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) globalInvocationId: vec3u) {
    initRand(globalInvocationId.x, simulationCtx.randSeed);

    let timeDelta = simulationCtx.timeDelta / 10.0;
    // Reason: windForce 从 uniform 传入，默认 0.02 对应原硬编码 0.0002
    let wind = sin(simulationCtx.time / 5000.0) * (uniforms.windForce / 100.0);

    var particle = writableParticles[globalInvocationId.x];

    // 未派生或已落出屏幕底部，尝试重新派生
    if (particle.spawned == 0 || particle.position.y > uniforms.viewportSize.y) {
        if (atomicSub(&simulationCtx.particlesToSpawn, 1) > 0) {
            // 成功获取派生配额
            particle.position.x = rand() * uniforms.viewportSize.x;
            particle.position.y = -100.0;

            // distance 控制雪花锐度，越小越锐利
            let nearCamera = rand() > 0.95;
            let baseDistance = select(6.0, 1.0, nearCamera);
            let distanceVariation = select(3.0, 1.0, nearCamera);
            let distance = baseDistance + rand() * distanceVariation;
            particle.distance = distance;

            // Reason: 保留双峰分布，sizeMin/sizeMax 作为缩放因子
            let largeFlake = rand() > 0.92;
            let sizeRange = uniforms.sizeMax - uniforms.sizeMin;
            // 小雪花占 0~35% 范围，大雪花占 50~100% 范围
            let baseRatio = select(0.0, 0.5, largeFlake);
            let variationRatio = select(0.35, 0.5, largeFlake);
            let size = uniforms.sizeMin + sizeRange * (baseRatio + rand() * variationRatio);
            let distanceFactor = (distance / 9.0) * 0.1 + 1.0;
            particle.size = vec2f(size) * distanceFactor;

            // 速度：snowSpeed 同时缩放初速度
            let vyVariation = select(2.0, particle.size.y, largeFlake);
            particle.velocity = vec2f(-1.5 + rand() * 3.0, rand() * vyVariation * uniforms.snowSpeed);

            particle.opacity = 1.0 - distance / 9.0;
            particle.spawned = 1;
        }
    }

    // 更新运动
    particle.velocity.x += wind * timeDelta;
    // Reason: snowSpeed 控制下落加速度，默认 1.0 对应原硬编码 0.03
    particle.velocity.y += 0.03 * uniforms.snowSpeed * timeDelta;
    particle.position += particle.velocity * timeDelta;

    writableParticles[globalInvocationId.x] = particle;
}

// ========== Vertex Shader ==========
struct QuadVertexInput {
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) particleIndex: u32,
}

struct QuadVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) distance: f32,
    @location(2) opacity: f32,
}

@vertex
fn particleVertex(in: QuadVertexInput) -> QuadVertexOutput {
    // triangle-strip 4 顶点
    const vertices = array<vec2f, 4>(
        vec2f(-0.5, -0.5),
        vec2f(0.5, -0.5),
        vec2f(-0.5, 0.5),
        vec2f(0.5, 0.5)
    );

    let particle = particles[in.particleIndex];

    // 像素坐标归一化到 [0, 1]
    let particlePosNorm = particle.position / uniforms.viewportSize;
    let particleSizeNorm = particle.size / uniforms.viewportSize;

    let vertexPos = vertices[in.vertexIndex];
    // 映射到 NDC [-1, 1]
    let pos = (particlePosNorm + particleSizeNorm * vertexPos) * 2.0 - 1.0;

    var out: QuadVertexOutput;
    // y 轴翻转（像素坐标 y 向下，NDC y 向上）
    out.position = vec4f(pos.x, -pos.y, 0.0, 1.0);
    out.uv = vertexPos * 2.0;
    out.distance = particle.distance;
    out.opacity = particle.opacity;
    return out;
}

// ========== SDF 辅助函数 ==========
fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.0001), 0.0, 1.0);
    return length(pa - ba * h);
}

fn sdHexagon(p: vec2f, r: f32) -> f32 {
    let k = vec3f(-0.866025404, 0.5, 0.577350269);
    var q = abs(p);
    q = q - 2.0 * min(dot(k.xy, q), 0.0) * k.xy;
    q = q - vec2f(clamp(q.x, -k.z * r, k.z * r), r);
    return length(q) * sign(q.y);
}

// Reason: 修正后的空间折叠，正确折叠到 0~30 度扇区
fn fold_hex_30(p_in: vec2f) -> vec2f {
    var p = vec2f(p_in.x, abs(p_in.y));
    let n60 = vec2f(0.8660254, -0.5);
    p -= 2.0 * min(dot(p, n60), 0.0) * n60;
    p.y = abs(p.y);
    let n30 = vec2f(0.5, -0.8660254);
    p -= 2.0 * min(dot(p, n30), 0.0) * n30;
    return vec2f(p.x, abs(p.y));
}

// ========== Fragment Shader ==========
@fragment
fn particleFragment(in: QuadVertexOutput) -> @location(0) vec4f {
    if (in.position.x > uniforms.viewportSize.x || in.position.y > uniforms.viewportSize.y) {
        discard;
    }

    let uv = in.uv * 1.1;
    let p = fold_hex_30(uv);

    var d = 100.0;

    // 中心六边形轮廓
    d = min(d, abs(sdHexagon(uv, 0.12)) - 0.008);

    // 主枝 (沿 X 轴)
    d = min(d, sdSegment(p, vec2f(0.1, 0.0), vec2f(0.9, 0.0)));

    // 末端分叉
    d = min(d, sdSegment(p, vec2f(0.7, 0.0), vec2f(0.88, 0.15)));

    // 中部分叉
    d = min(d, sdSegment(p, vec2f(0.45, 0.0), vec2f(0.6, 0.12)));

    // 次枝 (30度方向，位于两个主枝之间)
    let dir_30 = vec2f(0.8660254, 0.5);
    let proj_len = clamp(dot(p, dir_30), 0.2, 0.5);
    d = min(d, length(p - dir_30 * proj_len));

    // 渲染：固定线宽 + 基于 distance 的柔和度
    let thickness = 0.022;
    let softness = 0.008 + (1.0 / max(in.distance, 1.0)) * 0.02;
    var alpha = 1.0 - smoothstep(thickness, thickness + softness, d);

    // 柔和发光效果
    let glow = exp(-max(d, 0.0) * 12.0) * 0.25;
    alpha = alpha + glow;
    alpha = clamp(alpha * in.opacity, 0.0, 1.0);

    return vec4f(alpha);
}
`;

// ==================== 常量 ====================
const PARTICLE_COUNT = 10000;
const SIZEOF_F32 = 4;
const SIZEOF_I32 = 4;

// Particle 结构大小: position(8) + size(8) + velocity(8) + distance(4) + opacity(4) + spawned(4) + padding(4) = 40 bytes
const PARTICLE_STRUCT_SIZE = 40;

// SimulationContext 结构大小: time(4) + timeDelta(4) + randSeed(4) + particlesToSpawn(4) = 16 bytes
const SIMULATION_CTX_SIZE = 16;

// Uniforms 结构大小: viewportSize(8) + snowSpeed(4) + windForce(4) + sizeMin(4) + sizeMax(4) = 24 bytes, 对齐到 32
const UNIFORMS_SIZE = 32;

// ==================== WebGPU 渲染器类 ====================
class WebGPUSnowRenderer {
    constructor(canvas, config) {
        this.canvas = canvas;
        this.config = config;
        this.device = null;
        this.context = null;
        this.running = false;
        this.time = 0;
        this.lastTime = 0;
        this.simulationCtxLocalBuffer = new ArrayBuffer(SIMULATION_CTX_SIZE);
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
        if (!adapter) {
            throw new Error('Failed to get GPU adapter');
        }

        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        if (!this.context) {
            throw new Error('Failed to get WebGPU canvas context');
        }
        this.format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied'
        });

        this._createResources();
        return true;
    }

    _createResources() {
        // Reason: 对齐到 workgroup_size(64) 避免越界访问
        const requestedCount = this.config.particleCount || PARTICLE_COUNT;
        const particleCount = Math.max(64, Math.floor(requestedCount / 64) * 64);

        // 创建 Shader 模块
        this.shaderModule = this.device.createShaderModule({ code: SHADER_CODE });

        // Uniform Buffer (viewportSize + snowSpeed + windForce + sizeMin + sizeMax)
        this.uniformBuffer = this.device.createBuffer({
            size: UNIFORMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Particle Buffer (不做 CPU 初始化，由 compute shader 派生)
        this.particleBuffer = this.device.createBuffer({
            size: particleCount * PARTICLE_STRUCT_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
        });

        // SimulationContext Buffer (含 atomic 计数器)
        this.simulationCtxBuffer = this.device.createBuffer({
            size: SIMULATION_CTX_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Compute BindGroup Layout
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });

        // Render BindGroup Layout
        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            ]
        });

        // BindGroups
        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.particleBuffer } },
                { binding: 2, resource: { buffer: this.simulationCtxBuffer } },
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.particleBuffer } },
            ]
        });

        // Compute Pipeline
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
            compute: { module: this.shaderModule, entryPoint: 'updateParticles' },
        });

        // Render Pipeline (triangle-strip + premultiplied blend)
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
            vertex: { module: this.shaderModule, entryPoint: 'particleVertex' },
            fragment: {
                module: this.shaderModule,
                entryPoint: 'particleFragment',
                targets: [{
                    format: this.format,
                    blend: {
                        color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                    }
                }]
            },
            primitive: { topology: 'triangle-strip' },
        });

        this.particleCount = particleCount;
    }

    _updateCanvasSize() {
        const dpr = window.devicePixelRatio || 1;
        const scaleFactor = Math.min(dpr, 2);
        const width = Math.floor(this.canvas.clientWidth * scaleFactor);
        const height = Math.floor(this.canvas.clientHeight * scaleFactor);

        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
            this.context.configure({
                device: this.device,
                format: this.format,
                alphaMode: 'premultiplied'
            });
        }
    }

    _render(timestamp) {
        if (!this.running) return;

        const timeDelta = timestamp - this.lastTime;
        this.lastTime = timestamp;
        this.time += timeDelta;

        this._updateCanvasSize();

        if (this.canvas.width === 0 || this.canvas.height === 0) {
            requestAnimationFrame((t) => this._render(t));
            return;
        }

        // 更新 Uniforms (viewportSize + snowSpeed + windForce + sizeMin + sizeMax)
        const { snowSpeed, windForce, sizeMin, sizeMax } = this.config;
        const uniformData = new Float32Array([
            this.canvas.width, this.canvas.height,
            snowSpeed, windForce, sizeMin, sizeMax
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // 更新 SimulationContext
        const bufferView = new DataView(this.simulationCtxLocalBuffer);
        bufferView.setFloat32(0, this.time, true);
        bufferView.setFloat32(4, timeDelta, true);
        bufferView.setFloat32(8, Math.random(), true);
        // particlesToSpawn: 每帧派生数量 = random * timeDelta
        bufferView.setInt32(12, Math.floor(Math.random() * timeDelta), true);
        this.device.queue.writeBuffer(this.simulationCtxBuffer, 0, this.simulationCtxLocalBuffer);

        const commandEncoder = this.device.createCommandEncoder();

        // Compute Pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(this.particleCount / 64);
        computePass.end();

        // Render Pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        // triangle-strip: 4 顶点/instance
        renderPass.draw(4, this.particleCount);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame((t) => this._render(t));
    }

    start() {
        this.running = true;
        this.lastTime = performance.now();
        this.time = 0;
        requestAnimationFrame((t) => this._render(t));
    }

    stop() {
        this.running = false;
    }

    destroy() {
        this.stop();
        try { this.particleBuffer?.destroy(); } catch (_) {}
        try { this.uniformBuffer?.destroy(); } catch (_) {}
        try { this.simulationCtxBuffer?.destroy(); } catch (_) {}
        if (this.device && typeof this.device.destroy === 'function') {
            try { this.device.destroy(); } catch (_) {}
        }
    }
}

// ==================== Canvas 2D 降级渲染器 ====================
class Canvas2DSnowRenderer {
    constructor(canvas, config) {
        this.canvas = canvas;
        this.config = config;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.running = false;
        this.lastTime = 0;
        this.snowflakeCache = null;
    }

    async init() {
        if (!this.ctx) {
            throw new Error('Canvas 2D not supported');
        }

        this._createSnowflakeSprite();

        // 粒子逐帧派生，初始为空
        this.spawnBudget = 0;
        return true;
    }

    _createSnowflakeSprite() {
        const size = 64;
        const offscreen = document.createElement('canvas');
        offscreen.width = size;
        offscreen.height = size;
        const ctx = offscreen.getContext('2d');

        const cx = size / 2;
        const cy = size / 2;
        const scale = size / 2 * 0.85;

        ctx.strokeStyle = 'white';
        ctx.lineCap = 'round';
        ctx.lineWidth = 2;

        // 绘制 6 个主枝 + 分叉
        for (let i = 0; i < 6; i++) {
            const angle = (i * 60) * Math.PI / 180;
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(angle);

            // 主枝
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(scale, 0);
            ctx.stroke();

            // 末端分叉
            ctx.beginPath();
            ctx.moveTo(scale * 0.75, 0);
            ctx.lineTo(scale * 0.95, scale * 0.18);
            ctx.moveTo(scale * 0.75, 0);
            ctx.lineTo(scale * 0.95, -scale * 0.18);
            ctx.stroke();

            // 中部分叉
            ctx.beginPath();
            ctx.moveTo(scale * 0.5, 0);
            ctx.lineTo(scale * 0.65, scale * 0.12);
            ctx.moveTo(scale * 0.5, 0);
            ctx.lineTo(scale * 0.65, -scale * 0.12);
            ctx.stroke();

            ctx.restore();
        }

        // 中心六边形
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
            const angle = (i * 60 - 30) * Math.PI / 180;
            const x = cx + Math.cos(angle) * scale * 0.15;
            const y = cy + Math.sin(angle) * scale * 0.15;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();

        // 添加发光效果
        ctx.globalCompositeOperation = 'lighter';
        ctx.filter = 'blur(2px)';
        ctx.globalAlpha = 0.3;
        ctx.drawImage(offscreen, 0, 0);

        this.snowflakeCache = offscreen;
    }

    _updateCanvasSize() {
        const dpr = window.devicePixelRatio || 1;
        const scaleFactor = Math.min(dpr, 2);
        const width = Math.floor(this.canvas.clientWidth * scaleFactor);
        const height = Math.floor(this.canvas.clientHeight * scaleFactor);

        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
    }

    _render(timestamp) {
        if (!this.running) return;

        const timeDelta = timestamp - this.lastTime;
        this.lastTime = timestamp;

        this._updateCanvasSize();

        const { width, height } = this.canvas;
        if (width === 0 || height === 0) {
            requestAnimationFrame((t) => this._render(t));
            return;
        }

        const maxParticles = Math.min(this.config.particleCount || 10000, 3000);

        // Reason: 使用配置的 sizeMin/sizeMax 代替硬编码值
        const { sizeMin, sizeMax, snowSpeed, windForce } = this.config;
        const sizeRange = sizeMax - sizeMin;

        // 派生新粒子
        this.spawnBudget += Math.random() * timeDelta;
        while (this.spawnBudget > 0 && this.particles.length < maxParticles) {
            const nearCamera = Math.random() > 0.95;
            const baseDistance = nearCamera ? 1 : 6;
            const distance = baseDistance + Math.random() * (nearCamera ? 1 : 3);

            // Reason: 保留双峰分布，sizeMin/sizeMax 作为缩放因子
            const largeFlake = Math.random() > 0.92;
            const baseRatio = largeFlake ? 0.5 : 0;
            const variationRatio = largeFlake ? 0.5 : 0.35;
            const size = (sizeMin + sizeRange * (baseRatio + Math.random() * variationRatio)) * ((distance / 9) * 0.1 + 1);

            // Reason: snowSpeed 同时缩放初速度
            const vyVariation = largeFlake ? size : 2;

            this.particles.push({
                x: Math.random() * width,
                y: -100,
                vx: -1.5 + Math.random() * 3,
                vy: Math.random() * vyVariation * snowSpeed,
                size,
                distance,
                opacity: 1 - distance / 9
            });
            this.spawnBudget -= 1;
        }

        // Reason: windForce 控制基础风力强度，默认 0.02 对应原硬编码 0.0002
        const wind = Math.sin(timestamp / 5000) * (windForce / 100);
        const dt = timeDelta / 10;

        this.ctx.clearRect(0, 0, width, height);

        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];

            p.vx += wind * dt;
            // Reason: snowSpeed 控制下落加速度，默认 1.0 对应原硬编码 0.03
            p.vy += 0.03 * snowSpeed * dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;

            if (p.y > height) {
                this.particles.splice(i, 1);
                continue;
            }

            // Reason: 与 WebGPU 版本保持一致，size 即为绘制尺寸
            const drawSize = p.size;
            this.ctx.globalAlpha = p.opacity;
            this.ctx.drawImage(
                this.snowflakeCache,
                p.x - drawSize / 2,
                p.y - drawSize / 2,
                drawSize,
                drawSize
            );
        }

        this.ctx.globalAlpha = 1;
        requestAnimationFrame((t) => this._render(t));
    }

    start() {
        this.running = true;
        this.lastTime = performance.now();
        requestAnimationFrame((t) => this._render(t));
    }

    stop() {
        this.running = false;
    }

    destroy() {
        this.stop();
        this.particles = [];
        this.snowflakeCache = null;
    }
}

// ==================== Worker 相关常量 ====================
const WORKER_HANDSHAKE_TIMEOUT_MS = 1500;

function supportsOffscreenWorkerCanvas() {
    return typeof Worker !== 'undefined' &&
        typeof OffscreenCanvas !== 'undefined' &&
        typeof HTMLCanvasElement !== 'undefined' &&
        typeof HTMLCanvasElement.prototype.transferControlToOffscreen === 'function';
}

function getCanvasRenderSize(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const scaleFactor = Math.min(dpr, 2);
    return {
        width: Math.floor(canvas.clientWidth * scaleFactor),
        height: Math.floor(canvas.clientHeight * scaleFactor),
        scaleFactor
    };
}

// ==================== Worker 代理渲染器 ====================
class WorkerWebGPUSnowRenderer {
    constructor(canvas, config, hooks = {}) {
        this.canvas = canvas;
        this.config = config;
        this.worker = null;
        this.running = false;
        this._ready = false;
        this._destroyed = false;
        this._handshakeTimer = null;
        this._resizeObserver = null;
        this._resolveReady = null;
        this._rejectReady = null;
        this._onFatalError = typeof hooks.onFatalError === 'function' ? hooks.onFatalError : null;
        this._handleWorkerMessage = this._handleWorkerMessage.bind(this);
        this._handleWorkerError = this._handleWorkerError.bind(this);
        this._handleResize = this._handleResize.bind(this);
    }

    async init() {
        if (!supportsOffscreenWorkerCanvas()) {
            throw new Error('OffscreenCanvas/Worker not supported');
        }

        const workerUrl = this.config.workerUrl;
        if (!workerUrl) throw new Error('Missing workerUrl');

        let worker;
        try {
            worker = new Worker(workerUrl, { type: 'module' });
        } catch (_) {
            worker = new Worker(workerUrl);
        }

        this.worker = worker;
        worker.addEventListener('message', this._handleWorkerMessage);
        worker.addEventListener('error', this._handleWorkerError);

        const readyPromise = new Promise((resolve, reject) => {
            this._resolveReady = resolve;
            this._rejectReady = reject;
        });

        const { width, height, scaleFactor } = getCanvasRenderSize(this.canvas);
        let offscreenCanvas;
        try {
            offscreenCanvas = this.canvas.transferControlToOffscreen();
        } catch (e) {
            this.destroy();
            throw e;
        }

        this._handshakeTimer = window.setTimeout(() => {
            if (this._ready) return;
            this._rejectReady?.(new Error('Worker handshake timeout'));
        }, WORKER_HANDSHAKE_TIMEOUT_MS);

        worker.postMessage(
            { type: 'init', canvas: offscreenCanvas, config: this.config, width, height, scaleFactor },
            [offscreenCanvas]
        );

        try {
            await readyPromise;
        } catch (e) {
            this.destroy();
            throw e;
        } finally {
            if (this._handshakeTimer) {
                clearTimeout(this._handshakeTimer);
                this._handshakeTimer = null;
            }
        }

        this._ready = true;
        this._bindResize();
        return true;
    }

    _bindResize() {
        if (!this.worker) return;
        if (typeof ResizeObserver !== 'undefined') {
            this._resizeObserver = new ResizeObserver(this._handleResize);
            this._resizeObserver.observe(this.canvas);
        } else {
            window.addEventListener('resize', this._handleResize, { passive: true });
        }
    }

    _handleResize() {
        if (!this.worker || !this._ready) return;
        const { width, height, scaleFactor } = getCanvasRenderSize(this.canvas);
        try { this.worker.postMessage({ type: 'resize', width, height, scaleFactor }); } catch (_) {}
    }

    _handleWorkerMessage(event) {
        const msg = event.data;
        if (!msg || typeof msg.type !== 'string') return;

        if (msg.type === 'ready') {
            this._resolveReady?.(msg);
            this._resolveReady = null;
            this._rejectReady = null;
            return;
        }

        if (msg.type === 'error') {
            const err = new Error(msg.message || 'Worker error');
            if (!this._ready && this._rejectReady) {
                this._rejectReady(err);
                this._resolveReady = null;
                this._rejectReady = null;
                return;
            }
            this._fatal(err);
        }
    }

    _handleWorkerError(event) {
        const err = event?.error || new Error(event?.message || 'Worker error');
        if (!this._ready && this._rejectReady) {
            this._rejectReady(err);
            this._resolveReady = null;
            this._rejectReady = null;
            return;
        }
        this._fatal(err);
    }

    _fatal(err) {
        if (this._destroyed) return;
        this.destroy();
        try { this._onFatalError?.(err); } catch (_) {}
    }

    start() {
        if (this._destroyed || !this.worker) return;
        this.running = true;
        try { this.worker.postMessage({ type: 'start' }); } catch (_) {}
    }

    stop() {
        this.running = false;
        try { this.worker?.postMessage({ type: 'stop' }); } catch (_) {}
    }

    destroy() {
        if (this._destroyed) return;
        this._destroyed = true;
        if (this._handshakeTimer) { clearTimeout(this._handshakeTimer); this._handshakeTimer = null; }
        if (this._resizeObserver) { try { this._resizeObserver.disconnect(); } catch (_) {} }
        window.removeEventListener('resize', this._handleResize);
        this.stop();
        if (this.worker) {
            try { this.worker.postMessage({ type: 'destroy' }); } catch (_) {}
            this.worker.removeEventListener('message', this._handleWorkerMessage);
            this.worker.removeEventListener('error', this._handleWorkerError);
            try { this.worker.terminate(); } catch (_) {}
            this.worker = null;
        }
    }
}

// ==================== 配置校验 ====================
const DEFAULT_CONFIG = {
    particleCount: 10000,
    snowSpeed: 1.0,
    windForce: 0.02,
    sizeMin: 3,
    sizeMax: 12,
    zIndex: 9999,
    useFallback: 'auto'
};

function normalizeConfig(raw) {
    const cfg = { ...DEFAULT_CONFIG, ...(raw || {}) };

    const particleCount = Number(cfg.particleCount);
    cfg.particleCount = Number.isFinite(particleCount) ? Math.max(1, Math.floor(particleCount)) : DEFAULT_CONFIG.particleCount;
    cfg.particleCount = Math.min(cfg.particleCount, 100000);

    // Reason: snowSpeed 控制下落速度系数，必须为正数
    const snowSpeed = Number(cfg.snowSpeed);
    cfg.snowSpeed = (Number.isFinite(snowSpeed) && snowSpeed > 0) ? snowSpeed : DEFAULT_CONFIG.snowSpeed;

    // Reason: windForce 可为正负数（正向右，负向左），0 表示无风
    const windForce = Number(cfg.windForce);
    cfg.windForce = Number.isFinite(windForce) ? windForce : DEFAULT_CONFIG.windForce;

    // Reason: sizeMin/sizeMax 定义雪花大小范围，自动交换确保 min <= max
    let sizeMin = Number(cfg.sizeMin);
    let sizeMax = Number(cfg.sizeMax);
    sizeMin = (Number.isFinite(sizeMin) && sizeMin > 0) ? sizeMin : DEFAULT_CONFIG.sizeMin;
    sizeMax = (Number.isFinite(sizeMax) && sizeMax > 0) ? sizeMax : DEFAULT_CONFIG.sizeMax;
    if (sizeMax < sizeMin) {
        [sizeMin, sizeMax] = [sizeMax, sizeMin];
    }
    cfg.sizeMin = sizeMin;
    cfg.sizeMax = sizeMax;

    const zIndex = Number(cfg.zIndex);
    cfg.zIndex = Number.isFinite(zIndex) ? Math.floor(zIndex) : DEFAULT_CONFIG.zIndex;

    if (!['auto', 'webgpu', 'canvas'].includes(cfg.useFallback)) {
        cfg.useFallback = DEFAULT_CONFIG.useFallback;
    }

    return cfg;
}

// ==================== 主初始化函数 ====================
function createSnowCanvas(config) {
    const canvas = document.createElement('canvas');
    canvas.id = 'webgpu-snow-canvas';
    canvas.style.setProperty('--snow-z-index', String(config.zIndex));
    canvas.setAttribute('aria-hidden', 'true');
    document.body.appendChild(canvas);
    return canvas;
}

async function initSnow() {
    if (typeof window._snowRendererCleanup === 'function') {
        window._snowRendererCleanup();
    }

    const config = normalizeConfig(window.SNOW_CONFIG);
    let canvas = createSnowCanvas(config);
    let renderer = null;

    // 回退到主线程的函数
    const fallbackToMainThread = async (reason) => {
        console.warn('[WebGPU Snow] Worker failed, falling back:', reason);
        try { renderer?.destroy(); } catch (_) {}
        try { canvas?.remove(); } catch (_) {}
        // Reason: transferControlToOffscreen 后不可回退，需重建 canvas
        canvas = createSnowCanvas(config);

        const useWebGPU = config.useFallback !== 'canvas' && navigator.gpu;
        if (useWebGPU) {
            try {
                renderer = new WebGPUSnowRenderer(canvas, config);
                await renderer.init();
                console.log('[WebGPU Snow] Using main-thread WebGPU');
                renderer.start();
                return;
            } catch (_) {}
        }
        renderer = new Canvas2DSnowRenderer(canvas, config);
        await renderer.init();
        console.log('[WebGPU Snow] Using Canvas 2D fallback');
        renderer.start();
    };

    // 尝试 Worker 模式
    const canTryWorker = config.useFallback !== 'canvas' &&
        supportsOffscreenWorkerCanvas() &&
        config.workerUrl;

    if (canTryWorker) {
        try {
            renderer = new WorkerWebGPUSnowRenderer(canvas, config, {
                onFatalError: fallbackToMainThread
            });
            await renderer.init();
            console.log('[WebGPU Snow] Using Worker WebGPU');
        } catch (e) {
            console.warn('[WebGPU Snow] Worker init failed:', e);
            try { renderer?.destroy(); } catch (_) {}
            renderer = null;
            // 重建 canvas
            try { canvas.remove(); } catch (_) {}
            canvas = createSnowCanvas(config);
        }
    }

    // 回退到主线程
    if (!renderer) {
        const useWebGPU = config.useFallback !== 'canvas' && navigator.gpu;
        if (useWebGPU) {
            try {
                renderer = new WebGPUSnowRenderer(canvas, config);
                await renderer.init();
                console.log('[WebGPU Snow] Using main-thread WebGPU');
            } catch (_) {
                renderer = null;
            }
        }
        if (!renderer) {
            renderer = new Canvas2DSnowRenderer(canvas, config);
            await renderer.init();
            console.log('[WebGPU Snow] Using Canvas 2D');
        }
    }

    renderer.start();

    window._snowRendererCleanup = () => {
        try { renderer?.destroy(); } catch (_) {}
        try { canvas?.remove(); } catch (_) {}
    };
}

window._snowInit = initSnow;

if (!window.__webgpuSnowPjaxBound) {
    window.__webgpuSnowPjaxBound = true;
    document.addEventListener('pjax:complete', () => {
        if (typeof window._snowRendererCleanup === 'function') window._snowRendererCleanup();
        if (typeof window._snowInit === 'function') window._snowInit();
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSnow);
} else {
    initSnow();
}
