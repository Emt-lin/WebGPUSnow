/**
 * WebGPU Snow Worker
 * 独立 Worker：接收 OffscreenCanvas 并在 Worker 内执行 WebGPU 渲染
 *
 * Message protocol (main -> worker):
 * - init    { canvas: OffscreenCanvas, config, width, height, scaleFactor }
 * - resize  { width, height, scaleFactor }
 * - start   {}
 * - stop    {}
 * - destroy {}
 *
 * Message protocol (worker -> main):
 * - ready { particleCount }
 * - error { phase, message, stack }
 */

// ==================== WGSL Shader 代码 ====================
const SHADER_CODE = `
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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> writableParticles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> simulationCtx: SimulationContext;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct Pcg32RandomState { state: u32 }
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

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) globalInvocationId: vec3u) {
    initRand(globalInvocationId.x, simulationCtx.randSeed);
    let timeDelta = simulationCtx.timeDelta / 10.0;
    let wind = sin(simulationCtx.time / 5000.0) * (uniforms.windForce / 100.0);

    var particle = writableParticles[globalInvocationId.x];

    if (particle.spawned == 0 || particle.position.y > uniforms.viewportSize.y) {
        if (atomicSub(&simulationCtx.particlesToSpawn, 1) > 0) {
            particle.position.x = rand() * uniforms.viewportSize.x;
            particle.position.y = -100.0;

            let nearCamera = rand() > 0.95;
            let baseDistance = select(6.0, 1.0, nearCamera);
            let distanceVariation = select(3.0, 1.0, nearCamera);
            let distance = baseDistance + rand() * distanceVariation;
            particle.distance = distance;

            // 保留双峰分布，sizeMin/sizeMax 作为缩放因子
            let largeFlake = rand() > 0.92;
            let sizeRange = uniforms.sizeMax - uniforms.sizeMin;
            let baseRatio = select(0.0, 0.5, largeFlake);
            let variationRatio = select(0.35, 0.5, largeFlake);
            let size = uniforms.sizeMin + sizeRange * (baseRatio + rand() * variationRatio);
            let distanceFactor = (distance / 9.0) * 0.1 + 1.0;
            particle.size = vec2f(size) * distanceFactor;

            // snowSpeed 同时缩放初速度
            let vyVariation = select(2.0, particle.size.y, largeFlake);
            particle.velocity = vec2f(-1.5 + rand() * 3.0, rand() * vyVariation * uniforms.snowSpeed);

            particle.opacity = 1.0 - distance / 9.0;
            particle.spawned = 1;
        }
    }

    particle.velocity.x += wind * timeDelta;
    particle.velocity.y += 0.03 * uniforms.snowSpeed * timeDelta;
    particle.position += particle.velocity * timeDelta;
    writableParticles[globalInvocationId.x] = particle;
}

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
    const vertices = array<vec2f, 4>(
        vec2f(-0.5, -0.5), vec2f(0.5, -0.5),
        vec2f(-0.5, 0.5), vec2f(0.5, 0.5)
    );

    let particle = particles[in.particleIndex];
    let particlePosNorm = particle.position / uniforms.viewportSize;
    let particleSizeNorm = particle.size / uniforms.viewportSize;
    let vertexPos = vertices[in.vertexIndex];
    let pos = (particlePosNorm + particleSizeNorm * vertexPos) * 2.0 - 1.0;

    var out: QuadVertexOutput;
    out.position = vec4f(pos.x, -pos.y, 0.0, 1.0);
    out.uv = vertexPos * 2.0;
    out.distance = particle.distance;
    out.opacity = particle.opacity;
    return out;
}

@fragment
fn particleFragment(in: QuadVertexOutput) -> @location(0) vec4f {
    if (in.position.x > uniforms.viewportSize.x || in.position.y > uniforms.viewportSize.y) {
        discard;
    }
    let dis = pow(distance(in.uv.xy, vec2f(0.0, 0.0)), in.distance);
    let brightness = (1.0 - dis) * in.opacity;
    return vec4f(brightness);
}
`;

// ==================== 常量 ====================
const PARTICLE_COUNT = 10000;
const SIZEOF_F32 = 4;
const PARTICLE_STRUCT_SIZE = 40;
const SIMULATION_CTX_SIZE = 16;
const UNIFORMS_SIZE = 32;

function post(type, payload) {
    try { self.postMessage({ type, ...(payload || {}) }); } catch (_) {}
}

function postError(err, phase) {
    const e = err instanceof Error ? err : new Error(String(err));
    post('error', { phase: phase || 'unknown', message: e.message, stack: e.stack || '' });
}

// ==================== Worker 渲染器类 ====================
class WebGPUSnowWorkerRenderer {
    constructor(canvas, config) {
        this.canvas = canvas;
        this.config = config || {};
        this.device = null;
        this.context = null;
        this.format = null;
        this.running = false;
        this.time = 0;
        this.lastTime = 0;
        this.timer = null;
        this.particleCount = 0;
        this.simulationCtxLocalBuffer = new ArrayBuffer(SIMULATION_CTX_SIZE);
    }

    async init(width, height) {
        if (!self.navigator || !self.navigator.gpu) {
            throw new Error('WebGPU not supported in Worker');
        }

        const adapter = await self.navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
        if (!adapter) throw new Error('Failed to get GPU adapter');

        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        if (!this.context) throw new Error('Failed to get WebGPU canvas context');

        this.format = self.navigator.gpu.getPreferredCanvasFormat
            ? self.navigator.gpu.getPreferredCanvasFormat()
            : 'bgra8unorm';

        this.resize(width, height);
        this._createResources();

        this.device.lost?.then((info) => {
            postError(new Error(info?.message || 'GPU device lost'), 'device-lost');
        }).catch(() => {});

        return true;
    }

    resize(width, height) {
        const w = Math.max(0, Math.floor(Number(width) || 0));
        const h = Math.max(0, Math.floor(Number(height) || 0));
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
        }
        this._configureContext();
    }

    _configureContext() {
        if (!this.context || !this.device || !this.format) return;
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied'
        });
    }

    _createResources() {
        const requestedCount = this.config.particleCount || PARTICLE_COUNT;
        const particleCount = Math.max(64, Math.floor(requestedCount / 64) * 64);

        this.shaderModule = this.device.createShaderModule({ code: SHADER_CODE });

        this.uniformBuffer = this.device.createBuffer({
            size: UNIFORMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.particleBuffer = this.device.createBuffer({
            size: particleCount * PARTICLE_STRUCT_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
        });

        this.simulationCtxBuffer = this.device.createBuffer({
            size: SIMULATION_CTX_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const computeBGL = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });

        const renderBGL = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            ]
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBGL,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.particleBuffer } },
                { binding: 2, resource: { buffer: this.simulationCtxBuffer } },
            ]
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBGL,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.particleBuffer } },
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
            compute: { module: this.shaderModule, entryPoint: 'updateParticles' },
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
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

    _render(timestamp) {
        if (!this.running) return;

        const timeDelta = timestamp - this.lastTime;
        this.lastTime = timestamp;
        this.time += timeDelta;

        const width = this.canvas.width;
        const height = this.canvas.height;
        if (width === 0 || height === 0) {
            this._scheduleNext();
            return;
        }

        const { snowSpeed = 1.0, windForce = 0.02, sizeMin = 3, sizeMax = 12 } = this.config;
        const uniformData = new Float32Array([
            width, height,
            snowSpeed, windForce, sizeMin, sizeMax
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const bufferView = new DataView(this.simulationCtxLocalBuffer);
        bufferView.setFloat32(0, this.time, true);
        bufferView.setFloat32(4, timeDelta, true);
        bufferView.setFloat32(8, Math.random(), true);
        bufferView.setInt32(12, Math.floor(Math.random() * timeDelta), true);
        this.device.queue.writeBuffer(this.simulationCtxBuffer, 0, this.simulationCtxLocalBuffer);

        const commandEncoder = this.device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(this.particleCount / 64);
        computePass.end();

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
        renderPass.draw(4, this.particleCount);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
        this._scheduleNext();
    }

    _scheduleNext() {
        if (!this.running) return;
        this.timer = self.setTimeout(() => this._render(self.performance.now()), 16);
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.lastTime = self.performance.now();
        this.time = 0;
        this.timer = self.setTimeout(() => this._render(self.performance.now()), 0);
    }

    stop() {
        this.running = false;
        if (this.timer) {
            self.clearTimeout(this.timer);
            this.timer = null;
        }
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

// ==================== Worker 消息处理 ====================
let renderer = null;
let initialized = false;

self.addEventListener('message', async (event) => {
    const msg = event.data;
    if (!msg || typeof msg.type !== 'string') return;

    if (msg.type === 'init') {
        if (initialized) return;
        initialized = true;
        try {
            const canvas = msg.canvas;
            if (!canvas || typeof canvas.getContext !== 'function') {
                throw new Error('Invalid OffscreenCanvas');
            }
            renderer = new WebGPUSnowWorkerRenderer(canvas, msg.config || {});
            await renderer.init(msg.width, msg.height);
            renderer.start();
            post('ready', { particleCount: renderer.particleCount });
        } catch (e) {
            postError(e, 'init');
        }
        return;
    }

    if (!renderer) return;

    try {
        if (msg.type === 'resize') {
            renderer.resize(msg.width, msg.height);
        } else if (msg.type === 'start') {
            renderer.start();
        } else if (msg.type === 'stop') {
            renderer.stop();
        } else if (msg.type === 'destroy') {
            renderer.destroy();
            renderer = null;
            self.close();
        }
    } catch (e) {
        postError(e, msg.type);
    }
});
