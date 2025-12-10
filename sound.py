import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pyglet
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


# --- GLSL Shaders -----------------------------------------------------------
VERTEX_SHADER = """
#version 330 core
in vec3 position;
in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float timeSec;
uniform float accent;
uniform vec3 color;

out vec3 vNormal;
out vec3 vPosition;
out vec3 vColor;

void main() {
    vec4 worldPos = model * vec4(position, 1.0);
    // tiny vertex shimmer based on audio accent to keep motion silky
    worldPos.xyz += normal * accent * 0.02 * sin(timeSec + position.x * 4.0 + position.z * 2.0);
    vPosition = (view * worldPos).xyz;
    vNormal = mat3(view * model) * normal;
    vColor = color;
    gl_Position = projection * vec4(vPosition, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vNormal;
in vec3 vPosition;
in vec3 vColor;

out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float accent;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vPosition);
    vec3 V = normalize(viewPos - vPosition);
    vec3 R = reflect(-L, N);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(R, V), 0.0), 48.0);

    vec3 base = vColor;
    // Rim glow
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);
    vec3 rimColor = mix(base, vec3(1.0), rim * 0.4 * accent);

    vec3 ambient = 0.18 * base;
    vec3 diffuse = 0.72 * diff * rimColor;
    vec3 specular = 0.35 * spec * vec3(1.0);

    fragColor = vec4(ambient + diffuse + specular, 1.0);
}
"""


# --- Math helpers ----------------------------------------------------------
def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = center - eye
    forward /= np.linalg.norm(forward) if np.linalg.norm(forward) != 0 else 1

    side = np.cross(forward, up)
    side /= np.linalg.norm(side) if np.linalg.norm(side) != 0 else 1
    up_corrected = np.cross(side, forward)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = side
    m[1, :3] = up_corrected
    m[2, :3] = -forward

    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = -eye
    return m @ t


def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


# --- Audio Layer -----------------------------------------------------------
def read_wav(path: Path, target_rate: int) -> Tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    dtype = np.int16 if width == 2 else np.int8
    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_rate:
        ratio = target_rate / sample_rate
        x_old = np.linspace(0, 1, samples.size)
        x_new = np.linspace(0, 1, int(samples.size * ratio))
        samples = np.interp(x_new, x_old, samples)
        sample_rate = target_rate

    max_val = np.max(np.abs(samples)) or 1.0
    samples /= max_val
    return samples.astype(np.float32), sample_rate


@dataclass
class ProceduralTone:
    sample_rate: int
    frequency: float = 220.0
    phase: float = 0.0

    def next_chunk(self, size: int) -> np.ndarray:
        t = np.arange(size, dtype=np.float32) / self.sample_rate
        wave = 0.65 * np.sin(2 * np.pi * self.frequency * t + self.phase)
        self.phase = (self.phase + 2 * np.pi * self.frequency * size / self.sample_rate) % (2 * np.pi)
        return wave


@dataclass
class AudioStream:
    sample_rate: int
    source_path: Optional[str] = None
    tone: ProceduralTone = field(init=False)
    samples: Optional[np.ndarray] = field(init=False, default=None)
    cursor: int = 0

    def __post_init__(self) -> None:
        self.tone = ProceduralTone(sample_rate=self.sample_rate)
        if self.source_path and Path(self.source_path).exists():
            self.samples, _ = read_wav(Path(self.source_path), self.sample_rate)

    def adjust_frequency(self, delta: float) -> None:
        # Manual-only: no automatic frequency changes anywhere else
        self.tone.frequency = max(20.0, min(20000.0, self.tone.frequency + delta))

    def next_chunk(self, size: int) -> np.ndarray:
        if self.samples is None:
            return self.tone.next_chunk(size)

        end = self.cursor + size
        if end <= self.samples.size:
            chunk = self.samples[self.cursor:end]
            self.cursor = end
        else:
            part1 = self.samples[self.cursor:]
            part2 = self.samples[: end - self.samples.size]
            chunk = np.concatenate([part1, part2])
            self.cursor = end - self.samples.size
        return chunk


@dataclass
class AudioAnalyzer:
    sample_rate: int
    fft_size: int = 2048
    spectrum_bins: int = 256
    history: int = 180
    spectrogram: np.ndarray = field(init=False)
    last_rms: float = 0.0

    def __post_init__(self) -> None:
        self.spectrogram = np.zeros((self.history, self.spectrum_bins), dtype=np.float32)

    def analyze(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        window = np.hanning(min(self.fft_size, samples.size))
        windowed = samples[: window.size] * window
        fft = np.abs(np.fft.rfft(windowed, n=self.fft_size))
        spectrum = fft[: self.spectrum_bins]
        spectrum /= np.max(spectrum) if np.max(spectrum) != 0 else 1

        self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
        self.spectrogram[-1] = spectrum

        waveform = samples / (np.max(np.abs(samples)) if np.max(np.abs(samples)) != 0 else 1)
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        self.last_rms = 0.9 * self.last_rms + 0.1 * rms
        return waveform, spectrum, self.last_rms


# --- GL Utilities ----------------------------------------------------------
class Shader:
    def __init__(self) -> None:
        self.program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )

    def use(self) -> None:
        glUseProgram(self.program)

    def uniform_matrix(self, name: str, value: np.ndarray) -> None:
        glUniformMatrix4fv(glGetUniformLocation(self.program, name), 1, GL_TRUE, value)

    def uniform_vec3(self, name: str, value: Iterable[float]) -> None:
        glUniform3fv(glGetUniformLocation(self.program, name), 1, np.array(list(value), dtype=np.float32))

    def uniform_float(self, name: str, value: float) -> None:
        glUniform1f(glGetUniformLocation(self.program, name), value)


@dataclass
class Mesh:
    vao: int
    vbo: int
    ebo: Optional[int]
    mode: int
    count: int


def create_mesh(vertices: np.ndarray, indices: Optional[np.ndarray], mode: int) -> Mesh:
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

    stride = vertices.strides[0]
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

    ebo = None
    count = vertices.shape[0]
    if indices is not None:
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        count = indices.size

    glBindVertexArray(0)
    return Mesh(vao=vao, vbo=vbo, ebo=ebo, mode=mode, count=count)


# --- Geometry Builders -----------------------------------------------------
def build_wave_ribbon(samples: int) -> Mesh:
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    base_radius = 1.25
    half_width = 0.05

    vertices = []
    indices = []
    for idx, ang in enumerate(angles):
        dir_vec = np.array([math.cos(ang), 0.0, math.sin(ang)], dtype=np.float32)
        left = np.array([-math.sin(ang), 0.0, math.cos(ang)], dtype=np.float32)
        center = dir_vec * base_radius
        top = center + left * half_width
        bottom = center - left * half_width
        normal = dir_vec
        vertices.append(np.concatenate([top, normal]))
        vertices.append(np.concatenate([bottom, normal]))
        if idx < samples - 1:
            b = idx * 2
            indices.extend([b, b + 1, b + 2, b + 3])

    return create_mesh(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), GL_TRIANGLE_STRIP)


def build_spectrum_bars(bins: int) -> Mesh:
    xs = np.linspace(-1.6, 1.6, bins)
    bar_width = (xs[1] - xs[0]) * 0.55
    vertices = []
    indices = []
    for i, x in enumerate(xs):
        z = -1.35
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        base = len(vertices)
        vertices.append(np.concatenate([[x - bar_width, 0.0, z], normal]))
        vertices.append(np.concatenate([[x + bar_width, 0.0, z], normal]))
        vertices.append(np.concatenate([[x + bar_width, 0.1, z], normal]))
        vertices.append(np.concatenate([[x - bar_width, 0.1, z], normal]))
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    return create_mesh(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), GL_TRIANGLES)


def build_spectrogram_grid(history: int, bins: int) -> Mesh:
    t_axis = np.linspace(-3.0, 0.4, history)
    f_axis = np.linspace(-1.6, 1.6, bins)
    vertices = []
    indices = []
    for t_idx, t in enumerate(t_axis):
        for f in f_axis:
            pos = np.array([f, 0.0, t], dtype=np.float32)
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            vertices.append(np.concatenate([pos, normal]))

    for t_idx in range(history - 1):
        for f_idx in range(bins - 1):
            v0 = t_idx * bins + f_idx
            v1 = v0 + 1
            v2 = v0 + bins
            v3 = v2 + 1
            indices.extend([v0, v2, v1, v1, v2, v3])

    return create_mesh(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), GL_TRIANGLES)


def build_particle_cloud(count: int = 2048) -> Mesh:
    rng = np.random.default_rng(1)
    radius = rng.uniform(0.4, 1.8, size=count)
    theta = rng.uniform(0, 2 * np.pi, size=count)
    height = rng.uniform(-0.5, 0.5, size=count)
    positions = np.stack([
        radius * np.cos(theta),
        height,
        radius * np.sin(theta),
    ], axis=1).astype(np.float32)
    normals = positions / (np.linalg.norm(positions, axis=1, keepdims=True) + 1e-6)
    vertices = np.concatenate([positions, normals], axis=1)
    return create_mesh(vertices.astype(np.float32), None, GL_POINTS)


# --- Camera ----------------------------------------------------------------
@dataclass
class Camera:
    distance: float = 4.0
    height: float = 1.2
    yaw: float = 0.0
    pitch: float = -0.08
    mode: str = "orbit"  # orbit | manual | lock

    def update(self, keys: pyglet.window.key.KeyStateHandler, dt: float) -> None:
        if self.mode == "orbit":
            self.yaw += dt * 0.35
        if keys[pyglet.window.key.LEFT]:
            self.yaw -= dt * 1.2
        if keys[pyglet.window.key.RIGHT]:
            self.yaw += dt * 1.2
        if keys[pyglet.window.key.UP]:
            self.height += dt * 0.9
        if keys[pyglet.window.key.DOWN]:
            self.height -= dt * 0.9
        self.height = max(-1.0, min(2.5, self.height))

    def matrices(self, aspect: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        eye = np.array(
            [self.distance * math.sin(self.yaw), self.height, self.distance * math.cos(self.yaw)],
            dtype=np.float32,
        )
        center = np.array([0.0, 0.35, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = look_at(eye, center, up)
        projection = perspective(50, aspect, 0.1, 100.0)
        return eye, view, projection


# --- Visual Elements -------------------------------------------------------
class VisualElement:
    def __init__(self, mesh: Mesh, color: Tuple[float, float, float]) -> None:
        self.mesh = mesh
        self.color = color

    def update(self, *args) -> None:  # pragma: no cover - runtime visuals
        raise NotImplementedError


class WaveRibbon(VisualElement):
    def __init__(self, mesh: Mesh, amplitude: float = 0.55) -> None:
        super().__init__(mesh, (0.2, 0.95, 0.65))
        self.amplitude = amplitude

    def update(self, waveform: np.ndarray) -> float:
        sample_count = waveform.size
        angle_step = 2 * np.pi / sample_count
        base_radius = 1.25
        half_width = 0.05
        mapped = np.empty((sample_count * 2, 6), dtype=np.float32)
        for i in range(sample_count):
            ang = i * angle_step
            dir_vec = np.array([math.cos(ang), 0.0, math.sin(ang)], dtype=np.float32)
            left = np.array([-math.sin(ang), 0.0, math.cos(ang)], dtype=np.float32)
            radius = base_radius + waveform[i] * self.amplitude
            center = dir_vec * radius
            top = center + left * half_width
            bottom = center - left * half_width
            normal = dir_vec
            mapped[2 * i, :3] = top
            mapped[2 * i, 3:] = normal
            mapped[2 * i + 1, :3] = bottom
            mapped[2 * i + 1, 3:] = normal

        glBindBuffer(GL_ARRAY_BUFFER, self.mesh.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, mapped.nbytes, mapped)
        return float(np.max(np.abs(waveform)))


class SpectrumBars(VisualElement):
    def __init__(self, mesh: Mesh, scale: float = 1.4) -> None:
        super().__init__(mesh, (0.95, 0.4, 0.2))
        self.scale = scale

    def update(self, spectrum: np.ndarray) -> float:
        xs = np.linspace(-1.6, 1.6, spectrum.size)
        bar_width = (xs[1] - xs[0]) * 0.55
        vertices = []
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for x, value in zip(xs, spectrum):
            height = 0.1 + value * self.scale
            z = -1.35
            vertices.append(np.concatenate([[x - bar_width, 0.0, z], normal]))
            vertices.append(np.concatenate([[x + bar_width, 0.0, z], normal]))
            vertices.append(np.concatenate([[x + bar_width, height, z], normal]))
            vertices.append(np.concatenate([[x - bar_width, height, z], normal]))

        data = np.array(vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        return float(np.mean(spectrum))


class SpectrogramSurface(VisualElement):
    def __init__(self, mesh: Mesh, analyzer: AudioAnalyzer, height_scale: float = 1.6) -> None:
        super().__init__(mesh, (0.15, 0.5, 0.95))
        self.analyzer = analyzer
        self.height_scale = height_scale

    def update(self) -> float:
        history, bins = self.analyzer.spectrogram.shape
        t_axis = np.linspace(-3.0, 0.4, history)
        f_axis = np.linspace(-1.6, 1.6, bins)
        vertices = np.empty((history * bins, 6), dtype=np.float32)
        for t_idx, t in enumerate(t_axis):
            for f_idx, f in enumerate(f_axis):
                amplitude = self.analyzer.spectrogram[t_idx, f_idx]
                height = amplitude * self.height_scale
                pos = np.array([f, height, t], dtype=np.float32)
                normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                idx = t_idx * bins + f_idx
                vertices[idx, :3] = pos
                vertices[idx, 3:] = normal

        glBindBuffer(GL_ARRAY_BUFFER, self.mesh.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        return float(np.max(self.analyzer.spectrogram))


class ParticleField(VisualElement):
    def __init__(self, mesh: Mesh) -> None:
        super().__init__(mesh, (0.85, 0.9, 1.0))
        self.intensity = 0.0

    def update(self, spectrum: np.ndarray) -> float:
        energy = float(np.mean(spectrum))
        self.intensity = 0.9 * self.intensity + 0.1 * energy
        return self.intensity


# --- Visualizer ------------------------------------------------------------
class SonicMorphVisualizer:
    def __init__(self, width: int, height: int, audio_path: Optional[str]) -> None:
        self.width = width
        self.height = height
        self.sample_rate = 44100
        self.audio_stream = AudioStream(sample_rate=self.sample_rate, source_path=audio_path)
        self.analyzer = AudioAnalyzer(sample_rate=self.sample_rate)

        self.window = pyglet.window.Window(
            width=self.width,
            height=self.height,
            caption="SonicMorph – Advanced Audio Geometry",
            config=pyglet.gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3),
            resizable=False,
        )
        self.window.push_handlers(self)
        self.keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys)

        self.shader = Shader()
        self.camera = Camera()

        self.wave_mesh = build_wave_ribbon(1024)
        self.spectrum_mesh = build_spectrum_bars(self.analyzer.spectrum_bins)
        self.spectrogram_mesh = build_spectrogram_grid(self.analyzer.history, self.analyzer.spectrum_bins)
        self.particle_mesh = build_particle_cloud()

        self.wave = WaveRibbon(self.wave_mesh)
        self.spectrum = SpectrumBars(self.spectrum_mesh)
        self.spectrogram = SpectrogramSurface(self.spectrogram_mesh, self.analyzer)
        self.particles = ParticleField(self.particle_mesh)

        self.modes = ["waveform", "spectrum", "spectrogram", "particles", "all"]
        self.active_mode = "all"
        self.accent = 0.0
        self.last_time = time.time()

        pyglet.clock.schedule_interval(self._tick, 1 / 60.0)

    # -- Lifecycle -----------------------------------------------------
    def _tick(self, dt: float) -> None:
        now = time.time()
        delta = now - self.last_time
        self.last_time = now

        self.camera.update(self.keys, delta)
        samples = self.audio_stream.next_chunk(2048)
        waveform, spectrum, rms = self.analyzer.analyze(samples)

        accent_wave = self.wave.update(waveform)
        accent_spec = self.spectrum.update(spectrum)
        accent_specgram = self.spectrogram.update()
        accent_particles = self.particles.update(spectrum)

        self.accent = max(accent_wave, accent_spec, accent_specgram, accent_particles, rms)
        self.window.invalid = True

    # -- Rendering -----------------------------------------------------
    def _render_mesh(self, element: VisualElement, view: np.ndarray, projection: np.ndarray, point_size: float = 3.5) -> None:
        mesh = element.mesh
        self.shader.use()
        model = np.eye(4, dtype=np.float32)
        self.shader.uniform_matrix("model", model)
        self.shader.uniform_matrix("view", view)
        self.shader.uniform_matrix("projection", projection)
        self.shader.uniform_vec3("color", element.color)
        self.shader.uniform_vec3("lightPos", (2.6, 2.7, 2.4))
        self.shader.uniform_vec3("viewPos", (0.0, 0.0, 5.0))
        self.shader.uniform_float("timeSec", float(time.time()))
        self.shader.uniform_float("accent", float(self.accent))

        glBindVertexArray(mesh.vao)
        if mesh.mode == GL_POINTS:
            glEnable(GL_PROGRAM_POINT_SIZE)
            glPointSize(point_size + self.accent * 4.0)
        if mesh.ebo is not None:
            glDrawElements(mesh.mode, mesh.count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(mesh.mode, 0, mesh.count)
        glBindVertexArray(0)

    def on_draw(self) -> None:  # pragma: no cover - runtime visuals
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.015, 0.018, 0.035, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        eye, view, projection = self.camera.matrices(self.width / self.height)

        if self.active_mode in ("waveform", "all"):
            self._render_mesh(self.wave, view, projection)
        if self.active_mode in ("spectrum", "all"):
            self._render_mesh(self.spectrum, view, projection)
        if self.active_mode in ("spectrogram", "all"):
            self._render_mesh(self.spectrogram, view, projection)
        if self.active_mode in ("particles", "all"):
            self._render_mesh(self.particles, view, projection, point_size=2.7)

    # -- Input ---------------------------------------------------------
    def on_key_press(self, symbol: int, modifiers: int) -> None:  # pragma: no cover - runtime visuals
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
            return
        if symbol == pyglet.window.key.C:
            next_mode = {"orbit": "manual", "manual": "lock", "lock": "orbit"}
            self.camera.mode = next_mode.get(self.camera.mode, "orbit")
        if symbol == pyglet.window.key.M:
            idx = self.modes.index(self.active_mode)
            self.active_mode = self.modes[(idx + 1) % len(self.modes)]
        if symbol == pyglet.window.key.PLUS or symbol == pyglet.window.key.NUM_ADD:
            self.audio_stream.adjust_frequency(20.0)
        if symbol == pyglet.window.key.MINUS or symbol == pyglet.window.key.NUM_SUBTRACT:
            self.audio_stream.adjust_frequency(-20.0)

    # -- Run -----------------------------------------------------------
    def run(self) -> None:
        pyglet.app.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="SonicMorph – high-end 3D audio visualization")
    parser.add_argument("--audio", type=str, default=None, help="Path to WAV file; otherwise uses manual tone")
    args = parser.parse_args()

    visualizer = SonicMorphVisualizer(1280, 800, args.audio)
    visualizer.run()


if __name__ == "__main__":
    main()
