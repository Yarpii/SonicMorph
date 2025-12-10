import argparse
import io
import math
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyglet
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


VERTEX_SHADER = """
#version 330 core
in vec3 position;
in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 baseColor;

out vec3 vNormal;
out vec3 vPosition;
out vec3 vColor;

void main() {
    vec4 worldPos = model * vec4(position, 1.0);
    vPosition = (view * worldPos).xyz;
    vNormal = mat3(view * model) * normal;
    vColor = baseColor;
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

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vPosition);
    vec3 V = normalize(viewPos - vPosition);
    vec3 R = reflect(-L, N);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(R, V), 0.0), 32.0);

    vec3 ambient = 0.15 * vColor;
    vec3 diffuse = 0.75 * diff * vColor;
    vec3 specular = 0.4 * spec * vec3(1.0);

    fragColor = vec4(ambient + diffuse + specular, 1.0);
}
"""


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = center - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm != 0:
        forward /= forward_norm

    side = np.cross(forward, up)
    side_norm = np.linalg.norm(side)
    if side_norm != 0:
        side /= side_norm

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


def read_wav_file(path: Path, target_rate: int) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    dtype = np.int16 if sample_width == 2 else np.int8
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    if sample_rate != target_rate:
        ratio = target_rate / sample_rate
        x_old = np.linspace(0, 1, data.size)
        x_new = np.linspace(0, 1, int(data.size * ratio))
        data = np.interp(x_new, x_old, data)
        sample_rate = target_rate

    data /= np.max(np.abs(data)) if np.max(np.abs(data)) != 0 else 1
    return data.astype(np.float32), sample_rate


def generate_tone(sample_rate: int, freq: float = 440.0, duration: float = 5.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave_data = 0.7 * np.sin(2 * np.pi * freq * t)
    return wave_data.astype(np.float32)


@dataclass
class AudioStream:
    samples: np.ndarray
    sample_rate: int
    cursor: int = 0

    @classmethod
    def from_source(cls, path: Optional[str], sample_rate: int) -> "AudioStream":
        if path is not None and Path(path).exists():
            samples, rate = read_wav_file(Path(path), sample_rate)
        else:
            samples = generate_tone(sample_rate)
            rate = sample_rate
        return cls(samples=samples, sample_rate=rate)

    def next_chunk(self, size: int) -> np.ndarray:
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
    spectrogram_history: int = 120
    spectrum_bins: int = 128
    spectrogram: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.spectrogram = np.zeros((self.spectrogram_history, self.spectrum_bins), dtype=np.float32)

    def analyze(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        window = np.hanning(min(samples.size, self.fft_size))
        windowed = samples[: window.size] * window
        spectrum = np.abs(np.fft.rfft(windowed, n=self.fft_size))
        spectrum = spectrum[: self.spectrum_bins]
        spectrum /= np.max(spectrum) if np.max(spectrum) != 0 else 1

        self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
        self.spectrogram[-1] = spectrum

        waveform = samples / (np.max(np.abs(samples)) if np.max(np.abs(samples)) != 0 else 1)
        return waveform, spectrum


@dataclass
class Mesh:
    vao: int
    vbo: int
    ebo: Optional[int]
    count: int
    mode: int


class SonicMorphVisualizer:
    def __init__(self, width: int, height: int, audio_path: Optional[str]) -> None:
        self.width = width
        self.height = height
        self.sample_rate = 44100
        self.audio_stream = AudioStream.from_source(audio_path, self.sample_rate)
        self.analyzer = AudioAnalyzer(sample_rate=self.sample_rate)

        self.window = pyglet.window.Window(
            width=self.width,
            height=self.height,
            caption="SonicMorph - Real Audio Geometry",
            config=pyglet.gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3),
            resizable=False,
        )
        self.window.push_handlers(self)
        self.keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys)

        self.shader = self._build_shader()
        self.waveform_mesh = self._create_waveform_mesh(1024)
        self.spectrum_mesh = self._create_spectrum_mesh(self.analyzer.spectrum_bins)
        self.spectrogram_mesh = self._create_spectrogram_mesh(
            self.analyzer.spectrogram_history, self.analyzer.spectrum_bins
        )

        self.camera_angle = 0.0
        self.camera_distance = 4.0
        self.camera_height = 1.2
        self.auto_orbit = True

        self.last_time = time.time()
        pyglet.clock.schedule_interval(self._tick, 1 / 60.0)

    def _build_shader(self) -> int:
        return compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )

    def _create_buffered_mesh(self, vertices: np.ndarray, indices: Optional[np.ndarray], mode: int) -> Mesh:
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
        return Mesh(vao=vao, vbo=vbo, ebo=ebo, count=count, mode=mode)

    def _create_waveform_mesh(self, sample_count: int) -> Mesh:
        angles = np.linspace(0, 2 * np.pi, sample_count, endpoint=False)
        base_radius = 1.2
        ribbon_width = 0.04

        vertices = []
        indices = []
        for idx, angle in enumerate(angles):
            dir_vec = np.array([math.cos(angle), 0.0, math.sin(angle)], dtype=np.float32)
            left = np.array([-math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)
            center = dir_vec * base_radius

            top = center + left * ribbon_width
            bottom = center - left * ribbon_width
            normal = dir_vec
            vertices.append(np.concatenate([top, normal]))
            vertices.append(np.concatenate([bottom, normal]))
            if idx < sample_count - 1:
                base = idx * 2
                indices.extend([base, base + 1, base + 2, base + 3])

        vertex_array = np.array(vertices, dtype=np.float32)
        index_array = np.array(indices, dtype=np.uint32)
        return self._create_buffered_mesh(vertex_array, index_array, GL_TRIANGLE_STRIP)

    def _create_spectrum_mesh(self, bins: int) -> Mesh:
        x_positions = np.linspace(-1.5, 1.5, bins)
        bar_width = (x_positions[1] - x_positions[0]) * 0.6
        vertices = []
        indices = []
        for i, x in enumerate(x_positions):
            z = -1.2
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            base = len(vertices)
            vertices.append(np.concatenate([[x - bar_width, 0.0, z], normal]))
            vertices.append(np.concatenate([[x + bar_width, 0.0, z], normal]))
            vertices.append(np.concatenate([[x + bar_width, 0.1, z], normal]))
            vertices.append(np.concatenate([[x - bar_width, 0.1, z], normal]))
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

        vertex_array = np.array(vertices, dtype=np.float32)
        index_array = np.array(indices, dtype=np.uint32)
        return self._create_buffered_mesh(vertex_array, index_array, GL_TRIANGLES)

    def _create_spectrogram_mesh(self, history: int, bins: int) -> Mesh:
        time_axis = np.linspace(-2.5, 0.5, history)
        freq_axis = np.linspace(-1.5, 1.5, bins)
        vertices = []
        indices = []
        for t_idx, t in enumerate(time_axis):
            for f_idx, f in enumerate(freq_axis):
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

        vertex_array = np.array(vertices, dtype=np.float32)
        index_array = np.array(indices, dtype=np.uint32)
        return self._create_buffered_mesh(vertex_array, index_array, GL_TRIANGLES)

    def _update_waveform_mesh(self, waveform: np.ndarray) -> None:
        sample_count = waveform.size
        angle_step = 2 * np.pi / sample_count
        mapped = np.empty((sample_count * 2, 6), dtype=np.float32)
        base_radius = 1.2
        ribbon_width = 0.04
        for i in range(sample_count):
            angle = i * angle_step
            dir_vec = np.array([math.cos(angle), 0.0, math.sin(angle)], dtype=np.float32)
            left = np.array([-math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)
            center = dir_vec * (base_radius + waveform[i] * 0.4)
            top = center + left * ribbon_width
            bottom = center - left * ribbon_width
            normal = dir_vec
            mapped[2 * i, :3] = top
            mapped[2 * i, 3:] = normal
            mapped[2 * i + 1, :3] = bottom
            mapped[2 * i + 1, 3:] = normal

        glBindBuffer(GL_ARRAY_BUFFER, self.waveform_mesh.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, mapped.nbytes, mapped)

    def _update_spectrum_mesh(self, spectrum: np.ndarray) -> None:
        vertices = []
        x_positions = np.linspace(-1.5, 1.5, spectrum.size)
        bar_width = (x_positions[1] - x_positions[0]) * 0.6
        for x, value in zip(x_positions, spectrum):
            height = 0.15 + value * 1.2
            z = -1.2
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            vertices.append(np.concatenate([[x - bar_width, 0.0, z], normal]))
            vertices.append(np.concatenate([[x + bar_width, 0.0, z], normal]))
            vertices.append(np.concatenate([[x + bar_width, height, z], normal]))
            vertices.append(np.concatenate([[x - bar_width, height, z], normal]))

        vertex_array = np.array(vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.spectrum_mesh.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_array.nbytes, vertex_array)

    def _update_spectrogram_mesh(self) -> None:
        history, bins = self.analyzer.spectrogram.shape
        time_axis = np.linspace(-2.5, 0.5, history)
        freq_axis = np.linspace(-1.5, 1.5, bins)
        vertices = np.empty((history * bins, 6), dtype=np.float32)
        for t_idx, t in enumerate(time_axis):
            for f_idx, f in enumerate(freq_axis):
                amplitude = self.analyzer.spectrogram[t_idx, f_idx]
                height = amplitude * 1.2
                pos = np.array([f, height, t], dtype=np.float32)
                normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                idx = t_idx * bins + f_idx
                vertices[idx, :3] = pos
                vertices[idx, 3:] = normal

        glBindBuffer(GL_ARRAY_BUFFER, self.spectrogram_mesh.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

    def _update_camera(self, delta_time: float) -> None:
        if self.auto_orbit:
            self.camera_angle += delta_time * 0.4
        if self.keys[pyglet.window.key.LEFT]:
            self.camera_angle -= delta_time * 1.2
        if self.keys[pyglet.window.key.RIGHT]:
            self.camera_angle += delta_time * 1.2
        if self.keys[pyglet.window.key.UP]:
            self.camera_height += delta_time * 0.8
        if self.keys[pyglet.window.key.DOWN]:
            self.camera_height -= delta_time * 0.8

    def _camera_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        eye = np.array(
            [
                self.camera_distance * math.sin(self.camera_angle),
                self.camera_height,
                self.camera_distance * math.cos(self.camera_angle),
            ],
            dtype=np.float32,
        )
        center = np.array([0.0, 0.4, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = look_at(eye, center, up)
        projection = perspective(50, self.width / self.height, 0.1, 100.0)
        return view, projection

    def _render_mesh(self, mesh: Mesh, color: Tuple[float, float, float], view: np.ndarray, projection: np.ndarray) -> None:
        glUseProgram(self.shader)
        model = np.eye(4, dtype=np.float32)

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_TRUE, model)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_TRUE, projection)
        glUniform3fv(glGetUniformLocation(self.shader, "baseColor"), 1, np.array(color, dtype=np.float32))
        glUniform3fv(glGetUniformLocation(self.shader, "lightPos"), 1, np.array([2.5, 2.5, 2.5], dtype=np.float32))
        glUniform3fv(glGetUniformLocation(self.shader, "viewPos"), 1, np.array([0.0, 0.0, 5.0], dtype=np.float32))

        glBindVertexArray(mesh.vao)
        if mesh.ebo is not None:
            glDrawElements(mesh.mode, mesh.count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(mesh.mode, 0, mesh.count)
        glBindVertexArray(0)

    def _tick(self, dt: float) -> None:
        now = time.time()
        delta_time = now - self.last_time
        self.last_time = now

        self._update_camera(delta_time)
        chunk = self.audio_stream.next_chunk(2048)
        waveform, spectrum = self.analyzer.analyze(chunk)
        self._update_waveform_mesh(waveform)
        self._update_spectrum_mesh(spectrum)
        self._update_spectrogram_mesh()
        self.window.invalidate()

    def on_draw(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.02, 0.02, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view, projection = self._camera_matrices()
        self._render_mesh(self.spectrogram_mesh, (0.15, 0.45, 0.85), view, projection)
        self._render_mesh(self.spectrum_mesh, (0.9, 0.35, 0.15), view, projection)
        self._render_mesh(self.waveform_mesh, (0.2, 0.9, 0.6), view, projection)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
        elif symbol == pyglet.window.key.O:
            self.auto_orbit = not self.auto_orbit

    def run(self) -> None:
        pyglet.app.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize real audio as 3D geometry.")
    parser.add_argument("--audio", type=str, default=None, help="Path to a WAV file to visualize")
    args = parser.parse_args()

    visualizer = SonicMorphVisualizer(width=1280, height=800, audio_path=args.audio)
    visualizer.run()


if __name__ == "__main__":
    main()
