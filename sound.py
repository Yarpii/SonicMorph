import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import time
import math

# Vertex shader with parametric morphing capabilities
VERTEX_SHADER = """
#version 330
in vec2 position;  // Parametric coordinates (u,v)
uniform float time;
uniform float frequency;
uniform float amplitude;
uniform int shapeType;  // 0=sphere, 1=torus, 2=flower, 3=square, etc.
uniform float morphFactor;  // For smooth transitions between shapes
uniform float twist;  // For adding twists to the shape
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Base parametric equations
vec3 createSphere(vec2 p) {
    float u = p.x * 2.0 * 3.14159;
    float v = p.y * 3.14159;
    return vec3(
        cos(u) * sin(v),
        cos(v),
        sin(u) * sin(v)
    );
}

vec3 createTorus(vec2 p) {
    float u = p.x * 2.0 * 3.14159;
    float v = p.y * 2.0 * 3.14159;
    float R = 0.7;  // Major radius
    float r = 0.3;  // Minor radius
    return vec3(
        (R + r * cos(v)) * cos(u),
        r * sin(v),
        (R + r * cos(v)) * sin(u)
    );
}

vec3 createFlower(vec2 p) {
    float u = p.x * 2.0 * 3.14159;
    float v = p.y * 3.14159;
    float petals = 5.0 + amplitude * 10.0;  // Number of petals varies with amplitude
    float r = 0.5 + 0.3 * cos(petals * u);  // Flower shape
    return vec3(
        r * cos(u) * sin(v),
        cos(v),
        r * sin(u) * sin(v)
    );
}

vec3 createSquare(vec2 p) {
    float u = p.x * 2.0 - 1.0;  // Map to [-1,1]
    float v = p.y * 2.0 - 1.0;  // Map to [-1,1]
    
    // Create a rounded square using smoothing
    float r = 0.9;  // Size of square
    float smoothFactor = 0.1;  // Smoothing factor
    
    // Smooth max function for corners
    float x = u < 0.0 ? -max(abs(u), smoothFactor) : max(abs(u), smoothFactor);
    float z = v < 0.0 ? -max(abs(v), smoothFactor) : max(abs(v), smoothFactor);
    float y = 0.0;  // Flat square initially
    
    // Add wave deformation to the square
    y = amplitude * sin(frequency * 10.0 * (u*u + v*v) - time);
    
    return vec3(x, y, z);
}

vec3 createSpiral(vec2 p) {
    float u = p.x * 10.0 * 3.14159;  // More loops
    float v = p.y;
    
    float r = 0.1 + v * 0.9;  // Radius increases from center to edge
    float height = (v - 0.5) * 2.0;  // Height from -1 to 1
    
    return vec3(
        r * cos(u + twist * height * 5.0),
        height,
        r * sin(u + twist * height * 5.0)
    );
}

void main() {
    // Get base shape based on selected type
    vec3 basePos;
    vec3 morphPos;
    
    // First shape (based on shapeType)
    if (shapeType == 0) {
        basePos = createSphere(position);
    } else if (shapeType == 1) {
        basePos = createTorus(position);
    } else if (shapeType == 2) {
        basePos = createFlower(position);
    } else if (shapeType == 3) {
        basePos = createSquare(position);
    } else if (shapeType == 4) {
        basePos = createSpiral(position);
    } else {
        basePos = createSphere(position);  // Default
    }
    
    // Calculate the next shape for morphing
    int nextShape = (shapeType + 1) % 5;
    if (nextShape == 0) {
        morphPos = createSphere(position);
    } else if (nextShape == 1) {
        morphPos = createTorus(position);
    } else if (nextShape == 2) {
        morphPos = createFlower(position);
    } else if (nextShape == 3) {
        morphPos = createSquare(position);
    } else if (nextShape == 4) {
        morphPos = createSpiral(position);
    } else {
        morphPos = createSphere(position);
    }
    
    // Blend between shapes using the morph factor
    vec3 finalPos = mix(basePos, morphPos, morphFactor);
    
    // Apply frequency-based deformation
    finalPos += amplitude * 0.3 * sin(frequency * 2.0 * finalPos.x + time) * vec3(0.0, 1.0, 0.0);
    
    // Apply a twist based on height and time
    float twistAmount = twist * sin(time * 0.5);
    float cosT = cos(twistAmount * finalPos.y);
    float sinT = sin(twistAmount * finalPos.y);
    finalPos.xz = vec2(
        finalPos.x * cosT - finalPos.z * sinT,
        finalPos.x * sinT + finalPos.z * cosT
    );
    
    // Position in 3D space with animation
    gl_Position = projection * view * model * vec4(finalPos, 1.0);
}
"""

# Fragment shader with dynamic lighting
FRAGMENT_SHADER = """
#version 330
in vec3 fragNormal;
in vec3 fragPosition;
out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 baseColor;
uniform float time;
uniform float frequency;
uniform float energy;

void main() {
    // Color based on position for more interesting visuals
    vec3 color = baseColor;
    color = color * (0.7 + 0.3 * sin(fragPosition.y * 3.0 + time));
    
    // Add frequency-dependent color modulation
    color.r += 0.2 * sin(time * 0.3 + frequency * 0.01);
    color.g += 0.2 * sin(time * 0.5 + frequency * 0.02);
    color.b += 0.2 * sin(time * 0.7 + frequency * 0.03);
    
    // Energy pulsation effect
    float pulse = 0.8 + 0.2 * sin(time * 5.0) + energy * 0.3;
    color *= pulse;
    
    fragColor = vec4(color, 1.0);
}
"""

def lookAt(eye, center, up):
    """Creates a view matrix using the lookAt convention."""
    f = center - eye
    f_norm = np.linalg.norm(f)
    if f_norm > 0:
        f = f / f_norm
    
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm > 0:
        s = s / s_norm
    
    u = np.cross(s, f)
    
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = -eye[0]
    T[1, 3] = -eye[1]
    T[2, 3] = -eye[2]
    
    return M @ T

def perspective(fov, aspect, near, far):
    """Creates a perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj

class SoundShapeVisualizer:
    def __init__(self, width=1024, height=768):
        # Display settings
        self.width = width
        self.height = height
        
        # Sound parameters
        self.sample_rate = 44100
        self.frequency = 440.0
        self.amplitude = 0.5
        self.energy = 0.0
        self.morph_factor = 0.0
        self.twist = 0.0
        
        # Shape parameters
        self.shape_type = 0  # 0=sphere, 1=torus, 2=flower, 3=square, 4=spiral
        self.shape_names = ["Sphere", "Torus", "Flower", "Square", "Spiral"]
        self.auto_morph = True
        self.morph_speed = 0.2
        self.auto_rotate = True
        self.rotation_speed = 0.5
        
        # Visual settings
        self.resolution = 50  # Grid resolution
        self.base_color = np.array([0.2, 0.7, 1.0], dtype=np.float32)
        self.line_mode = False
        
        # Camera settings
        self.camera_distance = 3.0
        self.camera_height = 0.5
        self.camera_angle = 0.0
        self.update_camera_position()
        
        # Timing
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.fps = 0
        self.frame_count = 0
        
        # Setup audio
        self.setup_audio()
        
        # Create mesh geometry
        self.create_mesh()
        
        # Initialize OpenGL
        self.setup_display()
        self.init_opengl()
        
        # Font for overlay
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 18)
        self.show_info = True
    
    def setup_display(self):
        """Set up the Pygame display."""
        self.display = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Sound Shape Visualizer")
    
    def update_camera_position(self):
        """Update camera position based on angle and distance."""
        self.camera_position = np.array([
            self.camera_distance * np.sin(self.camera_angle),
            self.camera_height,
            self.camera_distance * np.cos(self.camera_angle)
        ], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    def create_mesh(self):
        """Create a parametric grid for the shapes."""
        # Create a grid of (u,v) coordinates in [0,1] range
        u = np.linspace(0, 1, self.resolution, dtype=np.float32)
        v = np.linspace(0, 1, self.resolution, dtype=np.float32)
        
        # Create vertices (u,v)
        vertices = []
        for i in range(self.resolution):
            for j in range(self.resolution):
                vertices.append([u[j], v[i]])
        self.vertices = np.array(vertices, dtype=np.float32)
        
        # Create indices for triangle strips
        indices = []
        for i in range(self.resolution - 1):
            for j in range(self.resolution):
                # Add vertices for two triangles
                indices.append(i * self.resolution + j)
                indices.append((i + 1) * self.resolution + j)
            
            # Add a degenerate triangle if not the last strip
            if i < self.resolution - 2:
                indices.append((i + 1) * self.resolution + (self.resolution - 1))
                indices.append((i + 1) * self.resolution)
        
        self.indices = np.array(indices, dtype=np.uint32)
    
    def init_opengl(self):
        """Initialize OpenGL resources."""
        # Compile shader program
        try:
            vertex_shader = compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
            fragment_shader = compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            self.shader = compileProgram(vertex_shader, fragment_shader)
        except Exception as e:
            print(f"Shader compilation error: {e}")
            import sys  # Make sure to import sys at the top of your file
            sys.exit(1)
            
        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        position_loc = glGetAttribLocation(self.shader, "position")
        glEnableVertexAttribArray(position_loc)
        glVertexAttribPointer(position_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        # Setup index buffer
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
        # Enable depth testing and blending
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def setup_audio(self):
        """Initialize audio generation."""
        pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
        self.sound_playing = False
        self.update_sound()
    
    def generate_waveform(self, freq, duration=1.0):
        """Generate a waveform for audio playback."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Create a complex waveform based on the current shape
        if self.shape_type == 0:  # Sphere - pure sine wave
            wave = self.amplitude * np.sin(2 * np.pi * freq * t)
        elif self.shape_type == 1:  # Torus - add harmonics
            wave = self.amplitude * np.sin(2 * np.pi * freq * t)
            wave += 0.5 * self.amplitude * np.sin(2 * np.pi * freq * 2 * t)
        elif self.shape_type == 2:  # Flower - add more harmonics with phase shifts
            wave = self.amplitude * np.sin(2 * np.pi * freq * t)
            wave += 0.3 * self.amplitude * np.sin(2 * np.pi * freq * 3 * t + 0.5)
        elif self.shape_type == 3:  # Square - approximate square wave
            wave = self.amplitude * np.sign(np.sin(2 * np.pi * freq * t))
        elif self.shape_type == 4:  # Spiral - frequency sweep
            wave = self.amplitude * np.sin(2 * np.pi * freq * t * (1 + 0.5 * t/duration))
        else:
            wave = self.amplitude * np.sin(2 * np.pi * freq * t)
        
        # Apply envelope to avoid clicks
        envelope = np.ones_like(t)
        attack = int(0.01 * self.sample_rate)
        decay = int(0.05 * self.sample_rate)
        
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        
        if decay > 0 and decay < len(envelope):
            decay_start = len(envelope) - decay
            envelope[decay_start:] = np.linspace(1, 0, decay)
        
        wave = wave * envelope
        
        # Calculate energy for visualization
        self.energy = np.mean(np.abs(wave)) * 2
        
        return (wave * 32767).astype(np.int16)
    
    def update_sound(self):
        """Update the currently playing sound."""
        try:
            if self.sound_playing:
                pygame.mixer.stop()
            
            wave_data = self.generate_waveform(self.frequency)
            sound = pygame.mixer.Sound(buffer=wave_data)
            sound.play(-1)  # Loop indefinitely
            self.sound_playing = True
        except Exception as e:
            print(f"Sound error: {e}")
            self.sound_playing = False
    
    def handle_input(self, delta_time):
        """Process user input."""
        keys = pygame.key.get_pressed()
        
        # Frequency controls
        freq_change = 0
        if keys[K_UP]:
            freq_change = 10
        elif keys[K_DOWN]:
            freq_change = -10
        
        if freq_change != 0:
            self.frequency = max(20, min(2000, self.frequency + freq_change))
            self.update_sound()
        
        # Amplitude controls
        amp_change = 0
        if keys[K_PAGEUP]:
            amp_change = 0.05
        elif keys[K_PAGEDOWN]:
            amp_change = -0.05
        
        if amp_change != 0:
            self.amplitude = max(0.05, min(1.0, self.amplitude + amp_change))
            self.update_sound()
        
        # Morph controls
        morph_change = 0
        if keys[K_m]:
            morph_change = 0.01
        elif keys[K_n]:
            morph_change = -0.01
        
        if morph_change != 0:
            self.morph_factor = max(0.0, min(1.0, self.morph_factor + morph_change))
        
        # Twist controls
        twist_change = 0
        if keys[K_t]:
            twist_change = 0.1
        elif keys[K_y]:
            twist_change = -0.1
        
        if twist_change != 0:
            self.twist = max(0.0, min(5.0, self.twist + twist_change))
        
        # Camera controls
        if keys[K_LEFT]:
            self.camera_angle += 0.05
            self.update_camera_position()
        elif keys[K_RIGHT]:
            self.camera_angle -= 0.05
            self.update_camera_position()
        
        if keys[K_w]:
            self.camera_height += 0.05
            self.update_camera_position()
        elif keys[K_s]:
            self.camera_height -= 0.05
            self.update_camera_position()
        
        if keys[K_a]:
            self.camera_distance -= 0.05
            self.update_camera_position()
        elif keys[K_d]:
            self.camera_distance += 0.05
            self.update_camera_position()
    
    def update(self, delta_time):
        """Update the visualization state."""
        # Auto-rotate camera if enabled
        if self.auto_rotate:
            self.camera_angle += delta_time * self.rotation_speed
            self.update_camera_position()
        
        # Auto-morph between shapes if enabled
        if self.auto_morph:
            self.morph_factor += delta_time * self.morph_speed
            if self.morph_factor >= 1.0:
                self.morph_factor = 0.0
                self.shape_type = (self.shape_type + 1) % len(self.shape_names)
                self.update_sound()  # Update sound to match new shape
    
    def render(self):
        """Render the visualization."""
        # Clear the frame
        glClearColor(0.05, 0.05, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Use shader program
        glUseProgram(self.shader)
        
        # Set uniforms
        current_time = time.time() - self.start_time
        
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, current_time)
        
        freq_loc = glGetUniformLocation(self.shader, "frequency")
        glUniform1f(freq_loc, self.frequency / 440.0)
        
        amp_loc = glGetUniformLocation(self.shader, "amplitude")
        glUniform1f(amp_loc, self.amplitude)
        
        shape_loc = glGetUniformLocation(self.shader, "shapeType")
        glUniform1i(shape_loc, self.shape_type)
        
        morph_loc = glGetUniformLocation(self.shader, "morphFactor")
        glUniform1f(morph_loc, self.morph_factor)
        
        twist_loc = glGetUniformLocation(self.shader, "twist")
        glUniform1f(twist_loc, self.twist)
        
        energy_loc = glGetUniformLocation(self.shader, "energy")
        glUniform1f(energy_loc, self.energy)
        
        color_loc = glGetUniformLocation(self.shader, "baseColor")
        glUniform3fv(color_loc, 1, self.base_color)
        
        # Set view matrix
        view = lookAt(self.camera_position, self.camera_target, self.camera_up)
        view_loc = glGetUniformLocation(self.shader, "view")
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        
        # Set projection matrix
        projection = perspective(45, self.width / self.height, 0.1, 100)
        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection)
        
        # Set model matrix (identity for now)
        model = np.eye(4, dtype=np.float32)
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        
        # Draw the shape
        glBindVertexArray(self.vao)
        
        if self.line_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glDrawElements(GL_TRIANGLE_STRIP, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        # Reset polygon mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # Render overlay if enabled
        if self.show_info:
            self.render_overlay()
    
    def render_overlay(self):
        """Render text overlay with information."""
        # Switch to 2D rendering
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        
        # Draw text
        self.draw_text(10, 10, f"FPS: {self.fps:.1f}")
        self.draw_text(10, 30, f"Shape: {self.shape_names[self.shape_type]}")
        self.draw_text(10, 50, f"Frequency: {self.frequency:.1f} Hz")
        self.draw_text(10, 70, f"Amplitude: {self.amplitude:.2f}")
        self.draw_text(10, 90, f"Morph: {self.morph_factor:.2f}")
        self.draw_text(10, 110, f"Twist: {self.twist:.1f}")
        
        # Controls help
        y_pos = self.height - 140
        self.draw_text(10, y_pos, "Controls:")
        self.draw_text(10, y_pos + 20, "↑/↓: Frequency, PgUp/PgDn: Amplitude")
        self.draw_text(10, y_pos + 40, "M/N: Manual morphing, T/Y: Twist")
        self.draw_text(10, y_pos + 60, "WASD: Camera movement, ←/→: Rotate camera")
        self.draw_text(10, y_pos + 80, "Space: Toggle sound, L: Toggle line mode")
        self.draw_text(10, y_pos + 100, "A: Toggle auto-morph, R: Toggle auto-rotate")
        
        # Restore 3D rendering
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def draw_text(self, x, y, text, color=(255, 255, 255)):
        """Draw text on the screen."""
        try:
            text_surface = self.font.render(text, True, color)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            width, height = text_surface.get_size()
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glRasterPos2d(x, y)
            glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            glDisable(GL_BLEND)
        except Exception as e:
            print(f"Text rendering error: {e}")
    
    def update_fps(self):
        """Calculate and update FPS."""
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_count += 1
        if current_time - self.start_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.start_time)
            self.frame_count = 0
            self.start_time = current_time
        
        return delta_time
    
    def run(self):
        """Main application loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            delta_time = self.update_fps()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_SPACE:
                        if self.sound_playing:
                            pygame.mixer.stop()
                            self.sound_playing = False
                        else:
                            self.update_sound()
                    elif event.key == K_h:
                        self.show_info = not self.show_info
                    elif event.key == K_l:
                        self.line_mode = not self.line_mode
                    elif event.key == K_a:
                        self.auto_morph = not self.auto_morph
                    elif event.key == K_r:
                        self.auto_rotate = not self.auto_rotate
                    elif event.key == K_TAB:
                        # Manually change shape
                        self.shape_type = (self.shape_type + 1) % len(self.shape_names)
                        self.morph_factor = 0.0
                        self.update_sound()
            
            # Process continuous input
            self.handle_input(delta_time)
            
            # Update state
            self.update(delta_time)
            
            # Render the scene
            self.render()
            
            # Swap buffers
            pygame.display.flip()
            
            # Control frame rate
            clock.tick(60)
        
        # Clean up
        if self.sound_playing:
            pygame.mixer.stop()
        pygame.quit()

if __name__ == "__main__":
    pygame.init()
    visualizer = SoundShapeVisualizer(width=1024, height=768)
    visualizer.run()
