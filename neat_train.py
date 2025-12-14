from ursina import *
# Rename Panda3D imports to avoid conflicts with Ursina
from panda3d.core import Camera as PandaCamera, Texture as PandaTexture
from panda3d.core import PerspectiveLens, NodePath, FrameBufferProperties, WindowProperties, GraphicsPipe
import numpy as np
import random
import pickle
import os
import time
import logging

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TEST_MODE = False           # <--- TRUE = Spectator Mode (1 pair), FALSE = Fast Training Mode (48 pairs)
GEN_SIZE = 48               # Population size
MATCH_DURATION = 25         # Seconds per match
VISION_RES = 10             # Resolution of agent vision (10x10 pixels)
MUTATION_RATE = 0.3         # Chance to change weights
TOPOLOGY_MUTATION_RATE = 0.05 # Chance to grow new brain connections
LOAD_SAVE = True       
LOAD_FILE = "neat_model.pkl"
SAVE_FILE = "neat_model.pkl" 
LOG_FILE = "training_log.txt"

# ==========================================
# --- LOGGING MANAGER ---
# ==========================================
class LogManager:
    def __init__(self, log_file):
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)
        if os.path.exists(log_file): 
            try: os.remove(log_file)
            except PermissionError: pass
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

    def log_gen(self, gen, scores, best_stats):
        if not scores: return
        avg_t = np.mean([s[0] for s in scores])
        avg_r = np.mean([s[1] for s in scores])
        msg = (f"GEN {gen:04d} | T-Avg: {avg_t:.1f} | R-Avg: {avg_r:.1f} | "
               f"Best T Nodes: {best_stats['t_nodes']} | Best R Nodes: {best_stats['r_nodes']}")
        print(msg)
        self.logger.info(msg)

    def log_msg(self, msg):
        print(f">> {msg}")
        self.logger.info(msg)

log_manager = LogManager(LOG_FILE)

# --- URSINA APP SETUP ---
app = Ursina(borderless=False, vsync=True if TEST_MODE else False)
window.title = "NEAT Tag AI: Evolution"
window.size = (1280, 720)
window.color = color.black
window.editor_ui.enabled = False # Disable default editor camera

# Reset main camera
camera.parent = scene
camera.position = (0, 0, 0)
camera.rotation = (0, 0, 0)
camera.ui.enabled = True 
camera.enabled = True 

# ==========================================
# --- NEAT-LITE BRAIN IMPLEMENTATION ---
# ==========================================
def tanh(x): return np.tanh(x)

class Connection:
    def __init__(self, in_node, out_node, weight, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled

class NeatBrain:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_nodes = []
        self.connections = []
        self.biases = {i + input_size: 0.0 for i in range(output_size)}
        for i in range(input_size):
            for o in range(output_size):
                self.connections.append(Connection(i, input_size + o, np.random.uniform(-1, 1)))

    def forward(self, inputs):
        node_values = {i: inputs[i] for i in range(len(inputs))}
        all_nodes = set(range(self.input_size + self.output_size)) | set(self.hidden_nodes)
        
        # Initialize biases for any missing nodes
        for n in all_nodes:
            if n not in node_values: 
                node_values[n] = self.biases.get(n, 0.0)

        current_vals = node_values.copy()
        
        # 2 Iterations to propagate signals through hidden layers
        for _ in range(2): 
            next_vals = {k: (self.biases[k] if k in self.biases else 0.0) for k in all_nodes}
            for i in range(self.input_size): 
                next_vals[i] = current_vals.get(i, 0.0)
                
            for conn in self.connections:
                if conn.enabled:
                    val = current_vals.get(conn.in_node, 0.0)
                    if conn.in_node >= self.input_size: 
                        val = tanh(val)
                    next_vals[conn.out_node] = next_vals.get(conn.out_node, 0.0) + (val * conn.weight)
            current_vals = next_vals

        outputs = []
        for i in range(self.output_size):
            outputs.append(tanh(current_vals.get(self.input_size + i, 0.0)))
        return np.array(outputs)

    def mutate(self):
        # Weight mutation
        for conn in self.connections:
            if random.random() < MUTATION_RATE:
                conn.weight = max(-3.0, min(3.0, conn.weight + np.random.normal(0, 0.2)))
        
        # Bias mutation
        for n in self.biases:
            if random.random() < MUTATION_RATE:
                self.biases[n] += np.random.normal(0, 0.1)
        
        # New Connection
        if random.random() < TOPOLOGY_MUTATION_RATE:
            valid = list(range(self.input_size + self.output_size)) + self.hidden_nodes
            i, o = random.choice(valid), random.choice(valid)
            # Ensure not connecting output to input directly backwards or duplicate
            if o >= self.input_size and i != o:
                if not any(c.in_node == i and c.out_node == o for c in self.connections):
                    self.connections.append(Connection(i, o, np.random.uniform(-1,1)))
        
        # New Node (Split connection)
        if random.random() < TOPOLOGY_MUTATION_RATE and self.connections:
            conn = random.choice(self.connections)
            if conn.enabled:
                conn.enabled = False 
                new_id = self.input_size + self.output_size + len(self.hidden_nodes) + 100
                while new_id in self.hidden_nodes: new_id += 1
                
                self.hidden_nodes.append(new_id)
                self.biases[new_id] = 0.0
                
                self.connections.append(Connection(conn.in_node, new_id, 1.0))
                self.connections.append(Connection(new_id, conn.out_node, conn.weight))

    def clone(self):
        nb = NeatBrain(self.input_size, self.output_size)
        nb.hidden_nodes = list(self.hidden_nodes)
        nb.biases = self.biases.copy()
        nb.connections = [Connection(c.in_node, c.out_node, c.weight, c.enabled) for c in self.connections]
        return nb

# ==========================================
# --- AGENT CLASS ---
# ==========================================
class Agent(Entity):
    def __init__(self, role, pair_id, origin_x, manager, brain=None, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.pair_id = pair_id 
        self.origin_x = origin_x 
        self.manager = manager
        self.speed = 18 if role == "tagger" else 16 
        self.turn_speed = 220
        self.jump_force = 16
        self.model = 'cube'
        self.color = color.red if role == "tagger" else color.azure
        self.collider = 'box'
        self.velocity_y = 0
        self.grounded = False
        self.jump_cooldown = 0 
        self.prev_pos = self.position
        self.current_vel = Vec3(0,0,0) 
        self.last_target_pos = None    
        self.estimated_target_vel = Vec3(0,0,0)
        self.fitness_score = 0
        self.stuck_timer = 0
        self.last_dist = 20.0 
        self.vision_res = VISION_RES
        self.brain = brain if brain else NeatBrain((self.vision_res**2) + 16, 4) 
        self.tex_buffer = PandaTexture() # Use the explicitly imported Panda Texture
        
        # Only setup heavy vision buffer for pair 0 (spectator) or if in Test Mode
        if not TEST_MODE or (TEST_MODE and self.pair_id == 0): 
            self.setup_vision()
            
        self.frame_skip = random.randint(0, 3) 

    def setup_vision(self):
        win = WindowProperties(size=(self.vision_res, self.vision_res))
        fb = FrameBufferProperties()
        fb.set_rgb_color(True); fb.set_depth_bits(1)
        
        # --- FIX: Use 'app' instead of 'base' ---
        self.buffer = app.graphicsEngine.make_output(app.pipe, f"buff_{self.role}_{self.pair_id}", -2, fb, win, GraphicsPipe.BF_refuse_window, app.win.get_gsg(), app.win)
        
        if self.buffer:
            # --- FIX: Use 'app.win' instead of 'base.win' ---
            self.buffer.add_render_texture(self.tex_buffer, app.win.RTM_copy_ram)
            self.cam_node = PandaCamera(f'{self.role}_cam_{self.pair_id}')
            self.cam_node.set_lens(PerspectiveLens())
            self.cam_node.get_lens().set_fov(100) 
            self.cam_node.set_scene(scene) 
            self.cam_np = NodePath(self.cam_node)
            self.cam_np.reparent_to(self)
            self.cam_np.set_pos(0, 0.5, 0.5) 
            self.display_region = self.buffer.make_display_region()
            self.display_region.set_camera(self.cam_np)

    def get_vision_data(self):
        if not hasattr(self, 'buffer') or not self.tex_buffer.has_ram_image(): 
            return np.zeros(self.vision_res**2)
            
        img = self.tex_buffer.get_ram_image_as("RGB")
        if not img: return np.zeros(self.vision_res**2)
        
        # Extract Green channel only for simple grayscale vision
        return np.frombuffer(img, dtype=np.uint8).astype(np.float32).reshape((self.vision_res, self.vision_res, 3))[:, :, 1].flatten() / 255.0

    def get_whiskers(self):
        sensors, min_d = [], 1.0
        # Use simple int for loop to ensure compatibility
        angles = [-45, -20, 0, 20, 45]
        for a in angles:
            r_rad = np.radians(self.rotation_y + a)
            dir_vec = Vec3(np.sin(r_rad), 0, np.cos(r_rad)).normalized()
            d = raycast(self.position + Vec3(0,0.5,0), dir_vec, distance=8, ignore=(self,)).distance
            val = d / 8.0 if d else 1.0
            sensors.append(val)
            if val < min_d: min_d = val
        return np.array(sensors), min_d

    def act(self, target, dt):
        # Calculate Velocity
        if dt > 0:
            self.current_vel = (self.position - self.prev_pos) / dt
        else:
            self.current_vel = Vec3(0,0,0)
            
        self.prev_pos = self.position
        
        # Estimate Target Velocity
        if self.last_target_pos:
            t_vel = (target.position - self.last_target_pos) / dt
            self.estimated_target_vel = lerp(self.estimated_target_vel, t_vel, dt * 5)
        self.last_target_pos = target.position
        
        # Gravity
        self.velocity_y -= 40 * dt
        ray = raycast(self.position + Vec3(0,0.1,0), Vec3(0, -1, 0), distance=0.7+abs(self.velocity_y*dt), ignore=(self,))
        
        if ray.hit: 
            self.velocity_y = 0
            self.grounded = True
            self.y = ray.world_point.y + 0.5
        else: 
            self.grounded = False
            self.y += self.velocity_y * dt
            
        if self.jump_cooldown > 0: self.jump_cooldown -= dt

        self.frame_skip += 1
        # Brain Update (Every 3rd frame to save performance)
        if self.frame_skip % 3 == 0: 
            vis = self.get_vision_data()
            whisk, wall = self.get_whiskers()
            
            to_t = target.position - self.position
            # Angle to target (-pi to pi)
            ang = (np.arctan2(to_t.x, to_t.z) - np.radians(self.rotation_y) + np.pi) % (2 * np.pi) - np.pi
            
            comp = np.array([ang / np.pi, min(to_t.length(), 40.0) / 40.0])
            rel_v = (self.estimated_target_vel - self.current_vel) / 20.0 
            
            inps = np.concatenate((
                vis, whisk, comp, 
                [rel_v.x, rel_v.y, rel_v.z], 
                [1.0 if self.grounded else 0.0, wall], 
                [1.0], # Bias node
                np.zeros(3) # Padding
            ))[:(self.vision_res**2 + 16)]
            
            self.decision = self.brain.forward(inps)

        # Movement Execution
        if hasattr(self, 'decision'):
            strafe, fwd, jump, turn = self.decision
            self.rotation_y += turn * self.turn_speed * dt
            
            # Forward/Strafe Vector
            mv = (self.forward * fwd + self.right * strafe).normalized()
            
            # Simple collision check
            hit_info = raycast(self.position + Vec3(0,0.5,0), mv, distance=(self.speed * dt) + 0.5, ignore=(self, target))
            if not hit_info.hit:
                self.position += mv * self.speed * dt
            else:
                self.stuck_timer += dt
                
            if jump > 0.0 and self.grounded and self.jump_cooldown <= 0:
                self.velocity_y = self.jump_force
                self.grounded = False
                self.jump_cooldown = 0.5
        
        # Out of bounds reset
        if self.y < -10: 
            self.y = 15
            self.x = self.origin_x
            self.velocity_y = 0
            self.fitness_score -= 200

    def on_destroy(self):
        if hasattr(self, 'buffer') and self.buffer: 
            # --- FIX: Use 'app' instead of 'base' ---
            app.graphicsEngine.remove_window(self.buffer)
        if hasattr(self, 'cam_np'): 
            self.cam_np.remove_node()

# ==========================================
# --- MANAGER & SIMULATION ---
# ==========================================
class TrainingManager(Entity):
    def __init__(self):
        super().__init__()
        self.generation = 1
        self.population = [] 
        self.agents = [] 
        self.scores = []
        self.active = False
        self.time_scale = 1.0
        self.cam_dist = 24.0
        self.cam_height = 14.0
        
        # UI
        self.ui_gen = Text(text="Gen: 1", position=(-0.85, 0.45), scale=1.5)
        self.ui_timer = Text(text="0.0s", position=(0, 0.45), scale=1.5, origin=(0,0))
        self.ui_info = Text(text="[Scroll] Zoom | [L-Click] Spectate Tagger | [0-9] Speed", position=(0, 0.40), scale=1, origin=(0,0))
        
        if LOAD_SAVE: self.load_brains()
        self.start_generation()

    def input(self, key):
        speed_map = {'0': 0.0, '1': 0.25, '2': 0.5, '3': 1.0, '4': 2.0, '5': 5.0, '9': 15.0}
        if key in speed_map: self.time_scale = speed_map[key]
        if key == 'scroll up': self.cam_dist = max(5, self.cam_dist - 3)
        if key == 'scroll down': self.cam_dist = min(50, self.cam_dist + 3)
        self.cam_height = self.cam_dist * 0.6 

    def start_generation(self):
        log_manager.log_msg(f"STARTING GEN {self.generation}")
        self.cleanup()
        local_gen_size = 1 if TEST_MODE else GEN_SIZE
        self.scores = [(0,0)] * local_gen_size
        
        # Init population if empty
        if not self.population:
            for _ in range(local_gen_size): 
                self.population.append((
                    NeatBrain((VISION_RES**2)+16, 4), 
                    NeatBrain((VISION_RES**2)+16, 4)
                ))
        
        # Fill population if needed
        while len(self.population) < local_gen_size:
             t, r = self.population[random.randint(0, len(self.population)-1)]
             t, r = t.clone(), r.clone()
             t.mutate(); r.mutate()
             self.population.append((t, r))
             
        self.population = self.population[:local_gen_size]
        self.spawn_walls(self.generation)
        
        for i in range(local_gen_size):
            t_brain, r_brain = self.population[i]
            off = i * 80 
            tx, tz = off + random.uniform(-10, 10), random.uniform(-10, 10)
            rx, rz = off + random.uniform(-10, 10), random.uniform(-10, 10)
            
            while distance((tx,0,tz), (rx,0,rz)) < 10: 
                rx, rz = off + random.uniform(-15, 15), random.uniform(-15, 15)
                
            self.agents.append((
                Agent("tagger", i, off, self, t_brain, position=(tx, 2, tz)), 
                Agent("runner", i, off, self, r_brain, position=(rx, 2, rz))
            ))

        self.time_elapsed = 0
        self.active = True

    def spawn_walls(self, seed):
        if hasattr(self, 'walls'): 
            for w in self.walls: destroy(w)
        self.walls = []
        random.seed(seed)
        count = 1 if TEST_MODE else GEN_SIZE
        
        for i in range(count):
            off = i * 80
            # Floor
            self.walls.append(Entity(model='cube', position=(off, -1, 0), scale=(60, 2, 60), texture='grass', collider='box'))
            # Borders
            self.walls.extend([
                Entity(model='cube', position=(off, 2, 30), scale=(60, 6, 2), color=color.gray, collider='box'),
                Entity(model='cube', position=(off, 2, -30), scale=(60, 6, 2), color=color.gray, collider='box'),
                Entity(model='cube', position=(off+30, 2, 0), scale=(2, 6, 60), color=color.gray, collider='box'),
                Entity(model='cube', position=(off-30, 2, 0), scale=(2, 6, 60), color=color.gray, collider='box')
            ])
            # Random obstacles
            for _ in range(12):
                h = random.choice([1.5, 2.5, 3.5])
                self.walls.append(Entity(
                    model='cube', 
                    position=(random.randint(-25, 25)+off, h/2, random.randint(-25, 25)), 
                    scale=(random.randint(3,6), h, random.randint(3,6)), 
                    texture='brick', 
                    collider='box'
                ))

    def update(self):
        if not self.active: return
        
        dt = 0.05
        steps = 1 if TEST_MODE else int(self.time_scale) 
        if TEST_MODE: dt = time.dt
        
        for _ in range(steps): 
            self.step_sim(dt)
            
        self.ui_timer.text = f"{self.time_elapsed:.1f}/{MATCH_DURATION}"
        
        # --- CAMERA FIXED LOGIC ---
        if self.agents:
            target = None
            # Try to find user selected target
            for t, r in self.agents:
                if t.enabled: 
                    target = t if mouse.left else r
                    break
            
            # Fallback target if current one disabled
            if not target:
                active_list = [a for pair in self.agents for a in pair if a.enabled]
                if active_list: target = active_list[0]

            if target:
                offset = Vec3(0, self.cam_height, -self.cam_dist)
                # Adjust camera speed by time_scale so it doesn't lag
                cam_speed = time.dt * 5 * (self.time_scale if self.time_scale >= 1 else 1)
                camera.position = lerp(camera.position, target.position + offset, min(cam_speed, 1.0))
                camera.look_at(target.position + Vec3(0, 1, 0))

    def step_sim(self, dt):
        self.time_elapsed += dt
        active_cnt = 0
        
        for i, (t, r) in enumerate(self.agents):
            if not t.enabled: continue
            active_cnt += 1
            
            t.act(r, dt)
            r.act(t, dt)
            
            dist = distance(t.position, r.position)
            
            # SCORING RULES
            r.fitness_score += 1.0 * dt # Survival Bonus
            
            # Juking Bonus (Running towards tagger but close)
            if dist < 8.0 and r.current_vel.length() > 5.0:
                 dot_prod = r.current_vel.normalized().dot((t.position - r.position).normalized())
                 if dot_prod < -0.5: r.fitness_score += 5.0 * dt
            
            if dist > 15.0: r.fitness_score += 2.0 * dt
            if r.stuck_timer > 1.0: r.fitness_score -= 10.0 * dt
            
            # Tagger Closing Distance Bonus
            if (t.last_dist - dist) > 0: 
                t.fitness_score += (t.last_dist - dist) * 10.0 
            t.last_dist = dist
            
            # Facing Target Bonus
            if t.forward.dot((r.position - t.position).normalized()) > 0.8: 
                t.fitness_score += 2.0 * dt

            # Tag Event
            if dist < 1.5:
                t.fitness_score += 300
                r.fitness_score -= 50
                self.finish_pair(i, "tagger")

        # Time Limit Reached
        if self.time_elapsed > MATCH_DURATION:
            for i, (t, r) in enumerate(self.agents):
                if t.enabled: 
                    r.fitness_score += 300
                    self.finish_pair(i, "runner")
                    
        # End Gen Check
        if active_cnt == 0:
            if TEST_MODE: self.start_generation()
            else: self.evolve()

    def finish_pair(self, idx, winner):
        t, r = self.agents[idx]
        self.scores[idx] = (t.fitness_score, r.fitness_score)
        t.enabled = False; r.enabled = False
        t.visible = False; r.visible = False

    def evolve(self):
        # Sort by fitness
        s_tag = sorted(zip(self.population, self.scores), key=lambda x: x[1][0], reverse=True)
        s_run = sorted(zip(self.population, self.scores), key=lambda x: x[1][1], reverse=True)
        
        stats = {'t_nodes': len(s_tag[0][0][0].hidden_nodes), 'r_nodes': len(s_run[0][0][1].hidden_nodes)}
        log_manager.log_gen(self.generation, self.scores, stats)
        
        # Save best brains
        self.save_brains(s_tag[0][0][0], s_run[0][0][1])

        # Selection & Reproduction
        new_pop = []
        # Keep top 4 exact copies (Elitism)
        for i in range(4): 
            new_pop.append((s_tag[i][0][0].clone(), s_run[i][0][1].clone()))
            
        while len(new_pop) < GEN_SIZE:
            # Tournament selectionish
            p1 = random.choice(s_tag[:10])[0][0]
            p2 = random.choice(s_run[:10])[0][1]
            
            child_t, child_r = p1.clone(), p2.clone()
            child_t.mutate()
            child_r.mutate()
            new_pop.append((child_t, child_r))
            
        self.population = new_pop
        self.generation += 1
        self.ui_gen.text = f"Gen: {self.generation}"
        self.start_generation()

    def save_brains(self, t, r):
        with open(SAVE_FILE, 'wb') as f: 
            pickle.dump({"gen": self.generation, "t": t, "r": r}, f)
        
    def load_brains(self):
        if os.path.exists(LOAD_FILE):
            try:
                with open(LOAD_FILE, 'rb') as f:
                    data = pickle.load(f)
                self.generation = data['gen']
                self.population = []
                # Re-seed population with saved brains
                for _ in range(GEN_SIZE):
                    t, r = data['t'].clone(), data['r'].clone()
                    t.mutate(); r.mutate()
                    self.population.append((t,r))
                log_manager.log_msg("Loaded Saved Brains")
            except Exception as e:
                print(f"Failed to load save: {e}")
    
    def cleanup(self):
        for t, r in self.agents: 
            destroy(t)
            destroy(r)
        self.agents.clear()

trainer = TrainingManager()
app.run()