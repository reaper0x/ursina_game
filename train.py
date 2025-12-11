from ursina import *
from panda3d.core import Camera as PandaCamera
from panda3d.core import PerspectiveLens, NodePath, FrameBufferProperties, WindowProperties, GraphicsPipe, Texture
import numpy as np
import random
import pickle
import os
import time

# --- CONFIGURATION ---
GEN_SIZE = 4           
MATCH_DURATION = 30    
VISION_RES = 16        
MUTATION_RATE = 0.15          
MUTATION_STRENGTH = 0.15      
SAVE_FILE = "best_brains.pkl"

app = Ursina(borderless=False, vsync=False)

from ursina import application
base = application.base

window.title = "Tag AI: CCTV Grid View"
window.size = (1280, 720)
window.color = color.black

camera.ui.enabled = True 
camera.enabled = False   

# --- NEURAL NETWORK ---
class SimpleBrain:
    def __init__(self, input_size, hidden_size, output_size):
        scale = 1.0 / np.sqrt(input_size)
        self.w1 = np.random.uniform(-scale, scale, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        
        scale2 = 1.0 / np.sqrt(hidden_size)
        self.w2 = np.random.uniform(-scale2, scale2, (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

        # BIAS: Balanced start
        self.b2[1] = 0.2   # Slight forward urge
        self.b2[3] = 0.0   
        self.b2[2] = 0.1   

    def forward(self, inputs):
        self.z1 = np.tanh(np.dot(inputs, self.w1) + self.b1)
        output = np.tanh(np.dot(self.z1, self.w2) + self.b2)
        return output 

    def mutate(self):
        if random.random() < MUTATION_RATE:
            self.w1 += np.random.randn(*self.w1.shape) * MUTATION_STRENGTH
            self.b1 += np.random.randn(*self.b1.shape) * MUTATION_STRENGTH
            self.w2 += np.random.randn(*self.w2.shape) * MUTATION_STRENGTH
            self.b2 += np.random.randn(*self.b2.shape) * MUTATION_STRENGTH

    def clone(self):
        clone = SimpleBrain(self.w1.shape[0], self.w1.shape[1], self.w2.shape[1])
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        return clone

# --- AGENT CLASS ---
class Agent(Entity):
    def __init__(self, role, pair_id, origin_x, manager, brain=None, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.pair_id = pair_id 
        self.origin_x = origin_x 
        self.manager = manager
        
        # MOVEMENT SETTINGS
        self.speed = 15  # Slightly faster
        self.turn_speed = 200
        
        self.model = 'cube'
        self.color = color.red if role == "tagger" else color.azure
        self.scale = (1, 1, 1) 
        self.collider = 'box' 
        self.velocity_y = 0
        self.grounded = False
        
        # TRACKING FOR FITNESS
        self.min_dist_to_target = 100.0 # Track the closest we ever got
        self.facing_score = 0.0 # Track how much time we spent looking at target

        # INPUTS: Vision(256) + RelPos(3) + Sensors(3) + Heading(1) = 263
        # I reduced vision to 16x16 (256 pixels) to make learning faster
        self.vision_res = 16 
        input_nodes = (self.vision_res * self.vision_res) + 7
        self.brain = brain if brain else SimpleBrain(input_nodes, 32, 4) # Increased hidden layer to 32

        self.tex_buffer = Texture()
        self.setup_vision()
        self.frame_skip = random.randint(0, 3) 

    def setup_vision(self):
        # Optimized lower resolution vision for faster processing
        win_props = WindowProperties()
        win_props.set_size(self.vision_res, self.vision_res)
        fb_props = FrameBufferProperties()
        fb_props.set_rgb_color(True)
        fb_props.set_depth_bits(1)
        
        self.buffer = base.graphicsEngine.make_output(
            base.pipe, f"buff_{self.role}_{self.pair_id}", -2,
            fb_props, win_props,
            GraphicsPipe.BF_refuse_window,
            base.win.get_gsg(), base.win
        )
        
        if self.buffer:
            self.buffer.add_render_texture(self.tex_buffer, base.win.RTM_copy_ram)
            self.cam_node = PandaCamera(f'{self.role}_cam_{self.pair_id}')
            self.cam_node.set_lens(PerspectiveLens())
            self.cam_node.get_lens().set_fov(110) # Wider FOV helps them see targets on sides
            self.cam_node.set_scene(scene) 
            
            self.cam_np = NodePath(self.cam_node)
            self.cam_np.reparent_to(self)
            self.cam_np.set_pos(0, 0.5, 0.5) 
            
            self.display_region = self.buffer.make_display_region()
            self.display_region.set_camera(self.cam_np)

    def get_vision_data(self):
        if not self.tex_buffer or not self.tex_buffer.has_ram_image(): 
            return np.zeros(self.vision_res * self.vision_res)
        
        img = self.tex_buffer.get_ram_image_as("RGB")
        if not img: return np.zeros(self.vision_res * self.vision_res)
        
        # Extract just the RED channel, normalize to 0-1
        arr = np.frombuffer(img, dtype=np.uint8).astype(np.float32) / 255.0
        arr = arr.reshape((self.vision_res, self.vision_res, 3))
        return arr[:, :, 0].flatten() 

    def get_sensor_data(self):
        sensors = []
        # Raycasts: Left, Center, Right to detect walls
        angles = [-35, 0, 35] 
        for angle in angles:
            theta = np.radians(angle + self.rotation_y)
            dir_vec = Vec3(np.sin(theta), 0, np.cos(theta)).normalized()
            ray = raycast(self.position + Vec3(0,0.5,0), dir_vec, distance=15, ignore=(self,))
            sensors.append(ray.distance / 15.0 if ray.hit else 1.0)
        return np.array(sensors)

    def act(self, target):
        # --- PHYSICS ---
        # Gravity
        ray = raycast(self.position + Vec3(0, -0.4, 0), Vec3(0, -1, 0), distance=0.6, ignore=(self,))
        if ray.hit:
            self.velocity_y = 0
            self.grounded = True
            self.y = ray.world_point.y + 0.5
        else:
            self.velocity_y -= 40 * time.dt
            self.grounded = False
        self.y += self.velocity_y * time.dt

        # --- BRAIN INPUTS ---
        self.frame_skip += 1
        if self.frame_skip % 3 == 0: 
            # 1. Vision & Sensors
            vision = self.get_vision_data()
            sensors = self.get_sensor_data() 
            
            # 2. Vector to Target
            vec = target.position - self.position
            dist = vec.length()
            
            # Update Fitness Metrics
            if dist < self.min_dist_to_target: self.min_dist_to_target = dist
            
            # 3. Calculate Heading Error (The "Secret Sauce" for natural turning)
            # This converts global coordinates into a relative angle (-1 to 1)
            # -1 = Target is fully Left, +1 = Target is fully Right, 0 = Target is Forward
            forward = self.forward
            vec_norm = vec.normalized()
            # Dot product for forward amount, Cross product y-component for angle
            heading_error = np.arctan2(vec.x, vec.z) - np.radians(self.rotation_y)
            # Normalize angle to -PI to PI
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            heading_input = heading_error / np.pi # Normalize to -1 to 1

            # Local coordinates (still useful)
            theta = np.radians(-self.rotation_y)
            c, s = np.cos(theta), np.sin(theta)
            local_x = vec.x * c - vec.z * s
            local_z = vec.x * s + vec.z * c
            
            # Accumulate facing score (fitness bonus for looking at target)
            if abs(heading_input) < 0.2: self.facing_score += 1
            
            # Assemble Inputs
            # Vision + Local Pos + Sensors + HEADING
            rel_pos = np.array([local_x/30.0, vec.y/10.0, local_z/30.0, heading_input])
            inputs = np.concatenate((vision, rel_pos, sensors))
            
            self.decision = self.brain.forward(inputs)

        # --- MOVEMENT EXECUTION ---
        if hasattr(self, 'decision'):
            # Outputs: 0:Strafe, 1:Fwd, 2:Jump, 3:Turn
            strafe = self.decision[0]
            fwd = self.decision[1]
            turn = self.decision[3]
            
            # Optimization: Allow natural turning regardless of forward speed
            self.rotation_y += turn * self.turn_speed * time.dt
            
            # Optimization: Smoother movement mixing
            move_dir = (self.forward * fwd + self.right * strafe).normalized()
            
            # Check collisions
            check_dist = 1.0
            if not raycast(self.position + Vec3(0,0.5,0), move_dir, distance=check_dist, ignore=(self,)).hit:
                 self.position += move_dir * self.speed * time.dt
            
            # Jump
            if self.decision[2] > 0.6 and self.grounded:
                self.velocity_y = 15

        # Respawn if fallen
        if self.y < -10: 
            self.y = 15
            self.x = self.origin_x
            self.z = 0
            self.velocity_y = 0
            
    def on_destroy(self):
        if hasattr(self, 'buffer') and self.buffer:
            base.graphicsEngine.remove_window(self.buffer)
        if hasattr(self, 'cam_np'):
            self.cam_np.remove_node()

# --- GRID CAMERAS ---
class GridCameraSystem:
    def __init__(self, agents):
        self.cameras = []
        self.regions = []
        
        count = len(agents)
        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        
        for i in range(len(agents)):
            t, r = agents[i]
            col = i % cols
            row = i // cols
            
            left = col / cols
            right = (col + 1) / cols
            bottom = 1.0 - ((row + 1) / rows)
            top = 1.0 - (row / rows)
            
            dr = base.win.makeDisplayRegion(left, right, bottom, top)
            dr.set_clear_color_active(True)
            dr.set_clear_color((0.1, 0.1, 0.1, 1)) 
            
            cam = PandaCamera(f'grid_cam_{i}')
            lens = PerspectiveLens()
            lens.set_fov(90)
            cam.set_lens(lens)
            cam.set_scene(scene)
            
            cam_np = NodePath(cam)
            cam_np.reparent_to(t) 
            cam_np.set_pos(0, 6, -9) 
            cam_np.look_at(t)
            
            dr.set_camera(cam_np)
            self.cameras.append(cam_np)
            self.regions.append(dr)

    def cleanup(self):
        for dr in self.regions:
            base.win.remove_display_region(dr)
        for cam in self.cameras:
            cam.remove_node()

# --- MANAGER ---
class TrainingManager(Entity):
    def __init__(self):
        super().__init__()
        self.generation = 1
        self.population = [] 
        self.agents = [] 
        self.scores = []
        self.wall_data = [] 
        self.time_elapsed = 0
        self.active = False
        self.stop_requested = False
        self.grid_sys = None

        self.ui_gen = Text(text="Gen: 1", position=(-0.85, 0.45), scale=2, color=color.white)
        self.ui_timer = Text(text="0.0s", position=(0, 0.45), scale=2, origin=(0,0), color=color.white)
        
        self.load_brains()
        self.start_generation()

    def start_generation(self):
        self.ui_gen.text = f"Gen: {self.generation}"
        print(f"--- STARTING GEN {self.generation} ---")
        self.cleanup()
        self.scores = [(0,0)] * GEN_SIZE
        
        # --- FIX: Update Input Size Calculation ---
        # Vision(16x16=256) + RelPos(3) + Sensors(3) + Heading(1) = 263
        input_size = (VISION_RES * VISION_RES) + 7 
        
        if not self.population:
            for _ in range(GEN_SIZE):
                # increased hidden size to 32 as recommended
                self.population.append((
                    SimpleBrain(input_size, 32, 4), 
                    SimpleBrain(input_size, 32, 4)
                ))
        
        self.spawn_walls(seed=self.generation)
        
        for i in range(GEN_SIZE):
            t_brain, r_brain = self.population[i]
            offset_x = i * 100 
            
            tx, tz = self.get_safe_spawn(offset_x)
            rx, rz = self.get_safe_spawn(offset_x)
            
            tagger = Agent("tagger", i, offset_x, self, brain=t_brain, position=(tx, 2, tz))
            runner = Agent("runner", i, offset_x, self, brain=r_brain, position=(rx, 2, rz))
            
            self.agents.append((tagger, runner))

        self.grid_sys = GridCameraSystem(self.agents)
        self.time_elapsed = 0
        self.active = True

    def spawn_walls(self, seed):
        self.walls = []
        self.wall_data = [] 
        random.seed(seed)
        
        def add_wall(pos, scale, mat='brick'):
            pad = 2.5 
            min_x = pos[0] - scale[0]/2 - pad
            max_x = pos[0] + scale[0]/2 + pad
            min_z = pos[2] - scale[2]/2 - pad
            max_z = pos[2] + scale[2]/2 + pad

            if mat != 'grass' and isinstance(mat, str):
                for (wx1, wx2, wz1, wz2) in self.wall_data:
                    if (min_x < wx2 and max_x > wx1 and min_z < wz2 and max_z > wz1):
                        return 
                self.wall_data.append((min_x, max_x, min_z, max_z))

            if isinstance(mat, str):
                w = Entity(model='cube', position=pos, scale=scale, texture=mat, collider='box')
                w.texture_scale = ((scale[0]+scale[2])/2, scale[1])
            else:
                w = Entity(model='cube', position=pos, scale=scale, color=mat, collider='box')
            self.walls.append(w)

        for i in range(GEN_SIZE):
            offset_x = i * 100
            
            add_wall((offset_x, 0, 0), (50, 1, 50), 'grass')

            add_wall((offset_x, 3, 25), (50, 6, 1), color.gray)
            add_wall((offset_x, 3, -25), (50, 6, 1), color.gray)
            add_wall((offset_x + 25, 3, 0), (1, 6, 50), color.gray)
            add_wall((offset_x - 25, 3, 0), (1, 6, 50), color.gray)
            
            self.wall_data.append((offset_x - 26, offset_x + 26, 24, 26))   
            self.wall_data.append((offset_x - 26, offset_x + 26, -26, -24)) 
            self.wall_data.append((offset_x + 24, offset_x + 26, -26, 26))  
            self.wall_data.append((offset_x - 26, offset_x - 24, -26, 26))  

            for _ in range(30): 
                wx = random.randint(-18, 18)
                wz = random.randint(-18, 18)
                
                if random.random() < 0.4:
                    h = 2.0
                    wsx = random.randint(2, 4)
                    wsz = random.randint(2, 4)
                    y_pos = h / 2
                    col = 'brick'
                else:
                    h = random.randint(5, 7)
                    wsx = random.randint(3, 6)
                    wsz = random.randint(3, 6)
                    y_pos = h / 2
                    col = 'brick'
                
                add_wall((wx + offset_x, y_pos, wz), (wsx, h, wsz), col)

    def get_safe_spawn(self, offset_x):
        for _ in range(100):
            x = offset_x + random.uniform(-18, 18)
            z = random.uniform(-18, 18)
            
            valid = True
            for (min_x, max_x, min_z, max_z) in self.wall_data:
                if min_x < x < max_x and min_z < z < max_z:
                    valid = False
                    break
            
            if valid:
                return x, z
        
        return offset_x, 0

    def update(self):
        if not self.active: return
        
        dt = time.dt * (3 if held_keys['p'] else 1)
        self.time_elapsed += dt
        self.ui_timer.text = f"{self.time_elapsed:.1f}s / {MATCH_DURATION}s"

        active_count = 0
        for i, (tagger, runner) in enumerate(self.agents):
            if not tagger.enabled: continue
            active_count += 1

            tagger.act(runner)
            runner.act(tagger)

            # --- FIX: Precise Collision (Must be very close, basically touching) ---
            if distance(tagger.position, runner.position) < 1.2:
                self.finish_pair(i, winner="tagger")
            
        if self.time_elapsed > MATCH_DURATION:
            for i, (tagger, runner) in enumerate(self.agents):
                if tagger.enabled:
                    self.finish_pair(i, winner="runner")
        
        if active_count == 0:
            self.evolve()

    def finish_pair(self, index, winner):
        t, r = self.agents[index]
        
        # --- TAGGER FITNESS CALCULATION ---
        # Base: 0
        # +500 if they caught the runner
        # + (StartDist - MinDist) * 10: Reward for getting closer (Dense Reward)
        # + FacingScore: Small reward for keeping the target in view
        
        start_dist = distance(Vec3(t.origin_x, 2, -12), Vec3(t.origin_x, 2, 12)) # Approx start dist
        dist_improvement = max(0, start_dist - t.min_dist_to_target)
        
        t_score = (dist_improvement * 5) + (t.facing_score * 0.5)
        
        if winner == "tagger":
            time_bonus = (MATCH_DURATION - self.time_elapsed) * 10
            t_score += 500 + time_bonus
            r_score = self.time_elapsed * 5 # Runner gets points for how long they lasted
        else:
            # Runner won (time out)
            r_score = 300 + distance(t.position, r.position) * 2
            
        self.scores[index] = (t_score, r_score)
        
        t.enabled = False
        r.enabled = False
        t.visible = False
        r.visible = False
        r.visible = False

    def evolve(self):
        print("--- EVOLVING ---")
        sorted_taggers = sorted(zip(self.population, self.scores), key=lambda x: x[1][0], reverse=True)
        sorted_runners = sorted(zip(self.population, self.scores), key=lambda x: x[1][1], reverse=True)
        
        best_t = sorted_taggers[0][0][0]
        best_r = sorted_runners[0][0][1]
        
        self.save_brains(best_t, best_r)
        
        if self.stop_requested:
            application.quit()
            return

        new_pop = []
        new_pop.append((best_t.clone(), best_r.clone())) 
        
        for _ in range(GEN_SIZE - 1):
            t = best_t.clone()
            t.mutate()
            r = best_r.clone()
            r.mutate()
            new_pop.append((t, r))
            
        self.population = new_pop
        self.generation += 1
        self.start_generation()

    def cleanup(self):
        if self.grid_sys: self.grid_sys.cleanup()
        for t, r in self.agents:
            destroy(t)
            destroy(r)
        self.agents.clear()
        if hasattr(self, 'walls'):
            for w in self.walls: destroy(w)

    def save_brains(self, t, r):
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump({"gen":self.generation, "t_brain":t, "r_brain":r}, f)
        print("Saved Best.")

    def load_brains(self):
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE, 'rb') as f:
                    data = pickle.load(f)
                    
                    # --- SAFETY CHECK ---
                    # Check if the loaded brain matches our current input size (263)
                    # The weight matrix shape is (Inputs, Hidden)
                    saved_input_size = data['t_brain'].w1.shape[0]
                    current_input_size = (VISION_RES * VISION_RES) + 7
                    
                    if saved_input_size != current_input_size:
                        print(f"Save file mismatch (Saved: {saved_input_size}, Current: {current_input_size}). Starting fresh.")
                        return # Ignore the file, start fresh

                    self.generation = data['gen']
                    t, r = data['t_brain'], data['r_brain']
                    self.population = []
                    
                    # Re-populate
                    for _ in range(GEN_SIZE):
                        tc, rc = t.clone(), r.clone()
                        tc.mutate(); rc.mutate()
                        self.population.append((tc, rc))
                    print(f"Loaded Gen {self.generation}")
            except Exception as e: 
                print(f"Load failed: {e}")

def req_stop():
    trainer.stop_requested = True
    b_stop.text = "Stopping..."
    b_stop.color = color.red

b_stop = Button(text="Save & Stop", color=color.orange, scale=(0.2, 0.05), position=(-0.7, -0.4), on_click=req_stop)

trainer = TrainingManager()
app.run()