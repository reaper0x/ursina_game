from ursina import *
from panda3d.core import Camera as PandaCamera
from panda3d.core import PerspectiveLens, NodePath, FrameBufferProperties, WindowProperties, GraphicsPipe, Texture, Vec3
import numpy as np
import random
import pickle
import os
import time
import math

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TEST_MODE = True         # <--- TRUE = Watch Mode, FALSE = Train Mode
GEN_SIZE = 48            
MATCH_DURATION = 20     
VISION_RES = 12         
MUTATION_RATE = 0.2     
MUTATION_STRENGTH = 0.3 
LOAD_SAVE = True       
LOAD_FILE = "backup_gen_130.pkl"   # <--- The old file you want to resume from
SAVE_FILE = "trained_model.pkl"      # <--- The new name you want to save to  

# --- SAFETY SYSTEMS ---
BACKUP_INTERVAL = 5     
ANTI_SPIN_THRESHOLD = 0.1 
MAX_BAD_GENS = 3         
# ==========================================

app = Ursina(borderless=False, vsync=True if TEST_MODE else False)

from ursina import application
base = application.base

window.title = "Tag AI: Spectator Mode" if TEST_MODE else "Tag AI: Evolution"
window.size = (1280, 720)
window.color = color.black

# Enable main camera for spectator mode
camera.ui.enabled = True 
camera.enabled = True if TEST_MODE else False   

# --- NEURAL NETWORK ---
class SimpleBrain:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Xavier Initialization
        scale1 = 1.0 / np.sqrt(input_size)
        self.w1 = np.random.uniform(-scale1, scale1, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        
        scale2 = 1.0 / np.sqrt(hidden_size)
        self.w2 = np.random.uniform(-scale2, scale2, (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

        # BIAS: [Strafe, Forward, Jump, Turn]
        self.b2[1] = 0.5 

    def forward(self, inputs):
        self.z1 = np.tanh(np.dot(inputs, self.w1) + self.b1)
        output = np.tanh(np.dot(self.z1, self.w2) + self.b2)
        return output 

    def mutate(self):
        if random.random() < MUTATION_RATE:
            mask1 = np.random.choice([0, 1], size=self.w1.shape, p=[1-MUTATION_RATE, MUTATION_RATE])
            self.w1 += np.random.randn(*self.w1.shape) * MUTATION_STRENGTH * mask1
            
            mask2 = np.random.choice([0, 1], size=self.w2.shape, p=[1-MUTATION_RATE, MUTATION_RATE])
            self.w2 += np.random.randn(*self.w2.shape) * MUTATION_STRENGTH * mask2
            
            if random.random() < 0.2:
                self.b2 += np.random.randn(self.output_size) * 0.1
            
            if random.random() < 0.05:
                self.b2[3] = 0.0

    def clone(self):
        clone = SimpleBrain(self.input_size, self.hidden_size, self.output_size)
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
        
        # PHYSICS & STATS
        self.speed = 15
        self.turn_speed = 200
        self.jump_force = 15
        
        # VISUALS
        self.model = 'cube'
        self.color = color.red if role == "tagger" else color.azure
        self.scale = (1, 1, 1) 
        self.collider = 'box' 
        
        self.velocity_y = 0
        self.grounded = False
        self.jump_cooldown = 0 
        
        # MEMORY
        self.last_actions = np.zeros(4) 
        self.stuck_timer = 0
        self.last_position = self.position
        
        # FITNESS
        self.fitness_score = 0
        self.min_dist_to_target = 100.0 
        self.time_in_sight = 0.0
        
        # BRAIN SETUP
        self.vision_res = VISION_RES 
        input_nodes = (self.vision_res * self.vision_res) + 13
        self.brain = brain if brain else SimpleBrain(input_nodes, 32, 4) 

        self.tex_buffer = Texture()
        self.setup_vision()
        self.frame_skip = random.randint(0, 3) 

    def setup_vision(self):
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
            self.cam_node.get_lens().set_fov(100) 
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
        arr = np.frombuffer(img, dtype=np.uint8).astype(np.float32) / 255.0
        arr = arr.reshape((self.vision_res, self.vision_res, 3))
        return arr[:, :, 1].flatten() 

    def get_whiskers(self):
        sensors = []
        angles = [-45, -20, 0, 20, 45]
        for angle in angles:
            r_rad = np.radians(self.rotation_y + angle)
            dx = np.sin(r_rad)
            dz = np.cos(r_rad)
            dir_vec = Vec3(dx, 0, dz).normalized()
            ray = raycast(self.position + Vec3(0,0.5,0), dir_vec, distance=8, ignore=(self,))
            sensors.append(ray.distance / 8.0 if ray.hit else 1.0)
        return np.array(sensors)

    def get_compass(self, target):
        vec = target.position - self.position
        dist = vec.length()
        angle_to_target = np.arctan2(vec.x, vec.z) - np.radians(self.rotation_y)
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi
        return np.array([angle_to_target / np.pi, 1.0 - min(dist, 40.0) / 40.0])

    def act(self, target, dt):
        # PHYSICS
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

        # BRAIN
        self.frame_skip += 1
        if self.frame_skip % 2 == 0: 
            vision = self.get_vision_data()
            whiskers = self.get_whiskers() 
            compass = self.get_compass(target)
            mem = self.last_actions
            aux = np.array([1.0 if self.grounded else 0.0, np.sin(time.time() * 3)])
            inputs = np.concatenate((vision, whiskers, compass, mem, aux))
            self.decision = self.brain.forward(inputs)
            self.last_actions = self.decision

        # MOVE
        if hasattr(self, 'decision'):
            strafe, fwd, jump_trig, turn = self.decision
            self.rotation_y += turn * self.turn_speed * dt
            move_vec = (self.forward * fwd + self.right * strafe).normalized()
            move_dist = self.speed * dt
            
            if move_vec.length() > 0.01:
                # --- WALL CLIPPING FIX (LAG SPIKE PROTECTION) ---
                check_dist = move_dist + 0.5 
                if not raycast(self.position+Vec3(0,0.5,0), move_vec, distance=check_dist, ignore=(self, target)).hit:
                    self.position += move_vec * move_dist
                    
            if jump_trig > 0.6 and self.grounded and self.jump_cooldown <= 0:
                self.velocity_y = self.jump_force
                self.grounded = False; self.jump_cooldown = 0.4

            # --- STATS ---
            dist = distance(self.position, target.position)
            if dist < self.min_dist_to_target: self.min_dist_to_target = dist
            
            # --- STRICT STUCK CHECK (Anti-Spin) ---
            if distance(self.position, self.last_position) < 2.0:
                self.stuck_timer += dt * 2
            else:
                self.stuck_timer = max(0, self.stuck_timer - dt)
            self.last_position = self.position

            if self.role == "tagger":
                to_target = (target.position - self.position).normalized()
                if self.forward.dot(to_target) > 0.8:
                    los = raycast(self.position+Vec3(0,0.5,0), to_target, distance=dist+1, ignore=(self,))
                    if los.hit and los.entity == target: self.time_in_sight += dt

        if self.y < -10: 
            self.y = 15; self.x = self.origin_x; self.velocity_y = 0; self.fitness_score -= 500

    def on_destroy(self):
        if hasattr(self, 'buffer') and self.buffer: base.graphicsEngine.remove_window(self.buffer)
        if hasattr(self, 'cam_np'): self.cam_np.remove_node()

# --- GRID CAMERA ---
class GridCameraSystem:
    def __init__(self, agents):
        self.cameras = []
        self.regions = []
        viewable = agents[:12]
        if not viewable: return
        cols = int(np.ceil(np.sqrt(len(viewable))))
        rows = int(np.ceil(len(viewable) / cols))
        for i in range(len(viewable)):
            t, r = viewable[i]
            col = i % cols; row = i // cols
            l, r_edge = col / cols, (col + 1) / cols
            b, t_edge = 1.0 - ((row + 1) / rows), 1.0 - (row / rows)
            dr = base.win.makeDisplayRegion(l, r_edge, b, t_edge)
            dr.set_clear_color_active(True); dr.set_clear_color((0.1, 0.1, 0.1, 1))
            cam = PandaCamera(f'grid_cam_{i}')
            cam.set_lens(PerspectiveLens()); cam.get_lens().set_fov(90)
            cam.set_scene(scene)
            cnp = NodePath(cam); cnp.reparent_to(t); cnp.set_pos(0, 6, -9); cnp.look_at(t)
            dr.set_camera(cnp)
            self.cameras.append(cnp); self.regions.append(dr)

    def cleanup(self):
        for dr in self.regions: base.win.remove_display_region(dr)
        for cam in self.cameras: cam.remove_node()

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
        self.time_scale = 1.0
        
        # SPECTATOR VARS
        self.spectate_tagger = True 
        
        # CAMERA CONTROL VARS
        self.free_look = False
        self.cam_yaw = 0
        self.cam_pitch = 20 # Locked pitch
        self.cam_dist = 22  # Starting distance
        self.faded_walls = [] 
        
        # Smooth Follow Vars
        self.smooth_focus = Vec3(0,0,0)

        # Hysteresis (Auto Cam)
        self.cam_reverse_timer = 0.0  
        self.cam_is_reversed = False  

        # MONITORING VARS
        self.bad_gen_count = 0
        self.force_reset_next = False

        # --- UI ---
        self.ui_gen = Text(text="Gen: 1", position=(-0.85, 0.45), scale=1.5, color=color.white)
        self.ui_status = Text(text="Status: OK", position=(-0.85, 0.40), scale=1.2, color=color.green)
        self.ui_timer = Text(text="0.0s", position=(0, 0.45), scale=1.5, origin=(0,0), color=color.white)
        self.ui_speed = Text(text="Speed: 1.0x", position=(0.50, 0.45), scale=1.5, color=color.yellow)
        self.ui_cam = Text(text="Cam: AUTO", position=(0.50, 0.40), scale=1.2, color=color.green)
        self.ui_info = Text(text="[SPACE] Switch Target | [F] Free Look | [0-5] Speed", position=(0.50, 0.35), scale=0.8, color=color.white)
        
        if LOAD_SAVE: self.load_brains()
        self.ui_gen.text = f"Gen: {self.generation}" if not TEST_MODE else "DEMO MODE"
        self.start_generation()

    def input(self, key):
        # SPEED CONTROLS
        speed_map = {
            '0': 0.0, 'numpad0': 0.0,
            '1': 0.25, 'numpad1': 0.25,
            '2': 0.5, 'numpad2': 0.5,
            '3': 1.0, 'numpad3': 1.0,
            '4': 2.0, 'numpad4': 2.0,
            '5': 5.0, 'numpad5': 5.0,
            '9': 15.0, 'numpad9': 15.0,
        }

        if key in speed_map:
            self.time_scale = speed_map[key]
            status_text = "PAUSED" if self.time_scale == 0 else f"{self.time_scale}x"
            self.ui_speed.text = f"Speed: {status_text}"
            self.ui_speed.color = color.red if self.time_scale == 0 else color.yellow
        
        if key == 'space':
            self.spectate_tagger = not self.spectate_tagger
        
        # TOGGLE FREE LOOK
        if key == 'f':
            self.free_look = not self.free_look
            if self.free_look:
                self.ui_cam.text = "Cam: MANUAL"
                self.ui_cam.color = color.orange
                mouse.locked = True
                
                # --- PERFECT SYNC FIX ---
                if self.agents:
                    tagger, runner = self.agents[0]
                    target = tagger if self.spectate_tagger else runner
                    if not target.enabled: target = runner if self.spectate_tagger else tagger

                    diff = camera.position - target.position
                    self.cam_dist = diff.length()
                    
                    angle_rad = math.atan2(diff.x, diff.z)
                    self.cam_yaw = math.degrees(angle_rad)
                    self.smooth_focus = target.position

            else:
                self.ui_cam.text = "Cam: AUTO"
                self.ui_cam.color = color.green
                mouse.locked = False
        
        # ZOOM WITH SCROLL
        if self.free_look:
            if key == 'scroll up': self.cam_dist = max(5, self.cam_dist - 2)
            if key == 'scroll down': self.cam_dist = min(50, self.cam_dist + 2)
        

    def start_generation(self):
        print(f"--- STARTING GEN {self.generation} ---")
        self.cleanup()
        
        local_gen_size = 1 if TEST_MODE else GEN_SIZE
        self.scores = [(0,0)] * local_gen_size
        input_size = (VISION_RES * VISION_RES) + 13 
        
        if not self.population:
            for _ in range(local_gen_size):
                self.population.append((SimpleBrain(input_size, 32, 4), SimpleBrain(input_size, 32, 4)))
        
        while len(self.population) < local_gen_size:
             self.population.append((SimpleBrain(input_size, 32, 4), SimpleBrain(input_size, 32, 4)))
        
        self.population = self.population[:local_gen_size]

        if not TEST_MODE and self.force_reset_next:
            print(">>> EMERGENCY INTERVENTION: FORCING STRAIGHT RUNNING <<<")
            self.ui_status.text = "Status: FORCED RESET"
            self.ui_status.color = color.red
            for (t_brain, r_brain) in self.population:
                r_brain.b2[3] = 0.0 
            self.force_reset_next = False
            self.bad_gen_count = 0
        else:
            self.ui_status.text = "Status: OK"
            self.ui_status.color = color.green

        self.spawn_walls(seed=self.generation if not TEST_MODE else random.randint(0,9999))
        
        for i in range(local_gen_size):
            t_brain, r_brain = self.population[i]
            offset_x = i * 100 
            tx, tz = self.get_safe_spawn(offset_x)
            rx, rz = self.get_safe_spawn(offset_x)
            if TEST_MODE:
                while distance((tx,0,tz), (rx,0,rz)) < 15:
                    rx, rz = self.get_safe_spawn(offset_x)

            tagger = Agent("tagger", i, offset_x, self, brain=t_brain, position=(tx, 2, tz))
            runner = Agent("runner", i, offset_x, self, brain=r_brain, position=(rx, 2, rz))
            self.agents.append((tagger, runner))

        if not TEST_MODE:
            self.grid_sys = GridCameraSystem(self.agents)

        self.time_elapsed = 0
        self.active = True

    def spawn_walls(self, seed):
        self.walls = []
        self.wall_data = [] 
        random.seed(seed)
        def add_wall(pos, scale, mat='brick'):
            pad = 2.5 
            min_x = pos[0]-scale[0]/2-pad; max_x = pos[0]+scale[0]/2+pad
            min_z = pos[2]-scale[2]/2-pad; max_z = pos[2]+scale[2]/2+pad
            if mat != 'grass' and isinstance(mat, str):
                for (wx1, wx2, wz1, wz2) in self.wall_data:
                    if (min_x < wx2 and max_x > wx1 and min_z < wz2 and max_z > wz1): return 
                self.wall_data.append((min_x, max_x, min_z, max_z))
            
            if isinstance(mat, str):
                w = Entity(model='cube', position=pos, scale=scale, texture=mat, collider='box')
                w.texture_scale = ((scale[0]+scale[2])/2, scale[1])
            else:
                w = Entity(model='cube', position=pos, scale=scale, color=mat, collider='box')
            self.walls.append(w)

        loop_count = 1 if TEST_MODE else GEN_SIZE
        for i in range(loop_count):
            off = i * 100
            add_wall((off, 0, 0), (50, 1, 50), 'grass')
            add_wall((off, 3, 25), (50, 6, 1), color.gray)
            add_wall((off, 3, -25), (50, 6, 1), color.gray)
            add_wall((off + 25, 3, 0), (1, 6, 50), color.gray)
            add_wall((off - 25, 3, 0), (1, 6, 50), color.gray)
            self.wall_data.append((off - 26, off + 26, 24, 26))   
            self.wall_data.append((off - 26, off + 26, -26, -24)) 
            self.wall_data.append((off + 24, off + 26, -26, 26))  
            self.wall_data.append((off - 26, off - 24, -26, 26))  

            for _ in range(25): 
                wx = random.randint(-18, 18); wz = random.randint(-18, 18)
                if random.random() < 0.4: add_wall((wx + off, 1, wz), (random.randint(2,4), 2, random.randint(2,4)), 'brick')
                else: add_wall((wx + off, 3, wz), (random.randint(3,6), 6, random.randint(3,6)), 'brick')

    def get_safe_spawn(self, offset_x):
        for _ in range(100):
            x, z = offset_x + random.uniform(-18, 18), random.uniform(-18, 18)
            valid = True
            for (mx1, mx2, mz1, mz2) in self.wall_data:
                if mx1 < x < mx2 and mz1 < z < mz2: valid = False; break
            if valid: return x, z
        return offset_x, 0

    def update(self):
        if not self.active: return

        # --- UPDATE TIME ---
        if TEST_MODE:
            if self.time_scale > 0:
                dt = min(time.dt * self.time_scale, 0.1)
                self.step_simulation(dt)
        else:
            step_dt = 0.04 
            steps = int(self.time_scale)
            for _ in range(steps): 
                self.step_simulation(step_dt)

        self.ui_timer.text = f"{self.time_elapsed:.1f}s / {MATCH_DURATION}s"

        # ==========================================
        # --- CAMERA & VISUALS ---
        # ==========================================
        if TEST_MODE and self.agents:
            tagger, runner = self.agents[0]
            target = tagger if self.spectate_tagger else runner
            other = runner if self.spectate_tagger else tagger
            
            if not target.enabled: 
                target = runner if self.spectate_tagger else tagger
                other = tagger if self.spectate_tagger else runner

            target.visible = True 
            other.visible = True

            target_pos = target.position

            # --- WALL TRANSPARENCY (X-RAY) ---
            for w in self.faded_walls:
                w.alpha = 1.0
            self.faded_walls.clear()

            cam_to_target = target_pos - camera.position
            dist = cam_to_target.length()
            
            for w in self.walls:
                if distance(camera.position, w.position) < 4:
                     w.alpha = 0.2
                     self.faded_walls.append(w)

            if dist > 1:
                hits = raycast(camera.position, cam_to_target.normalized(), distance=dist-1, ignore=(tagger, runner))
                if hits.hit:
                    hits.entity.alpha = 0.3
                    self.faded_walls.append(hits.entity)


            # --- CAMERA LOGIC ---
            if self.free_look:
                # MANUAL ORBIT MODE (LEFT/RIGHT ONLY)
                self.cam_yaw += mouse.velocity[0] * 100
                self.cam_pitch = 20 # Locked pitch
                
                # Math for instant rotation (NO LERP)
                yaw_rad = math.radians(self.cam_yaw)
                pitch_rad = math.radians(self.cam_pitch)
                
                h_dist = self.cam_dist * math.cos(pitch_rad)
                v_dist = self.cam_dist * math.sin(pitch_rad)
                
                x_off = math.sin(yaw_rad) * h_dist
                z_off = math.cos(yaw_rad) * h_dist
                
                # Smoothly lerp the *focus point* (Player position)
                self.smooth_focus = lerp(self.smooth_focus, target_pos, time.dt * 10)
                
                camera.position = self.smooth_focus + Vec3(x_off, v_dist, z_off)
                camera.look_at(self.smooth_focus + Vec3(0, 1, 0))
                
                # --- FIX: FORCE HORIZON LEVEL ---
                camera.rotation_z = 0
                
            else:
                # AUTOMATIC SMART MODE WITH DELAY (HYSTERESIS)
                current_reversing = False
                if hasattr(target, 'decision'):
                    if target.decision[1] < -0.1: current_reversing = True
                
                if current_reversing != self.cam_is_reversed:
                    self.cam_reverse_timer += time.dt
                    if self.cam_reverse_timer > 0.5:
                        self.cam_is_reversed = current_reversing
                        self.cam_reverse_timer = 0
                else:
                    self.cam_reverse_timer = 0
                
                if self.cam_is_reversed:
                    offset_dir = target.forward * 22 
                else:
                    offset_dir = target.forward * -22

                desired_pos = target_pos + offset_dir + Vec3(0, 10, 0)
                
                # For auto mode, we lerp position directly
                camera.position = lerp(camera.position, desired_pos, time.dt * 6)
                look_target = target_pos + Vec3(0, 2, 0)
                camera.look_at(look_target)
                camera.rotation_z = 0
                
                # Sync smooth focus variable so it's ready when we switch
                self.smooth_focus = target_pos

    def step_simulation(self, dt):
        self.time_elapsed += dt
        active_count = 0
        for i, (tagger, runner) in enumerate(self.agents):
            if not tagger.enabled: continue
            active_count += 1
            tagger.act(runner, dt)
            runner.act(tagger, dt)
            dist = distance(tagger.position, runner.position)

            # PENALTY: Radius 2.0 = -100 Score
            if tagger.stuck_timer > 2.0: tagger.fitness_score -= 100 * dt
            if runner.stuck_timer > 2.0: runner.fitness_score -= 100 * dt

            if dist < 6.0: runner.fitness_score -= 10 * dt 
            if dist < 1.3: self.finish_pair(i, winner="tagger")
            
        if self.time_elapsed > MATCH_DURATION:
            for i, (tagger, runner) in enumerate(self.agents):
                if tagger.enabled: self.finish_pair(i, winner="runner")
        
        if active_count == 0: 
            if TEST_MODE:
                # Infinite Loop for testing
                self.start_generation()
            else:
                self.evolve()

    def finish_pair(self, index, winner):
        t, r = self.agents[index]
        start_dist = 20.0
        dist_improv = max(0, start_dist - t.min_dist_to_target) * 10
        sight_bonus = min(t.time_in_sight * 20, 300)
        t_final = dist_improv + sight_bonus + t.fitness_score
        
        if winner == "tagger":
            time_bonus = (MATCH_DURATION - self.time_elapsed) * 20
            t_final += 500 + time_bonus
            r_final = self.time_elapsed * 10 
            if TEST_MODE: print("TAGGER WON!")
        else:
            r_final = 500 + r.fitness_score 
            r_final += distance(t.position, r.position) * 5 
            if TEST_MODE: print("RUNNER ESCAPED!")
        
        self.scores[index] = (t_final, r_final)
        t.enabled = False; r.enabled = False; t.visible = False; r.visible = False

    def evolve(self):
        print("--- EVOLVING ---")
        sorted_taggers = sorted(zip(self.population, self.scores), key=lambda x: x[1][0], reverse=True)
        sorted_runners = sorted(zip(self.population, self.scores), key=lambda x: x[1][1], reverse=True)
        
        # Save & Check Health
        best_t, best_r = sorted_taggers[0][0][0], sorted_runners[0][0][1]
        self.save_brains(best_t, best_r)
        
        # --- HEALTH MONITORING ---
        turn_bias = abs(best_r.b2[3])
        print(f"DEBUG: Best Runner Turn Bias: {turn_bias:.4f}")
        if turn_bias > ANTI_SPIN_THRESHOLD:
            self.bad_gen_count += 1
            print(f"WARNING: High Turn Bias detected ({self.bad_gen_count}/{MAX_BAD_GENS})")
        else:
            self.bad_gen_count = 0
            
        if self.bad_gen_count >= MAX_BAD_GENS:
            self.force_reset_next = True
        # -------------------------

        if self.stop_requested: application.quit(); return

        new_pop = []
        # ELITISM
        for i in range(3):
            new_pop.append((sorted_taggers[i][0][0].clone(), sorted_runners[i][0][1].clone()))

        # TOURNAMENT
        def tournament(sorted_list, is_tagger):
            idx1 = random.randint(0, GEN_SIZE // 2)
            idx2 = random.randint(0, GEN_SIZE // 2)
            brain_idx = 0 if is_tagger else 1
            if sorted_list[idx1][1][brain_idx] > sorted_list[idx2][1][brain_idx]:
                return sorted_list[idx1][0][brain_idx].clone()
            else:
                return sorted_list[idx2][0][brain_idx].clone()

        while len(new_pop) < GEN_SIZE:
            p_t = tournament(sorted_taggers, True)
            p_r = tournament(sorted_runners, False)
            p_t.mutate()
            p_r.mutate()
            new_pop.append((p_t, p_r))

        self.population = new_pop
        self.generation += 1
        self.ui_gen.text = f"Gen: {self.generation}"
        self.start_generation()

    def cleanup(self):
        if self.grid_sys: self.grid_sys.cleanup()
        for t, r in self.agents: destroy(t); destroy(r)
        self.agents.clear()
        if hasattr(self, 'walls'):
            for w in self.walls: destroy(w)

    def save_brains(self, t, r):
        data = {"gen":self.generation, "t_brain":t, "r_brain":r}
        with open(SAVE_FILE, 'wb') as f: pickle.dump(data, f)
        if self.generation % BACKUP_INTERVAL == 0:
            with open(f"backup_gen_{self.generation}.pkl", 'wb') as f: pickle.dump(data, f)
        print("Saved Best.")

    def load_brains(self):
        # CHANGE: Check LOAD_FILE instead of SAVE_FILE
        if os.path.exists(LOAD_FILE): 
            try:
                # CHANGE: Open LOAD_FILE
                with open(LOAD_FILE, 'rb') as f:
                    data = pickle.load(f)
                    saved_size = data['t_brain'].w1.shape[0]
                    curr_size = (VISION_RES * VISION_RES) + 13
                    if saved_size != curr_size:
                        print(f"Arch mismatch. Resetting."); return 
                    self.generation = data['gen'] + 1
                    t, r = data['t_brain'], data['r_brain']
                    self.population = []
                    
                    for _ in range(GEN_SIZE):
                        tc, rc = t.clone(), r.clone()
                        tc.mutate(); rc.mutate()
                        self.population.append((tc, rc))
                        
                    self.population[0] = (t, r)
                        
                    print(f"Loaded Gen {self.generation} from {LOAD_FILE}")
            except Exception as e: 
                print(f"Load failed: {e}"); self.generation = 1
        else:
            print(f"Could not find {LOAD_FILE}, starting fresh.")

def req_stop():
    trainer.stop_requested = True
    b_stop.text = "Stopping..."
    b_stop.color = color.red

b_stop = Button(text="Save & Stop", color=color.orange, scale=(0.2, 0.05), position=(-0.7, -0.4), on_click=req_stop)

trainer = TrainingManager()
app.run()
