from ursina import *
from panda3d.core import Vec3
import random

# --- CONFIG ---
GRAVITY = 40
JUMP_FORCE = 15
MOUSE_SENSITIVITY = 40
BASE_SPEED = 10 
BHOP_WINDOW = 0.15 
BHOP_SPEED_BOOST = 2.0
STRAFE_PENALTY = 0.6  

app = Ursina()
window.title = "Physics Tester (Strategic Map)"
window.color = color.black
window.borderless = False
window.exit_button.visible = False

Sky(texture='sky_sunset')

mouse.locked = True

class DebugPlayer(Entity):
    def __init__(self, position=(0,2,0)):
        super().__init__(
            model='cube',
            color=color.azure,
            scale=(1, 1, 1),
            collider='box',
            position=position
        )

        self.velocity_y = 0
        self.grounded = False
        self.jump_cooldown = 0
        self.ground_time = 0.0
        self.bhop_chain = 0
        self.speed = BASE_SPEED
        self.jump_ready = True 
        
        camera.parent = self
        camera.position = (0, 0.5, 0)
        camera.rotation = (0,0,0)
        camera.fov = 90

    def input(self, key):
        if key == 'escape':
            mouse.locked = not mouse.locked
        if key == 'r':
            self.position = (0, 10, 0)
            self.velocity_y = 0

    def update(self):
        dt = time.dt

        if mouse.locked:
            self.rotation_y += mouse.velocity[0] * MOUSE_SENSITIVITY
            camera.rotation_x -= mouse.velocity[1] * MOUSE_SENSITIVITY
            camera.rotation_x = clamp(camera.rotation_x, -90, 90)

        # --- INPUT ---
        fwd = held_keys['w'] - held_keys['s']
        strafe = held_keys['d'] - held_keys['a']
        
        jump_trig = False
        if held_keys['space']:
            if self.jump_ready:
                jump_trig = True; self.jump_ready = False
        else:
            self.jump_ready = True

        # --- GRAVITY ---
        self.velocity_y -= GRAVITY * dt
        ray = raycast(self.position + Vec3(0, 0.1, 0), Vec3(0, -1, 0), 
                      distance=0.7 + abs(self.velocity_y * dt), ignore=(self,))
        
        if ray.hit and self.velocity_y <= 0:
            self.velocity_y = 0
            self.grounded = True
            self.y = ray.world_point.y + 0.5
            self.ground_time += dt
            if self.ground_time > 0.2:
                self.bhop_chain = 0; self.speed = BASE_SPEED
        else:
            self.grounded = False
            self.y += self.velocity_y * dt
            self.ground_time = 0.0

        if self.jump_cooldown > 0: self.jump_cooldown -= dt

        # --- MOVEMENT ---
        move_vec = (self.forward * fwd + self.right * strafe).normalized()
        
        effective_speed = self.speed
        if abs(strafe) > 0 and abs(fwd) < 0.1:
            effective_speed *= STRAFE_PENALTY
        elif abs(strafe) > 0 and abs(fwd) > 0:
            effective_speed *= 0.8
            
        move_dist = effective_speed * dt

        # --- COLLISION ---
        if move_vec.length() > 0.01:
            check_dist = move_dist + 0.5
            perp_vec = move_vec.cross(Vec3(0, 1, 0)).normalized()
            shoulder_width = 0.45
            heights = [0.1, 0.5, 1.5]
            blocked = False
            
            for h in heights:
                if blocked: break
                origins = [
                    self.position + Vec3(0, h, 0),
                    self.position + Vec3(0, h, 0) + (perp_vec * shoulder_width),
                    self.position + Vec3(0, h, 0) - (perp_vec * shoulder_width)
                ]
                for org in origins:
                    if raycast(org, move_vec, distance=check_dist, ignore=(self,)).hit:
                        blocked = True; break
            
            if not blocked:
                self.position += move_vec * move_dist

        # --- JUMPING ---
        if jump_trig and self.grounded and self.jump_cooldown <= 0:
            self.velocity_y = JUMP_FORCE
            self.grounded = False
            self.jump_cooldown = 0.2
            
            moving_backwards = (fwd < -0.1)

            if self.ground_time < BHOP_WINDOW and not moving_backwards:
                self.bhop_chain = min(3, self.bhop_chain + 1)
                self.speed = BASE_SPEED + (self.bhop_chain * BHOP_SPEED_BOOST)
                print(f"B-HOP! Chain: {self.bhop_chain} (Speed: {self.speed})")
            else:
                self.bhop_chain = 0; self.speed = BASE_SPEED

            self.ground_time = 0

        if self.y < -10: self.position = (0, 2, 0); self.velocity_y = 0

# --- ENVIRONMENT GENERATOR ---

ground = Entity(
    model='plane', 
    scale=(200, 1, 200), 
    color=color.rgb(30, 30, 30), 
    texture='white_cube', 
    texture_scale=(100, 100), 
    collider='box'
)

roof = Entity(
    model='plane', 
    position=(0, 30, 0), 
    scale=(200, 1, 200), 
    color=color.white, 
    alpha=0.1, 
    rotation_z=180, 
    collider='box'
)

walls = []
boundaries = [
    ((0, 15, 50), (100, 30, 1)), ((0, 15, -50), (100, 30, 1)),
    ((50, 15, 0), (1, 30, 100)), ((-50, 15, 0), (1, 30, 100))
]
wall_data = [] 

for pos, scale in boundaries:
    w = Entity(
        model='cube', 
        position=pos, 
        scale=scale, 
        color=color.rgb(50, 50, 50), 
        collider='box', 
        texture='white_cube',
        texture_scale=(scale[0]/2, scale[1]/2)
    )
    walls.append(w)
    min_x, max_x = pos[0]-scale[0]/2, pos[0]+scale[0]/2
    min_z, max_z = pos[2]-scale[2]/2, pos[2]+scale[2]/2
    wall_data.append((min_x, max_x, min_z, max_z))

def is_overlapping(new_box, padding=2.5): # Increased padding for wider lanes
    nx1, nx2, nz1, nz2 = new_box
    for (wx1, wx2, wz1, wz2) in wall_data:
        if (nx1 - padding < wx2 and nx2 + padding > wx1 and nz1 - padding < wz2 and nz2 + padding > wz1): return True
    return False

print("Generating Balanced Map...")
random.seed(999) # New seed for better layout
count = 0

# --- STRATEGIC GENERATION (70 Walls max) ---
target_walls = 70 

for _ in range(target_walls):
    placed = False; attempts = 0
    while not placed and attempts < 20:
        attempts += 1
        
        # Determine Wall Type
        wall_type = random.random()
        
        if wall_type < 0.60: 
            # 60% Standard Block (Good for jumping/cover)
            w_width = random.randint(4, 8)
            w_length = random.randint(4, 8)
            w_height = random.randint(4, 7)
            
        elif wall_type < 0.85: 
            # 25% Long Wall (Barriers/Lanes)
            if random.random() < 0.5: # Horizontal
                w_width = random.randint(15, 25)
                w_length = 2
            else: # Vertical
                w_width = 2
                w_length = random.randint(15, 25)
            w_height = random.randint(4, 6)
            
        else: 
            # 15% Tall Pillar (Landmarks)
            w_width = random.randint(2, 4)
            w_length = random.randint(2, 4)
            w_height = random.randint(10, 20)

        wx = random.randint(-45, 45)
        wz = random.randint(-45, 45)
        
        min_x, max_x = wx - w_width/2, wx + w_width/2
        min_z, max_z = wz - w_length/2, wz + w_length/2
        
        player_safe = (-4 < wx < 4) and (-4 < wz < 4) # Larger safe zone
        
        if not player_safe and not is_overlapping((min_x, max_x, min_z, max_z)):
            w = Entity(
                model='cube', 
                position=(wx, w_height/2, wz), 
                scale=(w_width, w_height, w_length), 
                color=color.orange, 
                collider='box', 
                texture='white_cube',
                texture_scale=(w_width/2, w_height/2)
            )
            walls.append(w)
            wall_data.append((min_x, max_x, min_z, max_z))
            placed = True
            count += 1

print(f"Map Generated: {count} strategic walls created.")
Text(text="[ESC] Unlock Mouse | [R] Reset Position", position=(-0.65, 0.45), scale=1.2)

player = DebugPlayer()
app.run()