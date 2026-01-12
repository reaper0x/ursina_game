from ursina import *
import numpy as np
import config
import torch

class Agent(Entity):
    def __init__(self, role, pair_id, origin_x, manager, brain=None, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.pair_id = pair_id 
        self.origin_x = origin_x 
        self.manager = manager
        
        self.active = True
        
        self.speed = config.BASE_SPEED
        self.turn_speed = 200
        
        self.model = 'cube'
        self.color = color.red if role == "tagger" else color.azure
        self.scale = (1, 1, 1) 
        self.collider = 'box' 
        
        self.velocity_y = 0
        self.grounded = False
        self.jump_cooldown = 0 
        
        self.bhop_chain = 0
        self.ground_time = 0.0
        
        self.last_actions = np.zeros(4) 
        self.stuck_timer = 0
        self.last_position = self.position
        
        self.fitness_score = 0
        self.score_breakdown = {}
        self.min_dist_to_target = 100.0 
        self.time_in_sight = 0.0
        
        self.brain = brain
        self.saved_log_probs = []
        self.rewards = []
        self.frame_skip = 0

    def change_score(self, amount, reason):
        self.fitness_score += amount
        self.rewards.append(amount)
        if reason not in self.score_breakdown:
            self.score_breakdown[reason] = 0.0
        self.score_breakdown[reason] += amount

    def get_whiskers(self):
        sensors = []
        angles = [-30, 0, 30]
        for angle in angles:
            r_rad = np.radians(self.rotation_y + angle)
            dx = np.sin(r_rad)
            dz = np.cos(r_rad)
            dir_vec = Vec3(dx, 0, dz).normalized()
            
            ray_low = raycast(self.position + Vec3(0,0.5,0), dir_vec, distance=8, ignore=(self,))
            sensors.append(ray_low.distance / 8.0 if ray_low.hit else 1.0)
            
            ray_high = raycast(self.position + Vec3(0,1.5,0), dir_vec, distance=8, ignore=(self,))
            sensors.append(ray_high.distance / 8.0 if ray_high.hit else 1.0)
            
        return np.array(sensors)

    def get_compass(self, target):
        vec = target.position - self.position
        dist = vec.length()
        angle_to_target = np.arctan2(vec.x, vec.z) - np.radians(self.rotation_y)
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi
        return np.array([angle_to_target / np.pi, 1.0 - min(dist, 40.0) / 40.0])

    def act(self, target, dt):
        if not self.active: return

        self.frame_skip += 1
        if self.frame_skip % 2 == 0: 
            whiskers = self.get_whiskers() 
            compass = self.get_compass(target)
            mem = self.last_actions
            aux = np.array([1.0 if self.grounded else 0.0, self.bhop_chain / 5.0]) 
            inputs = np.concatenate((whiskers, compass, mem, aux))
            
            action, log_prob = self.brain.get_action(inputs)
            self.saved_log_probs.append(log_prob)
            self.last_actions = action

        if not hasattr(self, 'last_actions'): return

        strafe, fwd, jump_trig_val, turn = self.last_actions
        
        self.velocity_y -= config.GRAVITY * dt
        ray = raycast(self.position + Vec3(0,0.1,0), Vec3(0, -1, 0), distance=0.7+abs(self.velocity_y*dt), ignore=(self,))
        
        if ray.hit and self.velocity_y <= 0:
            self.velocity_y = 0
            self.grounded = True
            self.y = ray.world_point.y + 0.5
            self.ground_time += dt
            if self.ground_time > 0.2:
                self.bhop_chain = 0
                self.speed = config.BASE_SPEED
        else:
            self.grounded = False
            self.y += self.velocity_y * dt
            self.ground_time = 0.0

        if self.jump_cooldown > 0: self.jump_cooldown -= dt

        self.rotation_y += turn * self.turn_speed * dt

        effective_speed = self.speed
        if abs(strafe) > 0.1 and abs(fwd) < 0.1:
            effective_speed *= config.STRAFE_PENALTY
        elif abs(strafe) > 0.1 and abs(fwd) > 0.1:
            effective_speed *= 0.8
            
        move_vec = (self.forward * fwd + self.right * strafe).normalized()
        move_dist = effective_speed * dt
        
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
                    if raycast(org, move_vec, distance=check_dist, ignore=(self, target)).hit:
                        blocked = True; break
            
            if not blocked:
                self.position += move_vec * move_dist

        if jump_trig_val > 0.1 and self.grounded and self.jump_cooldown <= 0:
            self.velocity_y = config.JUMP_FORCE
            self.grounded = False
            self.jump_cooldown = 0.2
            
            moving_backwards = (fwd < -0.1)

            if self.ground_time < config.BHOP_WINDOW and not moving_backwards:
                self.bhop_chain = min(3, self.bhop_chain + 1)
                self.speed = config.BASE_SPEED + (self.bhop_chain * config.BHOP_SPEED_BOOST)
                self.change_score(self.bhop_chain * 5, "bhop_bonus")
            else:
                self.bhop_chain = 0
                self.speed = config.BASE_SPEED
            
            self.ground_time = 0

        dist = distance(self.position, target.position)
        if dist < self.min_dist_to_target: self.min_dist_to_target = dist
        
        if distance(self.position, self.last_position) < 1.0:
            self.stuck_timer += dt * 2
        else:
            self.stuck_timer = max(0, self.stuck_timer - dt)
        self.last_position = self.position

        if self.role == "tagger":
            to_target = (target.position - self.position).normalized()
            if self.forward.dot(to_target) > 0.7: 
                los = raycast(self.position+Vec3(0,0.5,0), to_target, distance=dist+1, ignore=(self,))
                if los.hit and los.entity == target: self.time_in_sight += dt

        if self.y < -10: 
            self.y = 15; self.x = self.origin_x; self.z = 0
            self.velocity_y = 0; 
            self.change_score(-500, "fall_penalty")