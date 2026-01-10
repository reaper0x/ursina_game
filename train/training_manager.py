from ursina import *
import random
import pickle
import os
import time
import math
import numpy as np
from brain import SimpleBrain
from agent import Agent
from camera_system import GridCameraSystem
import config

class TrainingManager(Entity):
    def __init__(self, log_manager):
        super().__init__()
        self.log_manager = log_manager
        self.generation = 1
        self.population = []
        self.agents = []
        self.scores = []
        self.detailed_scores_cache = []
        self.wall_data = []
        self.time_elapsed = 0
        self.active = False
        self.stop_requested = False
        self.grid_sys = None
        self.time_scale = 1.0
        
        self.current_mutation_rate = config.MUTATION_RATE_START
        self.current_wall_count = config.STARTING_WALLS
        self.stagnant_gens = 0
        self.best_historical_score = 0
        
        self.spectate_tagger = True
        self.free_look = False
        self.cam_yaw = 0; self.cam_pitch = 20; self.cam_dist = 22
        self.faded_walls = []; self.smooth_focus = Vec3(0,0,0)
        self.cam_reverse_timer = 0.0; self.cam_is_reversed = False
        self.bad_gen_count = 0; self.force_reset_next = False

        self.ui_gen = Text(text="Gen: 1", position=(-0.85, 0.45), scale=1.5, color=color.white)
        self.ui_status = Text(text="Status: OK", position=(-0.85, 0.40), scale=1.2, color=color.green)
        self.ui_timer = Text(text="0.0s", position=(0, 0.45), scale=1.5, origin=(0,0), color=color.white)
        self.ui_speed = Text(text="Speed: 1.0x", position=(0.50, 0.45), scale=1.5, color=color.yellow)
        self.ui_cam = Text(text="Cam: AUTO", position=(0.50, 0.40), scale=1.2, color=color.green)
        self.ui_info = Text(text="Info: Init", position=(0.50, 0.35), scale=0.8, color=color.white)
        
        if config.LOAD_SAVE: self.load_brains()
        self.ui_gen.text = f"Gen: {self.generation}" if not config.TEST_MODE else "DEMO MODE"
        self.start_generation()

    def input(self, key):
        speed_map = {'0':0, '1':0.25, '2':0.5, '3':1, '4':2, '5':5, '9':15}
        if key in speed_map:
            self.time_scale = speed_map[key]
            status_text = "PAUSED" if self.time_scale == 0 else f"{self.time_scale}x"
            self.ui_speed.text = f"Speed: {status_text}"
            self.ui_speed.color = color.red if self.time_scale == 0 else color.yellow
        
        if key == 'space': self.spectate_tagger = not self.spectate_tagger
        if key == 'f':
            self.free_look = not self.free_look
            if self.free_look:
                self.ui_cam.text = "Cam: MANUAL"; self.ui_cam.color = color.orange; mouse.locked = True
            else:
                self.ui_cam.text = "Cam: AUTO"; self.ui_cam.color = color.green; mouse.locked = False
        
        if self.free_look:
            if key == 'scroll up': self.cam_dist = max(5, self.cam_dist - 2)
            if key == 'scroll down': self.cam_dist = min(50, self.cam_dist + 2)

    def update_adaptive_parameters(self):
        decayed = config.MUTATION_RATE_START - (self.generation * config.MUTATION_DECAY_SPEED)
        self.current_mutation_rate = max(config.MUTATION_RATE_END, decayed)

        difficulty_level = int(self.best_historical_score / config.SCORE_TO_INCREASE_DIFFICULTY)
        extra_walls = difficulty_level * 5
        self.current_wall_count = min(config.MAX_WALLS, config.STARTING_WALLS + extra_walls)

        self.ui_info.text = f"Mut: {self.current_mutation_rate:.2f} | Walls: {self.current_wall_count} | Stag: {self.stagnant_gens}"

    def start_generation(self):
        self.update_adaptive_parameters()

        mut_mult = 1.0
        if self.stagnant_gens > config.EXTINCTION_LIMIT:
             self.log_manager.log_attention("!!! MASS EXTINCTION EVENT INITIATED !!!")
             self.ui_status.text = "Status: EXTINCTION EVENT"
             self.ui_status.color = color.red
        elif self.stagnant_gens > config.STAGNATION_LIMIT:
            mut_mult = 3.0 
            self.log_manager.log_attention("!!! PANIC MODE ACTIVATED (High Mutation) !!!")
            self.ui_status.text = "Status: PANIC MODE"
            self.ui_status.color = color.orange
        else:
            self.ui_status.text = "Status: OK"
            self.ui_status.color = color.green

        self.log_manager.log_generation_start(self.generation, mut_mult) 
        self.cleanup()
        
        if config.TEST_MODE:
            local_gen_size = 1
        else:
            local_gen_size = max(1, config.GEN_SIZE // config.NUM_PROCESSES)
            
        self.scores = [(0,0)] * local_gen_size
        self.detailed_scores_cache = [None] * local_gen_size
        
        input_size = 14
        
        if not self.population:
            for _ in range(local_gen_size):
                self.population.append((SimpleBrain(input_size, config.HIDDEN_LAYER_SIZE, 4), SimpleBrain(input_size, config.HIDDEN_LAYER_SIZE, 4)))
        
        while len(self.population) < local_gen_size:
             self.population.append((SimpleBrain(input_size, config.HIDDEN_LAYER_SIZE, 4), SimpleBrain(input_size, config.HIDDEN_LAYER_SIZE, 4)))
             
        self.population = self.population[:local_gen_size]

        if not config.TEST_MODE and self.force_reset_next:
            self.log_manager.log_attention("EMERGENCY: Resetting Spin Bias due to Bad Genes")
            for (t_brain, r_brain) in self.population:
                r_brain.b2[3] = 0.0 
            self.force_reset_next = False
            self.bad_gen_count = 0

        self.spawn_walls(seed=self.generation if not config.TEST_MODE else random.randint(0,9999))
        
        for i in range(local_gen_size):
            t_brain, r_brain = self.population[i]
            offset_x = i * 150
            tx, tz = self.get_safe_spawn(offset_x)
            rx, rz = self.get_safe_spawn(offset_x)
            
            if config.TEST_MODE:
                while distance((tx,0,tz), (rx,0,rz)) < 15:
                    rx, rz = self.get_safe_spawn(offset_x)

            tagger = Agent("tagger", i, offset_x, self, brain=t_brain, position=(tx, 2, tz))
            runner = Agent("runner", i, offset_x, self, brain=r_brain, position=(rx, 2, rz))
            self.agents.append((tagger, runner))

        if not config.TEST_MODE:
            self.grid_sys = GridCameraSystem(self.agents)

        self.time_elapsed = 0
        self.active = True
        self.gen_start_time = time.time()

    def spawn_walls(self, seed):
        self.walls = []
        self.wall_data = []
        random.seed(seed)
        
        def is_overlapping(new_box, padding=2.5):
            nx1, nx2, nz1, nz2 = new_box
            for (wx1, wx2, wz1, wz2) in self.wall_data:
                if (nx1 - padding < wx2 and nx2 + padding > wx1 and 
                    nz1 - padding < wz2 and nz2 + padding > wz1):
                    return True
            return False

        def create_wall_entity(pos, scale, mat):
            if isinstance(mat, str):
                w = Entity(model='cube', position=pos, scale=scale, texture=mat, collider='box')
                w.texture_scale = ((scale[0]+scale[2])/2, scale[1])
            else:
                w = Entity(model='cube', position=pos, scale=scale, color=mat, collider='box')
            self.walls.append(w)

        loop_count = 1 if config.TEST_MODE else max(1, config.GEN_SIZE // config.NUM_PROCESSES)
        
        for i in range(loop_count):
            off = i * 150
            
            create_wall_entity((off, 0, 0), (100, 1, 100), 'grass')
            
            boundaries = [
                ((off, 15, 50), (100, 30, 1)), ((off, 15, -50), (100, 30, 1)),
                ((off + 50, 15, 0), (1, 30, 100)), ((off - 50, 15, 0), (1, 30, 100))
            ]
            for pos, scale in boundaries:
                create_wall_entity(pos, scale, color.gray)
                min_x, max_x = pos[0]-scale[0]/2, pos[0]+scale[0]/2
                min_z, max_z = pos[2]-scale[2]/2, pos[2]+scale[2]/2
                self.wall_data.append((min_x, max_x, min_z, max_z))

            placed_count = 0
            for _ in range(self.current_wall_count):
                placed = False; attempts = 0
                while not placed and attempts < 20:
                    attempts += 1
                    
                    wall_type = random.random()
                    
                    if wall_type < 0.60: 
                        w_width = random.randint(4, 8)
                        w_length = random.randint(4, 8)
                        w_height = random.randint(4, 7)
                    elif wall_type < 0.85: 
                        if random.random() < 0.5: 
                            w_width = random.randint(15, 25); w_length = 2
                        else:
                            w_width = 2; w_length = random.randint(15, 25)
                        w_height = random.randint(4, 6)
                    else: 
                        w_width = random.randint(2, 4); w_length = random.randint(2, 4)
                        w_height = random.randint(10, 20)

                    wx = random.randint(-45, 45) + off
                    wz = random.randint(-45, 45)
                    
                    min_x, max_x = wx - w_width/2, wx + w_width/2
                    min_z, max_z = wz - w_length/2, wz + w_length/2
                    
                    rel_x = wx - off
                    player_safe = (-4 < rel_x < 4) and (-4 < wz < 4)

                    if not player_safe and not is_overlapping((min_x, max_x, min_z, max_z), padding=2.5):
                        create_wall_entity((wx, w_height/2, wz), (w_width, w_height, w_length), 'brick')
                        self.wall_data.append((min_x, max_x, min_z, max_z))
                        placed = True
                        placed_count += 1

    def get_safe_spawn(self, offset_x):
        for _ in range(2000):
            x = offset_x + random.uniform(-40, 40)
            z = random.uniform(-40, 40)
            
            p_size = 2.5
            min_x, max_x = x - p_size, x + p_size
            min_z, max_z = z - p_size, z + p_size
            
            valid = True
            for (wx1, wx2, wz1, wz2) in self.wall_data:
                if (min_x < wx2 and max_x > wx1 and min_z < wz2 and max_z > wz1):
                    valid = False
                    break
            
            if valid:
                return x, z
                
        return offset_x, 0

    def update(self):
        if not self.active: return
        if config.TEST_MODE:
            if self.time_scale > 0: self.step_simulation(min(time.dt * self.time_scale, 0.1))
        else:
            step_dt = 0.04; steps = int(self.time_scale)
            for _ in range(steps): self.step_simulation(step_dt)
        self.ui_timer.text = f"{self.time_elapsed:.1f}s / {config.MATCH_DURATION}s"

        if config.TEST_MODE and self.agents:
            tagger, runner = self.agents[0]
            target = tagger if self.spectate_tagger else runner
            other = runner if self.spectate_tagger else tagger
            if not target.active: 
                pass
            
            target.visible = True; other.visible = True; target_pos = target.position
            for w in self.faded_walls: w.alpha = 1.0
            self.faded_walls.clear()
            cam_to_target = target_pos - camera.position; dist = cam_to_target.length()
            for w in self.walls:
                if distance(camera.position, w.position) < 4: w.alpha = 0.2; self.faded_walls.append(w)
            if dist > 1:
                hits = raycast(camera.position, cam_to_target.normalized(), distance=dist-1, ignore=(tagger, runner))
                if hits.hit: hits.entity.alpha = 0.3; self.faded_walls.append(hits.entity)
            if self.free_look:
                self.cam_yaw += mouse.velocity[0] * 100; yaw_rad = math.radians(self.cam_yaw); pitch_rad = math.radians(20)
                h_dist = self.cam_dist * math.cos(pitch_rad); v_dist = self.cam_dist * math.sin(pitch_rad)
                x_off = math.sin(yaw_rad) * h_dist; z_off = math.cos(yaw_rad) * h_dist
                self.smooth_focus = lerp(self.smooth_focus, target_pos, time.dt * 10)
                camera.position = self.smooth_focus + Vec3(x_off, v_dist, z_off)
                camera.look_at(self.smooth_focus + Vec3(0, 1, 0)); camera.rotation_z = 0
            else:
                current_reversing = hasattr(target, 'decision') and target.decision[1] < -0.1
                if current_reversing != self.cam_is_reversed:
                    self.cam_reverse_timer += time.dt
                    if self.cam_reverse_timer > 0.5: self.cam_is_reversed = current_reversing; self.cam_reverse_timer = 0
                else: self.cam_reverse_timer = 0
                offset_dir = target.forward * 22 if self.cam_is_reversed else target.forward * -22
                desired_pos = target_pos + offset_dir + Vec3(0, 10, 0)
                camera.position = lerp(camera.position, desired_pos, time.dt * 6)
                camera.look_at(target_pos + Vec3(0, 2, 0)); camera.rotation_z = 0; self.smooth_focus = target_pos

    def step_simulation(self, dt):
        self.time_elapsed += dt
        active_count = 0
        for i, (tagger, runner) in enumerate(self.agents):
            if not tagger.active: continue 
            active_count += 1
            tagger.act(runner, dt); runner.act(tagger, dt)
            dist = distance(tagger.position, runner.position)
            
            if tagger.stuck_timer > 2.0: tagger.change_score(-20 * dt, "stuck_penalty")
            if runner.stuck_timer > 2.0: runner.change_score(-20 * dt, "stuck_penalty")
            
            if dist < 6.0: runner.change_score(-10 * dt, "proximity_penalty")
            if dist < 1.3: self.finish_pair(i, winner="tagger")
            
        if self.time_elapsed > config.MATCH_DURATION:
            for i, (tagger, runner) in enumerate(self.agents):
                if tagger.active: self.finish_pair(i, winner="runner")
        if active_count == 0: 
            if config.TEST_MODE: self.start_generation()
            else: self.evolve()

    def finish_pair(self, index, winner):
        t, r = self.agents[index]
        start_dist = 20.0
        
        dist_improv = max(0, start_dist - t.min_dist_to_target) * 10
        sight_bonus = min(t.time_in_sight * 20, 300)
        
        t.change_score(dist_improv, "dist_improv")
        t.change_score(sight_bonus, "sight_bonus")
        
        if winner == "tagger":
            time_bonus = (config.MATCH_DURATION - self.time_elapsed) * 20
            t.change_score(500, "win_bonus")
            t.change_score(time_bonus, "time_bonus")
            
            r.change_score(self.time_elapsed * 10, "survival_time")
            if config.TEST_MODE: print("TAGGER WON!")
        else:
            r.change_score(500, "win_bonus")
            dist_bonus = distance(t.position, r.position) * 5
            r.change_score(dist_bonus, "escape_dist_bonus")
            if config.TEST_MODE: print("RUNNER ESCAPED!")
        
        self.scores[index] = (t.fitness_score, r.fitness_score)
        
        self.detailed_scores_cache[index] = {
            "pair_id": index,
            "t_score": t.fitness_score,
            "t_breakdown": t.score_breakdown.copy(),
            "r_score": r.fitness_score,
            "r_breakdown": r.score_breakdown.copy()
        }
        
        t.active = False; r.active = False
        t.collider = None; r.collider = None
    
    def evolve(self):
        self.log_manager.log_status("EVOLVING")
        gen_time = time.time() - self.gen_start_time
        
        total_brains = len(self.population) * 2
        sum_jump_b = 0; sum_jump_w = 0
        sum_strafe_b = 0; sum_strafe_w = 0
        sum_move_b = 0; sum_move_w = 0
        
        for (t_brain, r_brain) in self.population:
            for b in [t_brain, r_brain]:
                sum_strafe_b += b.b2[0]
                sum_move_b += b.b2[1]
                sum_jump_b += b.b2[2]
                
                sum_strafe_w += np.mean(b.w2[:, 0])
                sum_move_w += np.mean(b.w2[:, 1])
                sum_jump_w += np.mean(b.w2[:, 2])

        nn_stats = {
            "jump_b": sum_jump_b / total_brains, "jump_w": sum_jump_w / total_brains,
            "strafe_b": sum_strafe_b / total_brains, "strafe_w": sum_strafe_w / total_brains,
            "move_b": sum_move_b / total_brains, "move_w": sum_move_w / total_brains
        }
        
        self.log_manager.log_generation_end(
            self.generation, gen_time, self.scores, 
            self.bad_gen_count, self.force_reset_next, nn_stats
        )
        
        if self.generation % 5 == 0:
            valid_details = [d for d in self.detailed_scores_cache if d is not None]
            self.log_manager.log_detailed_batch(self.generation, valid_details)

        current_max_score = max([s[0] for s in self.scores] + [s[1] for s in self.scores])
        if current_max_score <= self.best_historical_score + 10: self.stagnant_gens += 1
        else: self.best_historical_score = current_max_score; self.stagnant_gens = 0
            
        mut_mult = 1.0; str_mult = 1.0
        
        is_extinction_event = self.stagnant_gens > config.EXTINCTION_LIMIT
        if is_extinction_event:
            self.stagnant_gens = 0 

        elif self.stagnant_gens > config.STAGNATION_LIMIT:
            mut_mult = 3.0; str_mult = 2.0 
        
        sorted_taggers = sorted(zip(self.population, self.scores), key=lambda x: x[1][0], reverse=True)
        sorted_runners = sorted(zip(self.population, self.scores), key=lambda x: x[1][1], reverse=True)
        
        best_t, best_r = sorted_taggers[0][0][0], sorted_runners[0][0][1]
        self.save_brains(best_t, best_r)
        
        turn_bias = best_r.b2[3] 
        if abs(turn_bias) > config.ANTI_SPIN_THRESHOLD: self.bad_gen_count += 1
        else: self.bad_gen_count = 0
        if self.bad_gen_count >= config.MAX_BAD_GENS: self.force_reset_next = True

        if self.stop_requested: application.quit(); return

        new_pop = []
        
        local_gen_size = max(1, config.GEN_SIZE // config.NUM_PROCESSES)
        
        keep_count = 1 if is_extinction_event else max(1, int(local_gen_size * 0.1))
        
        for i in range(keep_count):
            if i < len(sorted_taggers):
                new_pop.append((sorted_taggers[i][0][0].clone(), sorted_runners[i][0][1].clone()))

        input_size = 14
        
        def tournament(sorted_list, is_tagger):
            max_idx = len(sorted_list) - 1
            idx1 = random.randint(0, max_idx); idx2 = random.randint(0, max_idx)
            brain_idx = 0 if is_tagger else 1
            if sorted_list[idx1][1][brain_idx] > sorted_list[idx2][1][brain_idx]: return sorted_list[idx1][0][brain_idx].clone()
            else: return sorted_list[idx2][0][brain_idx].clone()

        while len(new_pop) < local_gen_size:
            if is_extinction_event:
                new_pop.append((SimpleBrain(input_size, config.HIDDEN_LAYER_SIZE, 4), SimpleBrain(input_size, config.HIDDEN_LAYER_SIZE, 4)))
            else:
                p_t = tournament(sorted_taggers, True); p_r = tournament(sorted_runners, False)
                p_t.mutate(mut_mult * self.current_mutation_rate, str_mult) 
                p_r.mutate(mut_mult * self.current_mutation_rate, str_mult)
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
        data = {"gen":self.generation, "t_brain":t, "r_brain":r, "best_score": self.best_historical_score}
        try:
            with open(config.SAVE_FILE, 'wb') as f: pickle.dump(data, f)
            if self.generation % config.BACKUP_INTERVAL == 0:
                with open(f"backup_gen_{self.generation}.pkl", 'wb') as f: pickle.dump(data, f)
            self.log_manager.log_status(f"Saved Best Brains (Gen {self.generation})")
        except Exception as e: self.log_manager.log_error(f"Failed to save brains: {e}")

    def load_brains(self):
        if os.path.exists(config.LOAD_FILE): 
            try:
                with open(config.LOAD_FILE, 'rb') as f:
                    data = pickle.load(f)
                    if data['t_brain'].hidden_size != config.HIDDEN_LAYER_SIZE: return 
                    self.generation = data['gen'] + 1
                    self.best_historical_score = data.get('best_score', 0) 
                    t, r = data['t_brain'], data['r_brain']
                    
                    local_gen_size = max(1, config.GEN_SIZE // config.NUM_PROCESSES)
                    
                    self.population = []
                    for _ in range(local_gen_size):
                        tc, rc = t.clone(), r.clone()
                        tc.mutate(self.current_mutation_rate) 
                        rc.mutate(self.current_mutation_rate)
                        self.population.append((tc, rc))
                    self.population[0] = (t, r)
                    self.log_manager.log_status(f"Loaded Gen {self.generation-1} from {config.LOAD_FILE}")
            except Exception as e: self.log_manager.log_error(f"Load failed: {e}"); self.generation = 1
        else: self.log_manager.log_status(f"Could not find {config.LOAD_FILE}, starting fresh.")