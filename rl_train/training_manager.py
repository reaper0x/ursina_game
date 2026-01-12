from ursina import *
import random
import os
import time
import torch
import numpy as np
from brain import SimpleBrain
from agent import Agent
import config

class TrainingManager(Entity):
    def __init__(self, log_manager):
        super().__init__()
        self.log_manager = log_manager
        self.generation = 1
        self.agents = []
        self.time_elapsed = 0
        self.active = False
        self.stop_requested = False
        self.time_scale = 1.0
        
        self.global_tagger_brain = SimpleBrain(config.INPUT_SIZE, config.HIDDEN_LAYER_SIZE, config.OUTPUT_SIZE)
        self.global_runner_brain = SimpleBrain(config.INPUT_SIZE, config.HIDDEN_LAYER_SIZE, config.OUTPUT_SIZE)
        
        self.start_generation()

    def start_generation(self):
        self.cleanup()
        self.spawn_walls(seed=self.generation)
        
        tx, tz = self.get_safe_spawn(0)
        rx, rz = self.get_safe_spawn(0)
        
        tagger = Agent("tagger", 0, 0, self, brain=self.global_tagger_brain, position=(tx, 2, tz))
        runner = Agent("runner", 0, 0, self, brain=self.global_runner_brain, position=(rx, 2, rz))
        
        self.agents.append((tagger, runner))
        
        self.time_elapsed = 0
        self.active = True
        self.gen_start_time = time.time()
        print(f"Gen {self.generation} started.")

    def spawn_walls(self, seed):
        self.walls = []
        self.wall_data = []
        random.seed(seed)
        
        create_wall_entity = lambda pos, scale: self.walls.append(Entity(model='cube', position=pos, scale=scale, color=color.gray, collider='box'))
        
        create_wall_entity((0, -1, 0), (100, 1, 100))
        
        for _ in range(config.STARTING_WALLS):
            wx, wz = random.randint(-40, 40), random.randint(-40, 40)
            create_wall_entity((wx, 2, wz), (random.randint(4,8), random.randint(4,8), random.randint(4,8)))
            self.wall_data.append((wx-4, wx+4, wz-4, wz+4))

    def get_safe_spawn(self, offset_x):
        for _ in range(100):
            x, z = random.uniform(-30, 30), random.uniform(-30, 30)
            if not any(bx < x < tx and bz < z < tz for bx, tx, bz, tz in self.wall_data):
                return x, z
        return 0, 0

    def update(self):
        if not self.active: return
        
        dt = 0.05
        steps = int(self.time_scale) if not config.HEADLESS else 1
        
        for _ in range(steps):
            self.time_elapsed += dt
            for t, r in self.agents:
                t.act(r, dt)
                r.act(t, dt)
                
                dist = distance(t.position, r.position)
                if dist < 1.3:
                    t.change_score(100, "catch")
                    r.change_score(-100, "caught")
                    self.finish_episode()
                    return

            if self.time_elapsed > config.MATCH_DURATION:
                self.finish_episode()
                return

    def finish_episode(self):
        self.active = False
        t, r = self.agents[0]
        
        self.update_policy(self.global_tagger_brain, t)
        self.update_policy(self.global_runner_brain, r)
        
        print(f"Gen {self.generation} finished. Tagger: {t.fitness_score:.1f}, Runner: {r.fitness_score:.1f}")
        
        if self.generation % 10 == 0:
            torch.save(self.global_tagger_brain.state_dict(), os.path.join(config.MODEL_DIR, f"tagger_{self.generation}.pth"))
            
        self.generation += 1
        self.start_generation()

    def update_policy(self, brain, agent):
        R = 0
        returns = []
        for r in agent.rewards[::-1]:
            R = r + config.GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(agent.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        brain.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        brain.optimizer.step()

    def cleanup(self):
        for t, r in self.agents:
            destroy(t)
            destroy(r)
        self.agents.clear()
        if hasattr(self, 'walls'):
            for w in self.walls: destroy(w)