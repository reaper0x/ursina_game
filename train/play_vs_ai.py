from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import pickle
import os
import numpy as np
import random
import sys
import config
from brain import SimpleBrain
from agent import Agent

config.TEST_MODE = True
config.HEADLESS = False

app = Ursina(borderless=False, vsync=True)
window.color = color.black
window.title = "Human vs AI"

model_path = "best_global_model.pkl"
if not os.path.exists(model_path):
    print(f"Error: Could not find model file '{model_path}'")
    sys.exit()

print(f"Loading AI from {model_path}...")
with open(model_path, 'rb') as f:
    data = pickle.load(f)
    t_brain = data.get('t_brain')
    r_brain = data.get('r_brain') 
    
    if r_brain is None:
        print("Warning: r_brain not found in model, using t_brain for both roles.")
        r_brain = t_brain

    print(f"Loaded Gen {data.get('gen', '?')} Brains. Best Score: {data.get('best_score', 0)}")

ground = Entity(model='plane', scale=(200, 1, 200), color=color.dark_gray, texture='white_cube', texture_scale=(100,100), collider='box')

walls = []

def create_wall(pos, scale, mat):
    w = Entity(model='cube', position=pos, scale=scale, color=mat, collider='box', texture='brick', texture_scale=((scale[0]+scale[2])/2, scale[1]))
    walls.append(w)

boundaries = [
    ((0, 10, 50), (100, 20, 1)), ((0, 10, -50), (100, 20, 1)),
    ((50, 10, 0), (1, 20, 100)), ((-50, 10, 0), (1, 20, 100))
]
for pos, scale in boundaries:
    create_wall(pos, scale, color.gray)

random.seed(123) 
for _ in range(50):
    wx = random.randint(-45, 45)
    wz = random.randint(-45, 45)
    if -5 < wx < 5 and -5 < wz < 5: continue
    w_width = random.randint(4, 8)
    w_length = random.randint(4, 8)
    w_height = random.randint(4, 8)
    create_wall((wx, w_height/2, wz), (w_width, w_height, w_length), color.light_gray)

player = FirstPersonController(model='cube', color=color.azure, origin_y=-.5)
player.position = (0, 2, 0)
player.speed = 12
player.jump_height = 2
player.cursor.visible = False
player.gravity = 0.8 

player.bhop_chain = 0
player.ground_time = 0.0
player.base_speed = 12

class DummyManager:
    def __init__(self): pass

ai_agent = Agent(
    role="tagger", 
    pair_id=0, 
    origin_x=10, 
    manager=DummyManager(), 
    brain=t_brain,
    position=(10, 2, 10)
)
ai_agent.active = True

timer_text = Text(text="Wait...", position=(0, 0.4), scale=2, origin=(0,0))
status_text = Text(text="", position=(0, 0), scale=3, origin=(0,0), color=color.red, enabled=False)
dist_text = Text(text="Distance: 0m", position=(-0.85, 0.45), scale=1)
role_info = Text(text="", position=(-0.85, 0.48), scale=1.2, color=color.azure)
restart_info = Text(text="Press [R] to Restart", position=(-0.85, 0.40), scale=1)

game_active = True
start_time = time.time()
human_role = "runner" 

def reset_game():
    global game_active, start_time, human_role
    
    if random.random() < 0.5:
        human_role = "runner"
        
        ai_agent.role = "tagger"
        ai_agent.brain = t_brain
        ai_agent.color = color.red
        
        player.color = color.azure
        role_info.text = "YOU ARE: RUNNER"
        role_info.color = color.azure
        timer_text.text = "Survive!"
    else:
        human_role = "tagger"
        
        ai_agent.role = "runner"
        ai_agent.brain = r_brain
        ai_agent.color = color.azure
        
        player.color = color.red
        role_info.text = "YOU ARE: TAGGER"
        role_info.color = color.red
        timer_text.text = "Catch Him!"

    player.position = (0, 2, 0)
    ai_agent.position = (10, 2, 10)
    
    ai_agent.velocity_x = 0
    ai_agent.velocity_y = 0
    ai_agent.velocity_z = 0
    ai_agent.active = True
    
    player.speed = player.base_speed
    player.bhop_chain = 0
    
    status_text.enabled = False
    game_active = True
    start_time = time.time()
    print(f"Game Restarted. Human: {human_role}, AI: {ai_agent.role}")

def input(key):
    if key == 'space':
        if player.grounded:
            if player.ground_time < config.BHOP_WINDOW:
                player.bhop_chain = min(3, player.bhop_chain + 1)
                player.speed = player.base_speed + (player.bhop_chain * config.BHOP_SPEED_BOOST)
            else:
                player.bhop_chain = 0
                player.speed = player.base_speed
            
            player.ground_time = 0

def update():
    global game_active
    
    if held_keys['escape']: application.quit()
    
    if held_keys['r']:
        reset_game()

    if not game_active: return

    if player.grounded:
        player.ground_time += time.dt
        if player.ground_time > 0.2:
            player.bhop_chain = 0
            player.speed = player.base_speed
    else:
        player.ground_time = 0

    ai_agent.act(player, time.dt)
    
    if ai_agent.y < -10:
        ai_agent.position = (0, 5, 0)
        ai_agent.velocity_y = 0

    dist = distance(player.position, ai_agent.position)
    dist_text.text = f"Distance: {dist:.1f}m"

    elapsed = time.time() - start_time
    timer_text.text = f"Time: {elapsed:.1f}s"

    if dist < 1.5:
        game_active = False
        player.speed = 0
        ai_agent.active = False
        status_text.enabled = True
        
        if human_role == "runner":
            status_text.text = "CAUGHT!"
            status_text.color = color.red
            print("Game Over: AI Caught Human")
        else:
            status_text.text = "GOT HIM!"
            status_text.color = color.green
            print("Game Over: Human Caught AI")

    if elapsed > 60:
        game_active = False
        player.speed = 0
        ai_agent.active = False
        status_text.enabled = True
        
        if human_role == "runner":
            status_text.text = "YOU SURVIVED!"
            status_text.color = color.green
            print("Game Over: Human Survived")
        else:
            status_text.text = "AI ESCAPED!"
            status_text.color = color.red
            print("Game Over: AI Escaped")

reset_game()

app.run()