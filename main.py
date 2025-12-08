from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import requests
import threading
import time
import random

app = Ursina()

# --- CONFIG ---
SERVER_URL = "https://bloodyvamp1re.pythonanywhere.com"
username = ""
current_room = ""
my_pid = "" 
game_active = False
force_tag_signal = False 
my_role_var = "" 

# --- UI ELEMENTS ---
bg = Entity(parent=camera.ui, model='quad', scale=(2,2), color=color.black, z=1)

login_panel = WindowPanel(
    title='Login',
    content=(
        InputField(name='user', placeholder='Username'),
        Button(text='Play as Guest', color=color.azure, on_click=lambda: play_guest()),
    ),
    popup=False, enabled=True
)

lobby_panel = WindowPanel(
    title='Lobby',
    content=(
        InputField(name='code', placeholder='Room Code (e.g. 111)'),
        Button(text='Join Room', color=color.orange, on_click=lambda: join_room()),
        Text(text='Waiting...', enabled=False)
    ),
    popup=False, enabled=False
)

hud_role = Text(text='', position=(-0.85, 0.45), scale=2, enabled=False)
hud_time = Text(text='', position=(0, 0.45), origin=(0,0), scale=2, enabled=False)
win_text = Text(text='', origin=(0,0), scale=3, color=color.yellow, enabled=False)
debug_txt = Text(text='Dist: -', position=(0.7, -0.4), scale=1.5, color=color.green, enabled=False)

# --- GAME OBJECTS ---
ground = Entity(model='plane', scale=(100,1,100), texture='grass', collider='box', enabled=False)
walls = []
player = None
opponent = Entity(model='cube', color=color.white, scale=(1,2,1), y=1, enabled=False)

session = requests.Session()

# --- FX ---
def create_explosion(position, particle_color):
    Audio('assets/hit.wav', autoplay=False) 
    for i in range(20):
        e = Entity(model='cube', color=particle_color, position=position, scale=0.4)
        e.animate_position(e.position + Vec3(random.uniform(-3,3), random.uniform(2,5), random.uniform(-3,3)), duration=1)
        e.animate_rotation((random.randint(0,360), random.randint(0,360),0), duration=1)
        e.fade_out(duration=1)
        destroy(e, delay=1.1)

# --- FUNCTIONS ---
def play_guest():
    global username
    username = f"Guest{random.randint(100,999)}"
    login_panel.enabled = False
    lobby_panel.enabled = True

def join_room():
    global current_room, my_pid
    code = lobby_panel.content[0].text
    if not code: return
    
    lobby_panel.content[2].text = "Connecting..."
    lobby_panel.content[2].enabled = True
    
    try:
        r = session.post(f"{SERVER_URL}/join_room", json={"username":username, "room_code":code})
        data = r.json()
        if data.get("status") != "error":
            current_room = code
            my_pid = data['pid']
            start_game(data['seed'], data['status'] == 'start')
            threading.Thread(target=network_loop, daemon=True).start()
        else:
            lobby_panel.content[2].text = "Error or Full"
    except:
        lobby_panel.content[2].text = "Server Error"

def start_game(seed, ready):
    global player
    lobby_panel.enabled = False
    bg.enabled = False
    ground.enabled = True
    generate_walls(seed)
    
    if player: destroy(player)
    player = FirstPersonController(y=2, speed=14)
    mouse.locked = True
    
    if my_pid == "p1": player.position = (-10, 2, -10)
    else: player.position = (10, 2, 10)

    opponent.enabled = True
    opponent.visible = True
    hud_role.enabled = True
    hud_time.enabled = True
    debug_txt.enabled = True
    
    if not ready: hud_role.text = "Waiting..."

def generate_walls(seed):
    random.seed(seed)
    for w in walls: destroy(w)
    walls.clear()
    for x in range(-20, 20):
        for z in range(-20, 20):
            if x == -20 or x == 19 or z == -20 or z == 19:
                walls.append(Entity(model='cube', position=(x*2,2,z*2), scale=(2,6,2), texture='brick', collider='box'))
            elif random.random() < 0.1 and abs(x)>5:
                walls.append(Entity(model='cube', position=(x*2,2,z*2), scale=(2,6,2), texture='brick', collider='box'))

def reset_to_lobby():
    global game_active, current_room
    game_active = False
    current_room = ""
    mouse.locked = False
    bg.enabled = True
    lobby_panel.enabled = True
    
    if player: destroy(player)
    ground.enabled = False
    opponent.enabled = False
    hud_role.enabled = False
    win_text.enabled = False
    debug_txt.enabled = False
    for w in walls: destroy(w)

def update():
    global force_tag_signal
    if not game_active or not player: return

    # Distance Check
    dist = distance(player.position, opponent.position)
    debug_txt.text = f"Dist: {dist:.2f}"
    
    # Auto-Tag Logic
    if my_role_var == "tagger":
        if dist < 4.0:
            debug_txt.color = color.red
            force_tag_signal = True 
        else:
            debug_txt.color = color.green
    
    # NEW: Runner Logic - If I run into Tagger, I lose too
    elif my_role_var == "runner":
         if dist < 2.5: # Slightly smaller hitbox for runner suicide
             force_tag_signal = True # Tell server I died

def network_loop():
    global game_active, force_tag_signal, my_role_var
    
    while current_room:
        try:
            pos = (0,0,0)
            if player: pos = (player.x, player.y, player.z)
            
            payload = {
                "code": current_room, "pid": my_pid,
                "x": pos[0], "y": pos[1], "z": pos[2],
                "tagged": force_tag_signal
            }
            
            r = session.post(f"{SERVER_URL}/update", json=payload, timeout=0.5)
            
            if r.status_code == 200:
                data = r.json()
                
                if force_tag_signal: force_tag_signal = False
                
                if opponent and opponent.enabled:
                    opponent.x = lerp(opponent.x, data['opp_x'], 0.5)
                    opponent.y = lerp(opponent.y, data['opp_y'], 0.5)
                    opponent.z = lerp(opponent.z, data['opp_z'], 0.5)

                if data['status'] == "PLAYING":
                    game_active = True
                    hud_time.text = str(data['time'])
                    
                    my_role_var = data['role'] 
                    hud_role.text = f"You are: {my_role_var.upper()}"
                    
                    opp_role = data['opp_role']
                    hud_role.color = color.red if my_role_var == "tagger" else color.azure
                    opponent.color = color.red if opp_role == "tagger" else color.azure

                elif data['status'] == "GAME_OVER":
                    if game_active: 
                        # --- VISUAL FIX HERE ---
                        if my_role_var == "tagger":
                            # I am Tagger. My Opponent (Runner) explodes.
                            create_explosion(opponent.position, opponent.color)
                            opponent.enabled = False 
                        else:
                            # I am Runner. I explode. Opponent (Tagger) stays visible.
                            create_explosion(player.position, color.azure)
                            # We do NOT hide opponent. Tagger stands victorious.
                    
                    game_active = False
                    winner = data['winner']
                    win_text.text = f"{winner} WINS!"
                    win_text.enabled = True
                    time.sleep(4)
                    invoke(reset_to_lobby)
                    break 
        except: pass
        time.sleep(0.05)

app.run()