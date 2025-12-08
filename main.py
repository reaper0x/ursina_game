from ursina import *
import requests # The library that talks to the internet
import threading # To prevent the game from freezing while loading

app = Ursina()

# ==========================================
# 🌐 NETWORK CONFIGURATION
# ==========================================
# REPLACE THIS WITH YOUR REAL PYTHONANYWHERE URL!
SERVER_URL = "https://bloodyvamp1re.pythonanywhere.com" 
CLIENT_VERSION = "1.2.0" 

# ==========================================
# 🖥️ UI & LOGIC
# ==========================================

def start_game_logic():
    login_panel.disable()
    ground.enabled = True
    player.enabled = True
    ambient_light.enabled = True

def network_login(username, password):
    """Runs in a background thread to check login so the game doesn't freeze"""
    try:
        response = requests.post(
            f"{SERVER_URL}/login", 
            json={"username": username, "password": password},
            timeout=5
        )
        
        if response.status_code == 200:
            # We must update UI on the main thread
            invoke(lambda: setattr(status_text, 'text', "Success!"), delay=0)
            invoke(lambda: setattr(status_text, 'color', color.green), delay=0)
            invoke(start_game_logic, delay=1)
        else:
            invoke(lambda: setattr(status_text, 'text', "Wrong Password"), delay=0)
            invoke(lambda: setattr(status_text, 'color', color.red), delay=0)
            invoke(lambda: login_button.enable(), delay=0)
            
    except:
        invoke(lambda: setattr(status_text, 'text', "Server Offline"), delay=0)
        invoke(lambda: setattr(status_text, 'color', color.red), delay=0)
        invoke(lambda: login_button.enable(), delay=0)

def on_login_click():
    status_text.text = "Connecting..."
    status_text.color = color.yellow
    login_button.disable() # Prevent double clicking
    
    # Run the network request in a separate thread
    t = threading.Thread(target=network_login, args=(username_field.text, password_field.text))
    t.start()

# --- UI Setup ---
update_panel = Entity(parent=camera.ui, enabled=False)
Text(parent=update_panel, text="⚠ UPDATE REQUIRED", origin=(0,0), scale=2, position=(0, 0.2), color=color.red)
Text(parent=update_panel, text="Please download the latest patch.", origin=(0,0), position=(0, 0))

login_panel = Entity(parent=camera.ui, enabled=False)
bg = Entity(parent=login_panel, model='quad', scale=(10, 10), color=color.dark_gray, z=1)
Text(parent=login_panel, text="SERVER LOGIN", origin=(0,0), scale=2, position=(0, 0.35))
username_field = InputField(parent=login_panel, default_value='', position=(0, 0.1), placeholder='Username')
password_field = InputField(parent=login_panel, default_value='', position=(0, -0.05), placeholder='Password')
password_field.text_color = color.clear
Text(parent=password_field, text="*****", position=(-.45, 0))

login_button = Button(parent=login_panel, text='Login', color=color.azure, scale=(.25, .05), position=(0, -0.2))
login_button.on_click = on_login_click
status_text = Text(parent=login_panel, text='', origin=(0,0), position=(0, -0.3))

# --- Game World (Same as before) ---
ground = Entity(model='plane', texture='grass', scale=(50, 1, 50), collider='box', enabled=False)
player = Entity(model='cube', texture='brick', color=color.white, scale=1, position=(0, 0.5, 0), enabled=False)
ambient_light = AmbientLight(color=color.rgb(100, 100, 100))
ambient_light.enabled = False

def update():
    if player.enabled:
        speed = 6 * time.dt
        if held_keys['w']: player.z += speed
        if held_keys['s']: player.z -= speed
        if held_keys['a']: player.x -= speed
        if held_keys['d']: player.x += speed
        camera.position = lerp(camera.position, player.position + (0, 10, -15), time.dt * 2)
        camera.look_at(player)

# ==========================================
# 🚀 INITIAL VERSION CHECK
# ==========================================
def check_for_updates():
    try:
        r = requests.get(f"{SERVER_URL}/check_version", timeout=3)
        server_version = r.json().get("version")
        
        if server_version == CLIENT_VERSION:
            login_panel.enabled = True
        else:
            update_panel.enabled = True
    except:
        # If internet is down or server is offline
        login_panel.enabled = True
        status_text.text = "Offline Mode (Server Unreachable)"

# Run the version check immediately
check_for_updates()

app.run()