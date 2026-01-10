from ursina import *
import sys
import os
import config

worker_id = 0
win_x, win_y, win_w, win_h = 0, 0, 800, 600

for arg in sys.argv:
    if arg.startswith("id="): worker_id = int(arg.split("=")[1])
    if arg.startswith("win_x="): win_x = int(arg.split("=")[1])
    if arg.startswith("win_y="): win_y = int(arg.split("=")[1])
    if arg.startswith("win_w="): win_w = int(arg.split("=")[1])
    if arg.startswith("win_h="): win_h = int(arg.split("=")[1])

if worker_id > 0:
    window.title = f"Worker {worker_id}"
    window.position = (win_x, win_y)
    window.forced_aspect_ratio = None
    window.size = (win_w, win_h)
    window.vsync = False
else:
    window.title = "Spectator"
    window.forced_aspect_ratio = None

if worker_id > 0:
    if not os.path.exists(config.LOG_DIR): os.makedirs(config.LOG_DIR)
    if not os.path.exists(config.MODEL_DIR): os.makedirs(config.MODEL_DIR)
    
    config.LOG_FILE = os.path.join(config.LOG_DIR, f"worker_{worker_id}_log.txt")
    config.ERROR_LOG_FILE = os.path.join(config.LOG_DIR, f"worker_{worker_id}_attention_error.txt")
    config.SAVE_FILE = os.path.join(config.MODEL_DIR, f"worker_{worker_id}_model.pkl")
    config.LOAD_FILE = config.SAVE_FILE 
else:
    config.TEST_MODE = True 
    global_model_path = os.path.join(config.BASE_DIR, "best_global_model.pkl")
    if os.path.exists(global_model_path):
        config.LOAD_FILE = global_model_path

app = Ursina(borderless=False, vsync=True if config.TEST_MODE else False)

if hasattr(config, 'FULLSCREEN') and config.FULLSCREEN:
    window.fullscreen = True
else:
    window.position = (win_x, win_y)
    window.size = (win_w, win_h)

window.color = color.black

if config.TEST_MODE or (win_w > 400 and not config.HEADLESS):
    camera.ui.enabled = True 
else:
    camera.ui.enabled = False

camera.enabled = True if config.TEST_MODE else False   

from training_manager import TrainingManager
from log_manager import LogManager

log_manager = LogManager(config.LOG_FILE, config.ERROR_LOG_FILE)
trainer = TrainingManager(log_manager)

def req_stop():
    trainer.stop_requested = True
    b_stop.text = "STOP"
    b_stop.color = color.red

if config.TEST_MODE:
    b_stop = Button(text="Stop", color=color.orange, scale=(0.2, 0.05), position=(-0.7, -0.4), on_click=req_stop)

app.run()