import subprocess
import time
import os
import sys
import math
import tkinter as tk
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
MERGER_SCRIPT = os.path.join(BASE_DIR, "merger.py")

def get_screen_res():
    try:
        root = tk.Tk()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except:
        return 1920, 1080

def launch():
    if not os.path.exists(MAIN_SCRIPT) or not os.path.exists(MERGER_SCRIPT):
        print(f"ERROR: Cannot find main.py or merger.py in {BASE_DIR}")
        return

    total_workers = config.NUM_PROCESSES
    visible_limit = config.MAX_VISIBLE_WINDOWS
    
    screen_w, screen_h = get_screen_res()
    
    grid_count = min(total_workers, visible_limit)
    cols = int(math.ceil(math.sqrt(grid_count)))
    rows = int(math.ceil(grid_count / cols)) if cols > 0 else 1
    
    print(f"Launching {total_workers} processes ({grid_count} visible)...")
    
    if config.AUTO_LAYOUT and grid_count > 0:
        win_w = (screen_w // cols) - 10
        win_h = (screen_h // rows) - 40
    else:
        win_w = config.WINDOW_WIDTH
        win_h = int(win_w * (9/16))

    processes = []
    
    mp = subprocess.Popen([sys.executable, MERGER_SCRIPT])
    processes.append(mp)
    print("Merger process started.")

    for i in range(total_workers):
        worker_id = i + 1
        
        is_visible = (i < visible_limit) and (not config.HEADLESS)
        
        if is_visible:
            r = i // cols
            c = i % cols
            x = c * (win_w + 5)
            y = r * (win_h + 30)
        else:
            x, y = 0, 0 
            
        cmd = [
            sys.executable, MAIN_SCRIPT, 
            f"id={worker_id}", 
            f"win_x={x}", 
            f"win_y={y}", 
            f"win_w={win_w}", 
            f"win_h={win_h}"
        ]
        
        startupinfo = None
        if not is_visible:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0 
        
        p = subprocess.Popen(cmd, startupinfo=startupinfo)
        processes.append(p)
        
        time.sleep(2.0) 

    print(f"All {total_workers} workers running.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
            if any(p.poll() is not None for p in processes):
                pass 
    except KeyboardInterrupt:
        print("\nShutting down workers...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    launch()