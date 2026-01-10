import time
import pickle
import os
import shutil
import glob
import config

GLOBAL_FILE = os.path.join(config.BASE_DIR, "best_global_model.pkl")

def load_pkl(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def merge_logic():
    if not os.path.exists(config.MODEL_DIR): return

    files = glob.glob(os.path.join(config.MODEL_DIR, "worker_*_model.pkl"))
    if not files: return

    best_score = -999999
    best_file = None
    best_data = None
    
    scores = []

    for fpath in files:
        data = load_pkl(fpath)
        if data and 'best_score' in data:
            s = data['best_score']
            scores.append((fpath, s))
            if s > best_score:
                best_score = s
                best_file = fpath
                best_data = data

    if best_data:
        try:
            with open(GLOBAL_FILE, 'wb') as f:
                pickle.dump(best_data, f)
        except: pass

    if len(scores) > 2 and config.NUM_PROCESSES > 4:
        scores.sort(key=lambda x: x[1])
        
        num_to_replace = int(len(scores) * 0.2)
        
        for i in range(num_to_replace):
            worst_file = scores[i][0]
            worst_score = scores[i][1]
            
            if best_score > worst_score + 1000:
                print(f"[MERGER] Migrating Best ({best_score:.0f}) -> Worker {os.path.basename(worst_file)} ({worst_score:.0f})")
                try:
                    shutil.copyfile(best_file, worst_file)
                except: pass

if __name__ == "__main__":
    print(f"Merger Service Active (Managing {config.NUM_PROCESSES} workers)")
    while True:
        merge_logic()
        time.sleep(10)