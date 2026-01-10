import logging
import time
import os
import numpy as np
from config import MAX_BAD_GENS

class LogManager:
    def __init__(self, log_file, error_log_file):
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)
        self.log_file = log_file
        
        self.error_logger = logging.getLogger('ErrorLogger')
        self.error_logger.setLevel(logging.INFO)
        efh = logging.FileHandler(error_log_file)
        efh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.error_logger.addHandler(efh)
        
        self.detailed_log_dir = os.path.join(os.path.dirname(os.path.abspath(log_file)), "detailed_logs")
        if not os.path.exists(self.detailed_log_dir):
            os.makedirs(self.detailed_log_dir)

    def log_generation_start(self, gen_num, mutation_mult=1.0):
        mode_str = " (PANIC MUTATION)" if mutation_mult > 1.0 else ""
        msg = f"\n{'='*50}\nGEN {gen_num:04d} START{mode_str} ({time.strftime('%Y-%m-%d %H:%M:%S')})\n{'='*50}"
        self.logger.info(msg)
        if mutation_mult > 1.0:
            self.error_logger.info(f"GEN {gen_num} STARTED IN PANIC MODE (Mutation x{mutation_mult})")
        print(f"--- STARTING GEN {gen_num}{mode_str} ---")

    def log_generation_end(self, gen_num, time_taken, scores, bad_gen_count, force_reset, nn_stats=None):
        if not scores: return
        tagger_scores = [s[0] for s in scores]
        runner_scores = [s[1] for s in scores]
        
        best_t = max(tagger_scores)
        best_r = max(runner_scores)
        
        self.logger.info(f"Gen: {gen_num:04d} | Time: {time_taken:.2f}s")
        self.logger.info(f"Tagger: Best={best_t:.2f}, Avg={np.mean(tagger_scores):.2f}")
        self.logger.info(f"Runner: Best={best_r:.2f}, Avg={np.mean(runner_scores):.2f}")
        self.logger.info(f"Bad Gen Count: {bad_gen_count}/{MAX_BAD_GENS}")

        if nn_stats:
            self.logger.info("-" * 20)
            self.logger.info("AVG NEURAL NET STATS (Bias / Mean Weight):")
            self.logger.info(f"  Jump:   B={nn_stats['jump_b']:.4f} | W={nn_stats['jump_w']:.4f}")
            self.logger.info(f"  Strafe: B={nn_stats['strafe_b']:.4f} | W={nn_stats['strafe_w']:.4f}")
            self.logger.info(f"  Move:   B={nn_stats['move_b']:.4f} | W={nn_stats['move_w']:.4f}")

        print(f"--- GEN {gen_num} SUMMARY ---")
        print(f"🥇 Tagger Best: {best_t:.2f}")
        print(f"🥇 Runner Best: {best_r:.2f}")

    def log_detailed_batch(self, gen_num, detailed_data):
        filename = os.path.join(self.detailed_log_dir, f"gen_{gen_num}_detailed.txt")
        
        try:
            with open(filename, "w") as f:
                f.write(f"DETAILED REPORT FOR GENERATION {gen_num}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                for entry in detailed_data:
                    pid = entry['pair_id']
                    f.write(f"--- PAIR {pid} ---\n")
                    
                    f.write(f"  [TAGGER] Total: {entry['t_score']:.2f}\n")
                    for reason, val in entry['t_breakdown'].items():
                        f.write(f"      {reason}: {val:.2f}\n")
                        
                    f.write(f"  [RUNNER] Total: {entry['r_score']:.2f}\n")
                    for reason, val in entry['r_breakdown'].items():
                        f.write(f"      {reason}: {val:.2f}\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Detailed log saved to: {filename}")
            
        except Exception as e:
            self.log_error(f"Failed to write detailed log: {e}")

    def log_attention(self, message):
        self.logger.info(f"!!! ATTENTION: {message}")
        self.error_logger.info(f"ATTENTION: {message}")
        print(f"!!! ATTENTION: {message}")

    def log_error(self, message):
        self.logger.error(f"!!! ERROR: {message} !!!")
        self.error_logger.error(f"ERROR: {message}")
        print(f"!!! ERROR: {message} !!!")

    def log_status(self, message):
        self.logger.info(f"STATUS: {message}")
        print(f"STATUS: {message}")