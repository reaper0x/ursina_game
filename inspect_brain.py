import pickle
import numpy as np
import sys
import os

# ---------------------------------------------------------
# 1. PREPARATION: Define the Class Structure
# ---------------------------------------------------------
# Pickle needs this class definition to exist to load the file
class SimpleBrain:
    pass

# We map this class to '__main__' so pickle finds it regardless of where it was saved
sys.modules['__main__'].SimpleBrain = SimpleBrain

# ---------------------------------------------------------
# 2. THE BRAIN CLASS
# ---------------------------------------------------------
class ReconstructedBrain:
    def __init__(self, model_data):
        """
        Initializes the brain using data loaded directly from the pickle file.
        """
        print("Initializing Brain...")
        
        # Extract weights and biases directly from the loaded model object
        # We use getattr() to avoid errors if a specific field is missing
        self.w1 = getattr(model_data, 'w1', None)
        self.b1 = getattr(model_data, 'b1', None)
        self.w2 = getattr(model_data, 'w2', None)
        self.b2 = getattr(model_data, 'b2', None)

        # Validation: Check if we actually found the weights
        if self.w1 is None:
            raise ValueError("❌ Critical Error: Could not find 'w1' (Layer 1 weights) in the model data.")
        
        # If biases are missing (sometimes usually 0 in genetic algos), fill them with zeros
        if self.b1 is None:
            print("⚠ Warning: 'b1' missing. Defaulting to zeros.")
            self.b1 = np.zeros(self.w1.shape[1])
        if self.b2 is None:
            print("⚠ Warning: 'b2' missing. Defaulting to zeros.")
            self.b2 = np.zeros(self.w2.shape[1])

        print("✅ Brain parameters loaded successfully!")
        print(f"   - Input Size:  {self.w1.shape[0]}")
        print(f"   - Hidden Size: {self.w1.shape[1]}")
        print(f"   - Output Size: {self.w2.shape[1] if self.w2 is not None else 'Unknown'}")

    def predict(self, input_data):
        """
        Calculates the output action for a given input state.
        """
        # Ensure input is a numpy array
        x = np.array(input_data)
        
        # Validation: Input shape must match the first weight layer
        if x.shape[0] != self.w1.shape[0]:
            raise ValueError(f"Input mismatch! Model expects {self.w1.shape[0]} inputs, but got {x.shape[0]}.")

        # --- Layer 1 (Hidden) ---
        # Matrix Math: (Input • w1) + b1
        z1 = np.dot(x, self.w1) + self.b1
        
        # Activation Function: Tanh (Squashes values between -1 and 1)
        a1 = np.tanh(z1) 
        
        # --- Layer 2 (Output) ---
        # Matrix Math: (Hidden Output • w2) + b2
        z2 = np.dot(a1, self.w2) + self.b2
        
        # Decision: The highest value is the chosen action
        action_index = np.argmax(z2)
        
        return action_index, z2

# ---------------------------------------------------------
# 3. EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    file_path = 'trained_model.pkl'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: '{file_path}' not found in the current folder.")
    else:
        try:
            print(f"Loading '{file_path}'...")
            
            # Load the raw pickle data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Locate the actual model inside the dictionary
            # Based on our analysis, it's inside the 't_brain' key
            if isinstance(data, dict) and 't_brain' in data:
                model_object = data['t_brain']
            else:
                # Fallback: maybe the file IS the model object
                model_object = data

            # Initialize the Brain
            brain = ReconstructedBrain(model_object)

            # --- TEST RUN ---
            print("\n--- Running Test Prediction ---")
            
            # Generate fake random inputs (157 inputs as discovered earlier)
            dummy_input = np.random.uniform(-1, 1, 157)
            
            # Get a prediction
            action, values = brain.predict(dummy_input)
            
            print(f"Simulated Inputs (First 5): {dummy_input[:5]}")
            print(f"Brain Raw Outputs: {values}")
            print(f"Final Decision: Action #{action}")
            print("-" * 30)

        except Exception as e:
            print(f"\n❌ Execution failed: {e}")
