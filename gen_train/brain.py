import numpy as np
import random

class SimpleBrain:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        scale1 = 1.0 / np.sqrt(input_size)
        self.w1 = np.random.uniform(-scale1, scale1, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        
        scale2 = 1.0 / np.sqrt(hidden_size)
        self.w2 = np.random.uniform(-scale2, scale2, (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

        self.b2[1] = 0.5 

    def forward(self, inputs):
        self.z1 = np.tanh(np.dot(inputs, self.w1) + self.b1)
        output = np.tanh(np.dot(self.z1, self.w2) + self.b2)
        return output 

    def mutate(self, rate, strength_mult=1.0):
        eff_str = 0.3 * strength_mult
        
        if random.random() < rate:
            mask1 = np.random.choice([0, 1], size=self.w1.shape, p=[1-rate, rate])
            self.w1 += np.random.randn(*self.w1.shape) * eff_str * mask1
            
            mask2 = np.random.choice([0, 1], size=self.w2.shape, p=[1-rate, rate])
            self.w2 += np.random.randn(*self.w2.shape) * eff_str * mask2
            
            if random.random() < 0.2:
                self.b2 += np.random.randn(self.output_size) * 0.1 * strength_mult
            
            if random.random() < 0.05:
                self.b2[3] = 0.0

    def clone(self):
        clone = SimpleBrain(self.input_size, self.hidden_size, self.output_size)
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        return clone