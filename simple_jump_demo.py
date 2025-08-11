# Simple demonstration of Jump Diffusion simulation with visualization

import numpy as np
import matplotlib.pyplot as plt
import math

class SimpleJumpDiffusion:
    def __init__(self, S0=100, mu=0.05, sigma=0.2, jump_lambda=0.1, jump_mu=-0.1, jump_sigma=0.15):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.jump_lambda = jump_lambda
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma
    
    def simulate_path(self, T=1.0, n_steps=252, n_paths=10):
        dt = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize price array
        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = self.S0
        
        for i in range(1, n_steps + 1):
            # Generate random numbers
            Z = np.random.standard_normal(n_paths)  # Brownian motion
            N = np.random.poisson(self.jump_lambda * dt, n_paths)  # Jump process
            
            # Jump component
            jump_component = np.zeros(n_paths)
            for j in range(n_paths):
                if N[j] > 0:
                    # Generate jump sizes
                    jump_sizes = np.random.normal(self.jump_mu, self.jump_sigma, N[j])
                    jump_component[j] = np.sum(np.exp(jump_sizes) - 1)
            
            # Update prices
            drift = (self.mu - 0.5 * self.sigma**2 - self.jump_lambda * 
                    (np.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1)) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z
            
            prices[:, i] = prices[:, i-1] * np.exp(drift + diffusion) * (1 + jump_component)
            
        return time_grid, prices

# Run simulation and save plot
model = SimpleJumpDiffusion()
time_grid, prices = model.simulate_path(n_paths=5)

# Create plot
plt.figure(figsize=(12, 8))
for i in range(prices.shape[0]):
    plt.plot(time_grid, prices[i, :], linewidth=2, alpha=0.8, label=f'Path {i+1}')

plt.title('Jump Diffusion Model - Stock Price Simulation', fontsize=16)
plt.xlabel('Time (Years)', fontsize=12)
plt.ylabel('Stock Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('jump_diffusion_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

print("Jump Diffusion simulation completed!")
print(f"Initial price: ${model.S0}")
print(f"Final prices: {[f'${price:.2f}' for price in prices[:, -1]]}")
