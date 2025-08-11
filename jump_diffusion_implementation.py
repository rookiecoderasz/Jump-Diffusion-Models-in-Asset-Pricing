

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd
import math

class JumpDiffusionModel:
    """
    Implementation of Merton's Jump Diffusion Model for stock price simulation
    and option pricing.
    """
    
    def __init__(self, S0, mu, sigma, jump_lambda, jump_mu, jump_sigma, r=0.05, T=1.0):
        """
        Initialize the Jump Diffusion Model parameters.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        mu : float
            Drift rate (expected return)
        sigma : float
            Volatility (standard deviation of continuous part)
        jump_lambda : float
            Jump intensity (average number of jumps per unit time)
        jump_mu : float
            Mean of jump size (log-normal distribution)
        jump_sigma : float
            Standard deviation of jump size
        r : float
            Risk-free rate
        T : float
            Time horizon
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.jump_lambda = jump_lambda
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma
        self.r = r
        self.T = T
        
    def simulate_path(self, n_steps, n_paths=1):
        """
        Simulate stock price paths using Monte Carlo simulation.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
            
        Returns:
        --------
        numpy.ndarray
            Simulated price paths
        """
        dt = self.T / n_steps
        time_grid = np.linspace(0, self.T, n_steps + 1)
        
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
            
            # Update prices using the jump diffusion formula
            drift = (self.mu - 0.5 * self.sigma**2 - self.jump_lambda * 
                    (np.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1)) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z
            
            prices[:, i] = prices[:, i-1] * np.exp(drift + diffusion) * (1 + jump_component)
            
        return time_grid, prices
    
    def merton_option_price(self, K, option_type='call', n_terms=20):
        """
        Calculate European option price using Merton's jump diffusion formula.
        
        Parameters:
        -----------
        K : float
            Strike price
        option_type : str
            'call' or 'put'
        n_terms : int
            Number of terms in the infinite series approximation
            
        Returns:
        --------
        float
            Option price
        """
        option_price = 0.0
        
        # Expected jump size
        k = np.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1
        
        for n in range(n_terms):
            # Poisson probability
            poisson_prob = np.exp(-self.jump_lambda * self.T) * \
                          (self.jump_lambda * self.T)**n / math.factorial(n)
            
            # Adjusted parameters for n jumps
            sigma_n = np.sqrt(self.sigma**2 + n * self.jump_sigma**2 / self.T)
            r_n = self.r - self.jump_lambda * k + n * (self.jump_mu + 0.5 * self.jump_sigma**2) / self.T
            
            # Black-Scholes price with adjusted parameters
            bs_price = self._black_scholes(self.S0, K, r_n, sigma_n, self.T, option_type)
            
            option_price += poisson_prob * bs_price
            
        return option_price
    
    def _black_scholes(self, S, K, r, sigma, T, option_type):
        """
        Black-Scholes formula for European options.
        """
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            
        return price
    
    def calibrate_to_market(self, market_prices, strikes, maturities, option_types):
        """
        Calibrate model parameters to market option prices.
        
        Parameters:
        -----------
        market_prices : array
            Market option prices
        strikes : array
            Strike prices
        maturities : array
            Time to maturity
        option_types : array
            Option types ('call' or 'put')
            
        Returns:
        --------
        dict
            Calibrated parameters
        """
        def objective_function(params):
            sigma, jump_lambda, jump_mu, jump_sigma = params
            
            # Update model parameters
            temp_model = JumpDiffusionModel(
                self.S0, self.mu, sigma, jump_lambda, jump_mu, jump_sigma, self.r
            )
            
            total_error = 0
            for i, (market_price, K, T, opt_type) in enumerate(
                zip(market_prices, strikes, maturities, option_types)
            ):
                temp_model.T = T
                model_price = temp_model.merton_option_price(K, opt_type)
                total_error += (model_price - market_price)**2
                
            return total_error
        
        # Initial guess and bounds
        initial_guess = [self.sigma, self.jump_lambda, self.jump_mu, self.jump_sigma]
        bounds = [(0.01, 1.0), (0.0, 10.0), (-0.5, 0.5), (0.01, 1.0)]
        
        # Optimization
        result = minimize(objective_function, initial_guess, bounds=bounds, 
                         method='L-BFGS-B')
        
        return {
            'sigma': result.x[0],
            'jump_lambda': result.x[1],
            'jump_mu': result.x[2],
            'jump_sigma': result.x[3],
            'optimization_result': result
        }

def plot_simulation_results(time_grid, prices, title="Jump Diffusion Simulation"):
    """
    Plot simulation results.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot first 10 paths if more than 10
    n_plots = min(prices.shape[0], 10)
    for i in range(n_plots):
        plt.plot(time_grid, prices[i, :], alpha=0.7, linewidth=0.8)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_models():
    """
    Compare Jump Diffusion Model with Black-Scholes Model.
    """
    # Model parameters
    S0 = 100
    mu = 0.05
    sigma = 0.2
    jump_lambda = 0.1
    jump_mu = -0.1
    jump_sigma = 0.15
    r = 0.05
    T = 1.0
    
    # Initialize models
    jd_model = JumpDiffusionModel(S0, mu, sigma, jump_lambda, jump_mu, jump_sigma, r, T)
    
    # Simulate paths
    n_steps = 252  # Daily steps for 1 year
    n_paths = 1000
    
    time_grid, jd_prices = jd_model.simulate_path(n_steps, n_paths)
    
    # Compare option prices
    strikes = np.arange(80, 121, 5)
    
    print("Strike\tJump Diffusion\tBlack-Scholes\tDifference")
    print("-" * 50)
    
    for K in strikes:
        jd_price = jd_model.merton_option_price(K, 'call')
        bs_price = jd_model._black_scholes(S0, K, r, sigma, T, 'call')
        diff = jd_price - bs_price
        
        print(f"{K}\t{jd_price:.4f}\t\t{bs_price:.4f}\t\t{diff:.4f}")
    
    # Plot simulation results
    plot_simulation_results(time_grid, jd_prices, 
                          "Jump Diffusion vs Black-Scholes Simulation")

# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Basic simulation
    print("=== Jump Diffusion Model Implementation ===\n")
    
    # Model parameters
    S0 = 100.0      # Initial stock price
    mu = 0.05       # Drift rate
    sigma = 0.2     # Volatility
    jump_lambda = 0.1  # Jump intensity
    jump_mu = -0.1  # Mean jump size
    jump_sigma = 0.15  # Jump volatility
    r = 0.05        # Risk-free rate
    T = 1.0         # Time horizon
    
    # Create model instance
    model = JumpDiffusionModel(S0, mu, sigma, jump_lambda, jump_mu, jump_sigma, r, T)
    
    # Simulate stock price paths
    n_steps = 252
    n_paths = 5
    time_grid, prices = model.simulate_path(n_steps, n_paths)
    
    print(f"Simulated {n_paths} paths with {n_steps} time steps")
    print(f"Final prices: {prices[:, -1]}")
    
    # Calculate option prices
    K = 100  # At-the-money option
    call_price = model.merton_option_price(K, 'call')
    put_price = model.merton_option_price(K, 'put')
    
    print(f"\nOption Prices (Strike = {K}):")
    print(f"Call: ${call_price:.4f}")
    print(f"Put: ${put_price:.4f}")
    
    # Example 2: Model comparison
    print("\n=== Model Comparison ===")
    compare_models()
    
    print("\n=== Implementation Complete ===")
