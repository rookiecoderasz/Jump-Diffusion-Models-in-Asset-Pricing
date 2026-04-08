# Jump Diffusion Models in Asset Pricing

## Project Context
Implementation of Merton's Jump Diffusion Model for capturing discontinuous price movements in asset pricing.

## Active Agent Team
- **alpha-researcher**: Model calibration, parameter estimation, jump detection
- **risk-manager**: Tail risk analysis, jump risk hedging, VaR under jump diffusion
- **quant-reviewer**: Numerical stability of SDE solvers, Monte Carlo convergence

## Domain-Specific Rules
- Use log-likelihood estimation for jump parameters (λ, μ_J, σ_J)
- Monte Carlo paths must converge (check with increasing N)
- Compare GBM vs Jump Diffusion with likelihood ratio tests
- Validate option pricing against market implied volatility surface
- Ensure numerical stability in characteristic function inversion (FFT pricing)
- Document all model assumptions and parameter sensitivity
