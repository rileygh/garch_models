'''
Utility functions for GARCH modelling
'''

import numpy as np
from numpy.typing import NDArray
from typing import Optional

def simulate_garch_returns(n: int, mu: float=0.0, omega: float=0.001, alpha: float=0.05, beta: float=0.85, seed: Optional[int]=None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    Simulate returns from a GARCH(1, 1) process

    Parameters:
    n : int
    -> no. of observations
    mu : float
    -> mean of returns
    omega : float
    -> GARCH constant term
    alpha : float
    -> ARCH coeff.
    beta : float
    -> GARCH coeff.
    seed : int | None

    Returns:
    returns : np.ndarray
    -> simulated returns
    sigma2 : np.ndarray
    -> true conditional variance
    '''

    if seed is not None:
        np.random.seed(seed)

    returns = np.zeros(n)
    sigma2 = np.zeros(n)

    sigma2[0] = omega / (1 - alpha - beta) # unconditional variance

    for t in range(n):
        z_t = np.random.normal(0, 1) # standardised residual z_t ~ N(0, 1)
        epsilon_t = np.sqrt(sigma2[t]) * z_t # actual residual
        returns[t] = mu + epsilon_t

        if t < n - 1:
            sigma2[t + 1] = omega + alpha * epsilon_t**2 + beta * sigma2[t]

    return returns, sigma2
