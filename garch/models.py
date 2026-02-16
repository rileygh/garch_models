'''
Implements various GARCH-family models

Currently only supports GARCH models of order (1, 1) but plans to extend to (p, q) in the future
'''

import numpy as np
from .base import BaseGARCH

class GARCH(BaseGARCH):
    '''
    Standard GARCH(1, 1)

    Variance given by:
    sigma^2_t = omega + alpha*epsilon^2_(t-1) + beta*sigma^2_(t-1)
    '''

    def get_initial_params(self):
        mu_init: float = np.mean(self.returns)
        var_init: float = np.var(self.returns)

        omega: float = 0.01 * var_init
        alpha, beta = 0.05, 0.9

        return np.array([mu_init, omega, alpha, beta])
    
    def get_param_names(self):
        return ['mu', 'omega', 'alpha', 'beta']
    
    def get_constraints(self):
        return (
            {'type': 'ineq', 'fun': lambda x: x[1]}, # omega > 0
            {'type': 'ineq', 'fun': lambda x: x[2]}, # alpha > 0
            {'type': 'ineq', 'fun': lambda x: x[3]}, # beta > 0
            {'type': 'ineq', 'fun': lambda x: 0.999 - x[2] - x[3]}, # alpha + beta < 1
        )
    
    def get_bounds(self):
        '''Get parameter bounds for optimisation'''
        return (
            (None, None), # mu: unbounded
            (1e-8, None), # omega: strictly positive
            (1e-8, 0.999), # alpha: between 0 and 1
            (1e-8, 0.999), # beta: between 0 and 1
        )
    
    def _compute_variance_recursion(self, params, epsilon, sigma2):
        omega, alpha, beta = params

        for i in range(1, self.T):
            sigma2[i] = omega + alpha * epsilon[i-1] ** 2 + beta * sigma2[i-1]

        return sigma2
    
    def forecast(self, horizon=1):
        if not self._fitted:
            raise RuntimeError('model not fitted. make sure that you have called the fit method before trying to forecast')

        _, omega, alpha, beta = self.params

        last_residual2 = self.epsilon[-1] ** 2
        last_sigma2 = self.sigma2[-1]
        osa = omega + alpha * last_residual2 + beta * last_sigma2 # one-step-ahead forecast

        if horizon > 1:
            forecasts = np.zeros(horizon)
            forecasts[0] = osa

            for i in range(1, horizon):
                forecasts[i] = omega + (alpha + beta) * forecasts[i-1]
            
            return forecasts

        return osa