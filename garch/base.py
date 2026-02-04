'''
Abstract base class (ABC) for GARCH-family models
'''

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize

class BaseGARCH(ABC):
    '''
    All GARCH variants should inherit from this class and implement the abstractmethods

    Attributes:
    returns : np.ndarray
    -> time-series of returns
    T : int
    -> no. of observations
    params : np.ndarray | None
    -> estimated parameters after fitting
    sigma2 : np.ndarray | None
    -> conditional variance series
    epsilon : np.ndarray | None
    -> residual series
    std_residuals : np.ndarray | None
    -> standardised residuals (epsilon/sigma)
    log_likelihood : float | None
    -> log-likelihood at optimal parameters
    _fitted : bool
    -> represents if the model has been fitted or not
    '''

    def __init__(self, returns):
        '''
        Initialise GARCH model

        Parameters:
        returns : ArrayLike
        -> time-series of returns
        '''

        self.returns = np.array(returns)
        self.T = len(returns)
        self.params = None
        self.sigma2 = None
        self.epsilon = None
        self.std_residuals = None
        self.log_likelihood = None
        self._fitted = False

    @abstractmethod
    def get_initial_params(self):
        '''
        Gets initial parameter values for optimisation

        Returns:
        initial_params : np.ndarray
        -> initial parameter estimations
        '''
        pass

    @abstractmethod
    def get_constraints(self):
        '''
        Gets constraints for parameters used in optimisation

        Returns:
        constraints : tuple
        -> definitions for scipy.optimize.minimize
        '''

    @abstractmethod
    def _compute_variance_recursion(self, params, epsilon, sigma2):
        '''
        Computes conditional variance using model-specific recursion

        Parameters:
        params : np.ndarray
        -> model parameters (excl. mean)
        epsilon : np.ndarray
        -> residual series
        sigma2 : np.ndarray
        -> conditional variance series, array to fill in-place

        Returns:
        sigma2 : np.ndarray
        -> filled conditional variance series array
        '''
        pass

    @abstractmethod
    def forecast(self, horizon=1):
        '''
        Forecasts variance for future periods (horizon as the no. of periods)

        Parameters:
        horizon : int:
        -> number of periods to forecast

        Returns:
        forecasts : np.ndarray
        -> array of variance forecasts
        '''
        pass

    def _log_likelihood(self, params):
        '''
        Computes the log-likelihood

        Parameters:
        params : np.ndarray
        -> mean and model-specific parameters

        Returns:
        ll : float
        -> log-likelihood value
        '''

        mu = params[0]
        model_params = params[1:]
        sigma2 = np.zeros(self.T)
        sigma2[0] = np.var(self.returns) # initial variance taken from return series
        epsilon = self.returns - mu # residuals

        # populate sigma2 with model-specific formula for conditional variance
        sigma2 = self._compute_variance_recursion(model_params, epsilon, sigma2)

        if not np.all(sigma2 > 0): raise ValueError('require all variances to be positive')
        
        # applying log-likelihood formula
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) - (epsilon ** 2 / sigma2))

        return log_likelihood
    
    