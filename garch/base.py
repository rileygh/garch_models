'''
Abstract base class (ABC) for GARCH-family models
'''

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
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

        self.returns: NDArray[np.float64] = np.asarray(returns, dtype=np.float64)
        self.T: int = len(returns)
        self.params: NDArray[np.float64] | None = None
        self.sigma2: NDArray[np.float64] | None = None
        self.epsilon: NDArray[np.float64] | None = None
        self.std_residuals: NDArray[np.float64] | None = None
        self.log_likelihood: float | None = None
        self._fitted: bool = False

    @abstractmethod
    def get_initial_params(self) -> NDArray[np.float64]:
        '''
        Gets initial parameter values for optimisation

        Returns:
        initial_params : np.ndarray
        -> initial parameter estimations
        '''
        pass

    @abstractmethod
    def get_param_names(self) -> list[str]:
        '''
        Different models have different parameter names, so this retrieves them for display

        Returns:
        param_names : list[str]
        -> parameter names (ordered)
        '''
        pass

    @abstractmethod
    def get_constraints(self) -> tuple[dict]:
        '''
        Gets constraints for parameters used in optimisation

        Returns:
        constraints : tuple
        -> definitions for scipy.optimize.minimize
        '''

    @abstractmethod
    def _compute_variance_recursion(self, params, epsilon, sigma2) -> NDArray[np.float64]:
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
    def forecast(self, horizon=1) -> NDArray[np.float64]:
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

    def _log_likelihood(self, params) -> float:
        '''
        Computes the log-likelihood

        Parameters:
        params : np.ndarray
        -> mean and model-specific parameters

        Returns:
        ll : float
        -> log-likelihood value
        '''

        mu: float = params[0]
        model_params: NDArray[np.float64] = params[1:]
        sigma2: NDArray[np.float64] = np.zeros(self.T)
        sigma2[0] = np.var(self.returns) # initial variance taken from return series
        epsilon: NDArray[np.float64] = self.returns - mu # residuals

        # populate sigma2 with model-specific formula for conditional variance
        sigma2 = self._compute_variance_recursion(model_params, epsilon, sigma2)

        if not np.all(sigma2 > 0): raise ValueError('require all variances to be positive')
        
        # applying log-likelihood formula
        ll: float = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) - (epsilon ** 2 / sigma2))

        return ll
    
    def _compute_conditional_variance(self) -> NDArray[np.float64]:
        '''
        Computes conditional variances using fitted parameters

        Returns:
        conditional_var : np.ndarray
        -> conditional variance array
        '''

        mu: float = self.params[0]
        model_params = self.params[1:]
        
        conditional_var: NDArray[np.float64] = np.zeros(self.T)
        conditional_var[0] = np.var(self.returns)

        self.epsilon = self.returns - mu

        conditional_var = self._compute_variance_recursion(model_params, self.epsilon, conditional_var)

        self.std_residuals = self.epsilon / np.sqrt(conditional_var)

        return conditional_var
    
    def fit(self, initial_params=get_initial_params(), method='SLSQP', options={'maxiter': 1000}):
        '''
        Estimates model parameters using maximum likelihood estimation (MLE)

        Parameters:
        initial_params : np.ndarray | None
        -> initial parameter values (defaults available if none given)
        method : str - default SLSQP
        -> optimisation method for scipy.optimize.minimize
        options : dict | None
        -> additional options
        '''

        constraints: tuple[dict] = self.get_constraints()

        res = minimize(-self.log_likelihood, initial_params, method=method, constraints=constraints, options=options)

        if not res.success: raise RuntimeError(f'optimisation did not converge. message: {res.message}')

        self.params = res.x # solution array
        self.log_likelihood = res.fun
        self.sigma2 = self._compute_conditional_variance()
        self._fitted = True

        return self

    def summary(self):
        '''
        Provides a summary of parameter estimates and model statistics
        '''
        if not self._fitted:
            raise RuntimeError('model not fitted. make sure that you have called the fit method before trying to display a summary')
        
        param_names = self.get_param_names()

        print(f'\n{self.__class__.__name__} Model Estimates\n{'='*60}')

        for n, v in zip(param_names, self.params):
            print(f'{n:25s}: {v:.6e}' if abs(v) < 0.01 else f'{n:25s}: {v:.6f}')
        
        print(f'{'Log-likelihood':25s}: {self.log_likelihood:.2f}')

        # info criteria AIC/BIC
        k = len(self.params)
        aic = -2 * self.log_likelihood + 2*k
        bic = -2 * self.log_likelihood + k*np.log(self.T)

        print(f'Information criteria:\n{'AIC':25s}: {aic:.2f}\n{'BIC':25s}: {bic:.2f}\n{'='*60}')
        