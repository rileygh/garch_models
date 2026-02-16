import numpy as np
import matplotlib.pyplot as plt
from garch.models import GARCH
from garch.utils import simulate_garch_returns

# TODO:
# a lot is wrong so debug:
# alpha and beta estimates are exact, so clearly not estimated
# omega estimate is far too small
# log-likelihood seems to be negative (correct) with the wrong (sign-flipped) formula?
# unsure if magnitude of information criteria is reasonable
# forecast needs investigating

def main():
    print('Simulating GARCH data...')
    returns, true_sigma2 = simulate_garch_returns(n=1000, seed=42) # creates simulated data to fit models to

    print('Fitting models...')
    models = {
        'GARCH': GARCH(returns)
    }

    for n, model in models.items():
        print(f'Fitting model: {n}')
        model.fit()
        model.summary()
    
    # compare
    horizon = 10
    for n, model in models.items():
        forecast = model.forecast(horizon=horizon)
        print(f'10-step forecast:\n')
        print(forecast)

    _, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(returns, alpha=0.5, label='Returns')
    axes[0].set_title('Simulated Returns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # compare estimated volatilities
    axes[1].plot(np.sqrt(true_sigma2), 'k-', label='True', linewidth=2, alpha=0.7)
    for name, model in models.items():
        axes[1].plot(np.sqrt(model.sigma2), label=f'{name}', alpha=0.7)
    axes[1].set_title('Conditional Volatility Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("\nPlot saved as 'model_comparison.png'")

if __name__ == '__main__':
    main()