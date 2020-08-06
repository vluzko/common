import numpy as np

from scipy.stats import gaussian_kde, norm
from matplotlib import pyplot as plt

from fire import Fire

def plot_two_gaussians():
    x = np.linspace(-10, 10, 1000)
    mu_1 = 0
    std_1 = 1

    mu_2 = 1
    std_2 = 0.5

    y_1 = np.random.normal(mu_1, std_1, 1000)

    d_1 = gaussian_kde(y_1)

    plt.plot(x, d_1(x))
    plt.plot(x, norm.pdf(x, mu_2, std_2))
    plt.show()

def subplots_pattern():
    x = np.arange(0, 1000)
    y = x * 2
    _, ax = plt.subplots()
    ax.plot(x, y, label='Example')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    Fire()
    