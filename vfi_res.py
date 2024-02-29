import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SOGVFIResult:
    alpha: float
    beta: float
    gamma: float
    delta: float
    rho: float
    sigma: float
    eta: float
    B: float
    L: float
    labor_choice: np.array

    a_grid: np.array
    a_trans: np.array
    k_grid: np.array
    value: np.array
    partial_q: np.array
    policy_l: np.array
    policy_k: np.array

    def plot_value(self, title='', fname=None, show=False):
        for idxa in range(len(self.a_grid)):
            a = self.a_grid[idxa]
            plt.plot(self.k_grid, self.value[:, idxa], label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Value Function: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_q(self, title='', fname=None, show=False):
        for i in range(len(self.labor_choice)):
            plt.plot(self.k_grid, self.partial_q[i, :, ])

    def plot_policy(self, title='', fname=None, show=False):
        for idxa in range(len(self.a_grid)):
            a = self.a_grid[idxa]
            plt.plot(self.k_grid, self.k_grid[self.policy[:, idxa]], label='a=' + str(round(a, 2)))
        plt.plot(self.k_grid, self.k_grid, label='45 degree')
        # plt.scatter([self.k_ss], [self.k_ss])
        plt.legend()
        plt.title('Policy Function: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_capital_diff(self, title='', fname=None):
        for idxa in range(len(self.a_grid)):
            a = self.a_grid[idxa]
            plt.plot(self.k_grid, self.k_grid[self.policy[:, idxa]] - self.k_grid, label='a=' + str(round(a, 2)))
        plt.plot()
        plt.legend()
        plt.title('Capital Difference: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if self._show:
            plt.show()
        else:
            plt.clf()

    def euler_error(self, grids):
        policy_func = [SplineApprox() for _ in range(len(self.a_grid))]
        for idxa in range(len(self.a_grid)):
            policy_func[idxa].approx(self.k_grid, self.k_grid[self.policy[:, idxa]])

        grids = np.array(grids)
        k_list = grids[:, 0]
        idxa_list = np.argmin((grids[:, 1].reshape(-1, 1) - self.a_grid.reshape(1, -1)) ** 2, axis=1)
        res = []
        for k, idxa in zip(k_list, idxa_list):
            k_prime = policy_func[idxa].eval(k)
            c = np.exp(self.a_grid[idxa]) * k ** self.alpha + (1 - self.delta) * k - k_prime
            k_2primes = np.array([policy_func[i].eval(k_prime) for i in range(len(self.a_grid))])
            c_primes = np.exp(self.a_grid) * k_prime ** self.alpha + (1 - self.delta) * k_prime - k_2primes
            r_primes = self.alpha * np.exp(self.a_grid) * k_prime ** (self.alpha - 1) + 1 - self.delta
            res.append(1 - (self.beta * np.sum(r_primes * self.a_trans[idxa] * c_primes ** (-self.gamma))) ** (-1 / self.gamma) / c)
        return res

    def plot_euler_err(self, title='', fname=None, show=False):
        errs = []
        for idxa in range(len(self.a_grid)):
            grids = [(self.k_grid[i], self.a_grid[idxa]) for i in range(len(self.k_grid))]
            err = self.euler_error(grids)
            err = np.log10(np.abs(err))
            plt.plot(self.k_grid, err, label='a=' + str(round(self.a_grid[idxa], 2)))
            errs.append(err)
        plt.legend()
        plt.title('Euler Error: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()
        return errs

    def plot_value_derivative(self, title='', fname=None, show=False):
        for idxa in range(len(self.a_grid)):
            a = self.a_grid[idxa]
            plt.plot((self.k_grid[1:] + self.k_grid[:-1]) / 2, 
                     (self.value[1:, idxa] - self.value[:-1, idxa]) / (self.k_grid[1:] - self.k_grid[:-1]), 
                     label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Value Function Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_value_2derivative(self, title='', fname=None, show=False):
        for idxa in range(len(self.a_grid)):
            a = self.a_grid[idxa]
            gradient = (self.value[1:, idxa] - self.value[:-1, idxa]) / (self.k_grid[1:] - self.k_grid[:-1])
            gradient_2 = (gradient[1:] - gradient[:-1]) / (self.k_grid[2:] - self.k_grid[:-2]) * 2
            plt.plot((self.k_grid[2:] + self.k_grid[:-2]) / 2, gradient_2, label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Value Function 2nd Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_policy_derivative(self, title='', fname=None, show=False):
        for idxa in range(len(self.a_grid)):
            a = self.a_grid[idxa]
            gradient = (self.k_grid[self.policy[1:, idxa]] - self.k_grid[self.policy[:-1, idxa]]) / (self.k_grid[1:] - self.k_grid[:-1])
            plt.plot((self.k_grid[1:] + self.k_grid[:-1]) / 2, gradient, label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Policy Function Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_policy_2derivative(self, title='', fname=None, show=False):
        for idxa in range(len(self.a_grid)):
            a = self.a_grid[idxa]
            gradient = (self.k_grid[self.policy[1:, idxa]] - self.k_grid[self.policy[:-1, idxa]]) / (self.k_grid[1:] - self.k_grid[:-1])
            gradient_2 = (gradient[1:] - gradient[:-1]) / (self.k_grid[2:] - self.k_grid[:-2]) * 2
            plt.plot((self.k_grid[2:] + self.k_grid[:-2]) / 2, gradient_2, label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Policy Function 2nd Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def simulate(self, periods=2500, tfp_series=None):
        policy_func = [SplineApprox() for _ in range(len(self.a_grid))]
        for idxa in range(len(self.a_grid)):
            policy_func[idxa].approx(self.k_grid, self.k_grid[self.policy[:, idxa]])

        if tfp_series is None:
            idxa_series, a_series = FiniteMarkov(self.a_grid, self.a_trans).simulate(periods, 0)
            tfp_series = (idxa_series, a_series)
        else:
            idxa_series, a_series = tfp_series
        k_series = np.zeros(periods + 2)
        k_series[0] = np.mean(self.k_grid)

        for t in range(0, periods + 1):
            k_series[t + 1] = policy_func[idxa_series[t]].eval(k_series[t])
        a_series = a_series[1:]
        k_series = k_series[1:]
        i_series = k_series[1:] - (1 - self.delta) * k_series[:-1]
        k_series = k_series[:-1]
        y_series = np.exp(a_series) * k_series ** self.alpha
        c_series = y_series - i_series
        return a_series, k_series, c_series, i_series, y_series, tfp_series

    def to_dict(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'a_grid': self.a_grid,
            'a_trans': self.a_trans,
            'k_grid': self.k_grid,
            'value': self.value,
            'policy': self.policy,
        }
    