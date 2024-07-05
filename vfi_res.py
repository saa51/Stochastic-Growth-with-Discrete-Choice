import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.interpolate import interp1d, interp2d


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
    labor_choice: np.ndarray

    a_grid: np.ndarray      # (na, )
    a_trans: np.ndarray     # (na, na)
    k_grid: np.ndarray      # (nk, )
    value: np.ndarray       # (nk, na)
    partial_q: np.ndarray
    policy_l: np.ndarray
    policy_k: np.ndarray

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

    def plot_q(self, idxa_vec=None, title='', fname=None, show=False):
        idxa_vec = np.arange(len(self.a_grid)) if idxa_vec is None else idxa_vec
        for i in range(len(self.labor_choice)):
            for idxa in idxa_vec:
                plt.plot(self.k_grid, 
                         self.partial_q[i, :, idxa], 
                         label=f"l={self.labor_choice[i]:.2f}, a={self.a_grid[idxa]:.2f}"
                        )
        plt.legend()
        plt.title('Partial Q Function: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_policy(self, idxa_vec=None, title='', fname=None, show=False):
        idxa_vec = np.arange(len(self.a_grid)) if idxa_vec is None else idxa_vec
        for i in range(len(self.labor_choice)):
            for idxa in range(len(self.a_grid)):
                plt.plot(self.k_grid, 
                         self.k_grid[self.policy_k[i, :, idxa]], 
                         label=f"l={self.labor_choice[i]:.2f}, a={self.a_grid[idxa]:.2f}"
                        )
        plt.plot(self.k_grid, self.k_grid, label='45 degree')
        # plt.scatter([self.k_ss], [self.k_ss])
        plt.legend()
        plt.title('Conditional Policy Function: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_capital_diff(self, idxa_vec=None, title='', fname=None, show=False):
        idxa_vec = np.arange(len(self.a_grid)) if idxa_vec is None else idxa_vec
        for i in range(len(self.labor_choice)):
            for idxa in range(len(self.a_grid)):
                plt.plot(self.k_grid, 
                         self.k_grid[self.policy_k[i, :, idxa]] - self.k_grid, 
                         label=f"l={self.labor_choice[i]:.2f}, a={self.a_grid[idxa]:.2f}"
                        )
        plt.legend()
        plt.title('Capital Difference: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def euler_error(self, grids):
        policy_func = [[interp1d(self.k_grid, self.k_grid[self.policy_k[l, :, idxa]]) for idxa in range(len(self.a_grid))] for l in range(len(self.labor_choice))]

        grids = np.array(grids)
        k_list = grids[:, 0]
        idxa_list = np.argmin((grids[:, 1].reshape(-1, 1) - self.a_grid.reshape(1, -1)) ** 2, axis=1)
        res = []
        for k, idxa in zip(k_list, idxa_list):
            # TO DO: compute Euler error
            idxl = self.policy_l
            k_prime = policy_func[idxa](k)
            c = np.exp(self.a_grid[idxa]) * k ** self.alpha + (1 - self.delta) * k - k_prime
            k_2primes = np.array([policy_func[i](k_prime) for i in range(len(self.a_grid))])
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

    def plot_policy_derivative(self, idxa_vec=None, title='', fname=None, show=False):
        idxa_vec = np.arange(len(self.a_grid)) if idxa_vec is None else idxa_vec
        for i in range(len(self.labor_choice)):
            for idxa in idxa_vec:
                gradient = (self.k_grid[self.policy_k[i, 1:, idxa]] - self.k_grid[self.policy_k[i, :-1, idxa]]) / (self.k_grid[1:] - self.k_grid[:-1])
                plt.plot((self.k_grid[1:] + self.k_grid[:-1]) / 2, gradient, label=f"l={self.labor_choice[i]:.2f}, a={self.a_grid[idxa]:.2f}")
        plt.legend()
        plt.title('Policy Function Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_policy_2derivative(self, idxa_vec=None, title='', fname=None, show=False):
        idxa_vec = np.arange(len(self.a_grid)) if idxa_vec is None else idxa_vec
        for i in range(len(self.labor_choice)):
            for idxa in idxa_vec:
                gradient = (self.k_grid[self.policy_k[i, 1:, idxa]] - self.k_grid[self.policy_k[i, :-1, idxa]]) / (self.k_grid[1:] - self.k_grid[:-1])
                gradient_2 = (gradient[1:] - gradient[:-1]) / (self.k_grid[2:] - self.k_grid[:-2]) * 2
                plt.plot((self.k_grid[2:] + self.k_grid[:-2]) / 2, gradient_2, label=f"l={self.labor_choice[i]:.2f}, a={self.a_grid[idxa]:.2f}")
        plt.legend()
        plt.title('Policy Function 2nd Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def simulate(self, start, number=1000, periods=2500, tfp_series=None):
        partial_q_funcs = [interp2d(self.k_grid, self.a_grid, self.partial_q[i].T) for i in range(len(self.labor_choice))]
        policy_funcs = [interp2d(self.k_grid, self.a_grid, self.k_grid[self.policy_k[i]].T) for i in range(len(self.labor_choice))]
        # To Do: simulation
        k_0, a_0 = start
        if tfp_series is None:
            a_series = np.empty((number, periods + 1), dtype=np.float64)
            a_series[:, 0] = a_0
            for t in range(periods):
                a_series[:, t + 1] = self.rho * a_series[:, t] + np.random.normal(0, self.sigma, a_series[:, t + 1].shape)
            tfp_series = np.exp(a_series)
        else:
            assert (np.log(tfp_series[:, 0]) == a_0).all()
            a_series = np.log(tfp_series)
        k_series = np.zeros_like(a_series, dtype=np.float64)
        k_series[:, 0] = k_0
        l_series = np.zeros_like(a_series, dtype=np.int32)

        for t in range(a_series.shape[1] - 1):
            l_series[:, t] = np.argmax([partial_q_funcs[l](k_series[:, t], a_series[:, t]).diagonal() for l in range(len(self.labor_choice))], axis=0)
            for n in range(a_series.shape[0]):
                k_series[n, t + 1] = policy_funcs[l_series[n, t]](k_series[n, t], a_series[n, t])

        i_series = k_series[:, 1:] - (1 - self.delta) * k_series[:, :-1]
        y_series = np.exp(a_series) * k_series ** self.alpha
        c_series = y_series[:, :-1] - i_series
        return {
            'a': a_series[:, :-1],
            'k': k_series[:, :-1],
            'c': c_series,
            'i': i_series,
            'y': y_series,
            'l': l_series[:, :-1],
            'TFP': tfp_series,
        }

    @staticmethod
    def pow_utility(x, rra):
        return np.log(x) if rra == 1 else x ** (1 - rra) / (1 - rra)

    def plot_exact_value(self, a=0, simu_num=1000, periods=1000, title='', fname=None, show=False):
        value = np.zeros(self.k_grid.shape + (simu_num, periods))
        for idxk, k in enumerate(self.k_grid):
            res = self.simulate((k, a), simu_num, periods)
            c = res['c']
            l = self.labor_choice[res['l']]
            utility = self.pow_utility(c, self.gamma) + self.B * self.pow_utility(self.L - l, self.eta)
            discount = self.beta ** np.arange(utility.shape[-1])
            value[idxk] = np.sum(utility * discount, axis=-1)
        value_mean = np.mean(value, axis=-1)
        lower_ci = np.quantile(value, q=0.025, axis=-1)
        upper_ci = np.quantile(value, q=0.975, axis=-1)           
        plt.fill_between(self.k_grid, lower_ci, upper_ci, alpha=0.5, label='95% CI')
        plt.plot(self.k_grid, value_mean, label='simulated')
        value_func = interp2d(self.k_grid, self.a_grid, self.value.T)
        plt.plot(self.k_grid, value_func(self.k_grid, a * np.ones(self.k_grid)).diagonal(), label='numeric')
        plt.legend()
        plt.title('Value Function: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()

    def plot_l(self, idxa_vec=None, title='', fname=None, show=False):
        idxa_vec = np.arange(len(self.a_grid)) if idxa_vec is None else idxa_vec
        for idxa in idxa_vec:
            l = np.argmax(self.partial_q[..., idxa], axis=0)
            plt.plot(self.k_grid, self.labor_choice[l], label=f"a={self.a_grid[idxa]:.2f}")
        plt.legend()
        plt.title('Labor Function: ' + title)
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()


    def to_dict(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'rho': self.rho,
            'sigma': self.sigma,
            'eta': self.eta,
            'B': self.B,
            'L': self.L,
            'labor_choice': self.labor_choice,
            'a_grid': self.a_grid,
            'a_trans': self.a_trans,
            'k_grid': self.k_grid,
            'value': self.value,
            'policy_k': self.policy_k,
            'policy_l': self.policy_l,
            'partial_q': self.partial_q,
        }
    