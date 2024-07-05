import gin
import numpy as np
from scipy.optimize import root_scalar
from MarkovApprox import Rowenhorst
from vfi_res import SOGVFIResult
from tqdm import tqdm
import pandas as pd


@gin.configurable
class SOGVFISolver:
    def __init__(
            self,
            alpha=gin.REQUIRED,
            beta=gin.REQUIRED,
            rho=gin.REQUIRED,
            sigma=gin.REQUIRED,
            delta=gin.REQUIRED,
            gamma=gin.REQUIRED,
            eta=gin.REQUIRED,
            B=gin.REQUIRED,
            L=gin.REQUIRED,
            labor_choice=gin.REQUIRED,
            normalized=True,
        ):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.sigma = sigma
        self.delta = delta
        self.gamma = gamma
        self.eta = eta
        self.B = B
        self.L = L
        self.labor_choice = np.array(labor_choice)

    def steady_state(self):
        kl_ratio = ((1 / self.beta - 1 + self.delta) / self.alpha) ** (1 / (self.alpha - 1))
        cl_ratio = kl_ratio ** self.alpha - self.delta * kl_ratio
        def foc_err(l):
            return kl_ratio ** self.alpha * (1 - self.alpha) * (cl_ratio * l) ** (-self.gamma) - self.B * (self.L - l) ** (-self.eta)
        def foc_err_drv(l):
            return -self.gamma * cl_ratio * kl_ratio ** self.alpha * (1 - self.alpha) * (cl_ratio * l) ** (-self.gamma - 1) - self.eta * self.B * (self.L - l) ** (-self.eta - 1)
        res = root_scalar(foc_err, method='newton', fprime=foc_err_drv, x0=self.L / 2)
        self.l_ss = res.root
        self.k_ss = self.l_ss * kl_ratio
        self.c_ss = self.l_ss * cl_ratio
        self.y_ss = self.c_ss + self.delta * self.k_ss
        ss = [{'k': self.k_ss, 'c': self.c_ss, 'y': self.y_ss, 'l': self.l_ss}]
        for l in self.labor_choice:
            ss.append({'k': l * kl_ratio, 'c': l * cl_ratio, 'y': l * kl_ratio ** self.alpha, 'l': l})
        return pd.DataFrame(ss)

    def crra(self, x, coeff):
        return x ** (1 - coeff) / (1 - coeff) if coeff != 1 else np.log(x)

    def utility(self, c):
        u = np.ones_like(c)
        u[c > 0] = c[c > 0] ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c[c > 0])
        u[c <= 0] = -np.inf
        return u

    def mqp(self, new_v, old_v):
        b_up = self.beta / (1 - self.beta) * np.max(new_v - old_v)
        b_low = self.beta / (1 - self.beta) * np.min(new_v - old_v)
        return new_v + (b_up + b_low) / 2

    @gin.configurable
    def solve(self, nk, na, width, tol=1e-6, **optimize_params) -> SOGVFIResult:
        a_grids, trans_mat = Rowenhorst(self.rho, self.sigma ** 2, 0).approx(na)
        k_min = self.k_ss - width
        k_max = self.k_ss + width
        k_grids = np.linspace(k_min, k_max, nk, endpoint=True)

        value_mat = np.zeros((nk, na))

        if 'HPI' in optimize_params.get('methods', []):
            hpi_policy_iter = optimize_params.get('HPI_policy_iter', 5)
            hpi_value_iter = optimize_params.get('HPI_value_iter', 20)
            hpi = True
        else:
            hpi = False
            hpi_value_iter = 1e8
            hpi_policy_iter = 0
        mqp = 'MQP' in optimize_params.get('methods', [])

        # (na, nk, nl)
        y = np.kron(np.exp(a_grids), np.kron(k_grids ** self.alpha, self.labor_choice)).reshape((na, nk, len(self.labor_choice)))
        # (na, nl, nk)
        cash_on_hand = y.transpose(0, 2, 1) + (1 - self.delta) * k_grids
        # (na, nl, nk, nk')
        u_c = self.utility(np.repeat(np.expand_dims(cash_on_hand, -1), nk, axis=-1) - k_grids)
        # (nl,)
        u_l = self.crra(self.L - self.labor_choice, self.eta)
        # (na, nk, nk', nl)
        u_mat = u_c.transpose(0, 2, 3, 1) + u_l
        # (nl, nk, na, nk')
        u_mat = u_mat.transpose(3, 1, 0, 2) * (1 - self.beta)

        old_value_mat = np.random.random((nk, na))

        iter = 0
        while not np.allclose(old_value_mat, value_mat, atol=tol, rtol=0):
            iter += 1
            if hpi and iter % hpi_policy_iter == 0:
                for _ in range(hpi_value_iter):
                    local_vmat = np.copy(value_mat)
                    value_mat = u_mat + self.beta * trans_mat @ local_vmat.T
                    value_mat = np.take_along_axis(value_mat, k_policy_mat, axis=-1).squeeze(axis=-1)
                    value_mat = np.take_along_axis(value_mat, l_policy_mat, axis=0).squeeze(axis=0)
                    if mqp:
                        value_mat = self.mqp(value_mat, local_vmat)

            old_value_mat = np.copy(value_mat)
            # (nl, nk, na)
            partial_q = np.max(u_mat + self.beta * trans_mat @ old_value_mat.T, axis=-1)
            k_policy_mat = np.argmax(u_mat + self.beta * trans_mat @ old_value_mat.T, axis=-1, keepdims=True)
            # (nk, na)
            value_mat = np.max(partial_q, axis=0)
            l_policy_mat = np.argmax(partial_q, axis=0, keepdims=True)

            if mqp:
                value_mat = self.mqp(value_mat, old_value_mat)
            
        return SOGVFIResult(self.alpha, self.beta, self.gamma, self.delta, self.rho,
                            self.sigma, self.eta, self.B, self.L, self.labor_choice,
                            a_grids, trans_mat, k_grids, value_mat, partial_q,
                            l_policy_mat.squeeze(), k_policy_mat.squeeze())
