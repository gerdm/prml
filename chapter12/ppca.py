import numpy as np
from numba import int32, float32
from numba import jit
from numba.experimental import jitclass


class ProbabilisticPCA:
    """
    Implementation of the Probabilistic PCA (PPCA) algorithm
    in which we assume that a datapoint in drawn from a latent
    variable z R^M, shifted via a parameter mu, and then projected
    onto R^D via the projecting matrix W plus some additive noise
    sigma2. Hence
    
        x = Wz + mu + eps; eps ~ N(0, sigma2)
        
    """
    def __init__(self, X, M, W_init=None, sigma2_init=None, seed=None):
        self.seed = seed
        self.M = M
        self.N, self.D = X.shape
        self.X = X
        self.Xbar = X - X.mean(axis=0)
        self.Xnorm2 = (self.Xbar ** 2).sum()
        self.W = self._initialize_W(W_init)
        self.sigma2 = self._initialize_sigma2(sigma2_init)
        self.E_zn, self.E_znzn = self._compute_expectations()
        
    
    def _initialize_W(self, W_init):
        np.random.seed(self.seed)
        if W_init is None:
            return np.random.randn(self.D, self.M)
        else:
            return W_init
        
        
    def _initialize_sigma2(self, sigma2_init):
        np.random.seed(self.seed)
        if sigma2_init is None:
            return np.random.rand()
        else:
            return sigma2_init
        
        
    def missing_expectation(self, Xsamp):
        """
        Compute the expected value of a vector
        with missing entries
        """
        mu_final = Xsamp.copy()
        # missing columns of an observation
        ix_miss = np.isnan(Xsamp)
        # Number of missing dimensions
        D_miss = ix_miss.sum()

        X_miss, X_obs = Xsamp[ix_miss], Xsamp[~ix_miss]
        W_miss, W_obs = self.W[ix_miss], self.W[~ix_miss]    
        mu_miss, mu_obs = self.mu[ix_miss], self.mu[~ix_miss]

        C_obs = W_obs @ W_obs.T + sigma2 * np.identity(self.D - D_miss)
        ix_miss_obs = W_miss @ W_obs.T

        mu_miss_giv_obs = ix_miss_obs @ np.linalg.inv(C_obs) @ (X_obs - mu_obs).reshape(-1, 1)

        mu_final[ix_miss] = mu_miss_giv_obs.ravel()
        return mu_final


    def _compute_missing_expectation(self):
        Xfinal = self.X[:]
        missing_ix = np.isnan(Xfinal).sum(axis=1) > 0
        missing_ix = np.where(missing_ix)[0]

        for ix, Xsamp in zip(missing_ix, Xfinal[missing_ix]):
            Ex = self.missing_expectation(Xsamp)
            Xfinal[ix] = Ex

        return Xfinal


    def norm2_missing_sample(Xsamp, mu, W, sigma2):
        D = Xsamp.shape[0]
        ix_miss = np.isnan(Xsamp)
        D_miss = ix_miss.sum()

        X_miss, X_obs = Xsamp[ix_miss], Xsamp[~ix_miss]
        W_miss, W_obs = W[ix_miss], W[~ix_miss]    
        mu_miss, mu_obs = mu[ix_miss], mu[~ix_miss]

        C_miss = W_miss @ W_miss.T + sigma2 * np.identity(D_miss)
        C_obs = W_obs @ W_obs.T + sigma2 * np.identity(D - D_miss)
        C_miss_obs = W_miss @ W_obs.T


        expected_norm = (X_obs - mu_obs) @ (X_obs - mu_obs)
        expected_norm = expected_norm + np.trace(C_miss - C_miss_obs @ np.linalg.inv(C_obs) @ C_miss_obs.T)

        return expected_norm

    def norm2_missing_sum(X, mu, W, sigma2):
        norm2_sum = 0
        for Xsamp in X:
            norm2_sum = norm2_sum + norm2_missing_sample(Xsamp, mu, W, sigma2)
        return norm2_sum


    def _compute_expectations(self):
        """
        :::E-Step:::

        Compute ∀n in the dataset. E[zn], E[zn@zn.T]
        """
        Mz = self.W.T @ self.W + self.sigma2 * np.identity(self.M)
        i_Mz = np.linalg.inv(Mz)

        E_zn = np.einsum("ij,mj,km->ik", i_Mz, self.W, self.Xbar, optimize=True)
        E_znzn = np.einsum("in,kn->kin", E_zn, E_zn, optimize=True) + i_Mz[..., None] / self.sigma2

        return E_zn, E_znzn
    
    
    def update_sigma2(self):
        T2 = - 2 * np.einsum("in,mi,nm->", self.E_zn, self.W, self.Xbar, optimize=True)
        T3 = np.einsum("ijn,mi,mj->", self.E_znzn, self.W, self.W, optimize="optimal")
        sigma2_new = (self.Xnorm2 + T2 + T3) / (self.N * self.D)
        
        return sigma2_new
    
    
    def update_W(self):
        T1 = np.einsum("nd,mn,jmn->dj", self.Xbar,
                       self.E_zn, self.E_znzn, optimize="optimal")
        T2 = self.E_znzn.sum(axis=-1)
        
        W_new = (T1 @ np.linalg.inv(T2))
        
        return W_new
    
    
    def _update_parameters(self):
        """
        :::M-step:::
        
        Compute one step of the EM-algorithm
        (assuming that the E-step has already been)
        computed
        """
        sigma2_new = self.update_sigma2()
        W_new = self.update_W()
        
        return sigma2_new, W_new
        
        
    def data_log_likelihood(self):
        """
        Compute the expected value of the complete-data log-likelihood
        with respect to the posterior distribution of the latent
        variable
        """
        Q = -(self.N * self.M * np.log(2 * np.pi) / 2 +
              np.einsum("iin->", self.E_znzn) + 
              D * np.log(2 * np.pi * self.sigma2) / 2 +
              self.Xnorm2 / (2 *  self.sigma2) +
              -np.einsum("in,mi,nm->", self.E_zn, self.W, self.Xbar, optimize=True) / (self.sigma2) + 
              np.einsum("ijn,mi,mj->", self.E_znzn, self.W, self.W, optimize="optimal") / (self.sigma2)
              )
        
        return Q
    
    def project(self):
        Mz = self.W.T @ self.W + self.sigma2 * np.identity(self.M)
        Xproj = inv(Mz) @ self.W.T @ self.Xbar.T
        return Xproj
    
    def EM_step(self):
        # E-step
        self.E_zn, self.E_znzn = self._compute_expectations()
        # M-step
        self.sigma2, self.W = self._update_parameters()
        
        Q = self.data_log_likelihood()
        return Q