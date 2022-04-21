from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import numpy as np
import multiprocessing as mp


class TensorRegression():
    def __init__(self, r, core_num=4):
        self.r = r
        self.core_num = core_num

    def risk(self, B, x, y):
        # theta: d by r by p_d --> B: pd
        # x:     n by pd
        #B = np.einsum('ri,rj->ij',theta[0],theta[1])
        y_hat = np.tensordot(x, B, axes=2)
        return np.mean((y-y_hat)**2)

    def fit(self, X, y, alpha, eps=1e-3, iter_max=10000):
        n = len(y)
        p1, p2 = X.shape[1:]
        d = len(X.shape)-1
        B1 = np.random.normal(size=self.r*p1).reshape((self.r, p1))
        B2 = np.random.normal(size=self.r*p2).reshape((self.r, p2))

        eps = 1
        t = 0
        l_new = 1
        while eps > 0 and t < iter_max and np.max(np.abs(B2)) > 1e-16:
            t += 1
            l_old = l_new

            X1 = np.einsum('nij,rj->nri', X, B2).reshape((n, -1))
            fit = Lasso(alpha=alpha/n, fit_intercept=False,
                        max_iter=10000).fit(X1, y)
            B1 = fit.coef_.reshape((self.r, p1))

            X2 = np.einsum('nij,ri->nrj', X, B1).reshape((n, -1))
            fit = Lasso(alpha=alpha/n, fit_intercept=False,
                        max_iter=10000).fit(X2, y)
            B2 = fit.coef_.reshape((self.r, p2))

            l_new = sum((y-X2.dot(B2.flatten()))**2)/2+alpha * \
                (np.sum(np.abs(B1))+np.sum(np.abs(B2)))
            eps = abs(l_new-l_old)
            B = np.einsum('ri,rj->ij', B1, B2)
        return {'theta': np.hstack((B1.flatten(), B2.flatten())), 'B': B}

    def alpha_err(self, alpha):
        kf = KFold(n_splits=self.k_fold, shuffle=False)
        return np.mean([self.risk(self.fit(self.X[train_index], self.y[train_index], alpha)['B'], self.X[test_index], self.y[test_index])
                        for train_index, test_index in kf.split(self.X)])

    def cv_fit(self, X, y, n_alpha=20, k_fold=10, eps=1e-3, iter_max=10000):
        self.X = X
        self.y = y
        self.n = len(y)
        self.k_fold = k_fold
        alpha = 10**np.arange(0, -2, -2/(n_alpha-1))*self.n

        if not self.core_num:
            kf = KFold(n_splits=k_fold, shuffle=False)
            k_index = [x for x in kf.split(X)]

            def cv(k, alpha):
                train_index, test_index = k_index[k]
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]

                B = self.fit(X_train, y_train, alpha, eps, iter_max)['B']
                return self.risk(B, X_test, y_test)
            cv_vec = np.vectorize(cv)

            def alpha_err(alpha):
                return np.mean(cv_vec(range(k_fold), alpha))

            alpha_err_vec = np.vectorize(alpha_err)
            alpha_err_val = alpha_err_vec(alpha)

        else:
            with mp.Pool(self.core_num) as pool:
                alpha_err_val = pool.map(self.alpha_err, list(alpha))

        alpha_sel = alpha[np.argmin(alpha_err_val)]

        return alpha_sel
