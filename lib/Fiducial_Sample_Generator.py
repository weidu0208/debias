import matplotlib.pyplot as plt
import multiprocessing as mp
from lib.TensorRegression import TensorRegression
from lib.convert_to_preferred_format import convert_to_preferred_format
import time
import os
import numpy as np
from numpy import random
import pandas as pd

import jax.numpy as jnp
from jax import jacfwd, jacrev, vmap, jit, soft_pmap
from jax.config import config
config.update('jax_platform_name', 'cpu')


class Biased_Fiducial_Sample_Generator():
    def __init__(self, x, y, r, sample_size, alpha=False, core_num=4):
        self.x = x
        self.y = y
        self.n = len(y)
        self.d = len(x.shape)-1
        self.p1 = x.shape[1]
        self.p2 = x.shape[2]
        self.r = r

        self.sample_size = sample_size

        U = np.random.normal(size=(self.sample_size, self.n))
        self.U = U

        self.core_num = core_num

        if not alpha:
            cv_tis = time.time()
            ftr = TensorRegression(self.r)
            self.alpha = ftr.cv_fit(self.x, self.y)
            cv_tic = time.time()
            print('FTR_cv with 10 folds needs %s time and results in alpha as %f' %
                  (convert_to_preferred_format(cv_tic-cv_tis), self.alpha))
        else:
            self.alpha = alpha

    def biased_sample_generator(self, ui):
        ft = TensorRegression(r=self.r, core_num=False).fit(
            self.x, self.y-ui, self.alpha)
        biased_sample = ft['theta']
        B_biased = ft['B']
        return {'biased_sample': biased_sample, 'B_biased': B_biased}

    def get_samples(self):
        gfi_tis = time.time()
        with mp.Pool(self.core_num) as pool:
            biased_samples = pool.map(
                self.biased_sample_generator, list(self.U))
            pool.close()
        gfi_tic = time.time()
        print('Generating %d biased samples with %d cores needs %s time' %
              (self.sample_size, self.core_num, convert_to_preferred_format(gfi_tic-gfi_tis)))
        self.biased_samples = [sample['biased_sample']
                               for sample in biased_samples]
        self.biased_B = [sample['B_biased'].flatten()
                         for sample in biased_samples]


class Debiasing():
    def __init__(self, Biased_Fiducial_Sample_Generator, model_sel=True):

        self.model_sel = model_sel
        self.x = Biased_Fiducial_Sample_Generator.x
        self.y = Biased_Fiducial_Sample_Generator.y
        self.n = Biased_Fiducial_Sample_Generator.n
        self.d = Biased_Fiducial_Sample_Generator.d
        self.p1 = Biased_Fiducial_Sample_Generator.p1
        self.p2 = Biased_Fiducial_Sample_Generator.p2
        self.r = Biased_Fiducial_Sample_Generator.r

        self.sample_size = Biased_Fiducial_Sample_Generator.sample_size
        self.alpha = Biased_Fiducial_Sample_Generator.alpha

        self.U = jnp.array(Biased_Fiducial_Sample_Generator.U)

        self.core_num = Biased_Fiducial_Sample_Generator.core_num

        self.biased_samples = jnp.array(
            Biased_Fiducial_Sample_Generator.biased_samples)
        self.samples_biased = jnp.array(
            Biased_Fiducial_Sample_Generator.biased_B)

        self.jacrev_risk = jacrev(self.risk)
        self.hessian_risk = jacfwd(self.jacrev_risk)
        gfi_tis_un = time.time()
        if self.model_sel == True:
            #self.samples_unbiased = soft_pmap(self._debias,in_axes=(0,0))(self.biased_samples,self.U).block_until_ready()
            self.samples_unbiased = jnp.array(list(
                map(self._debias, range(self.sample_size))))
        else:
            self.samples_unbiased = jit(vmap(self._debias_full, (0, 0), 0), backend='cpu')(
                self.biased_samples, self.U).block_until_ready()
        gfi_tic_un = time.time()
        print('Debiasing %d biased samples needs %s time' %
              (self.sample_size, convert_to_preferred_format(gfi_tic_un-gfi_tis_un)))
        #self.samples_unbiased = soft_pmap(self._debias,in_axes=(0,0))(self.biased_samples,self.U)

    def risk(self, theta, x, ui):
        B1 = theta[0:self.p1*self.r].reshape((self.r, self.p1))
        B2 = theta[self.p1*self.r:].reshape((self.r, self.p2))
        # theta: d by r by p_d --> B: pd
        # x:     n by pd
        B = jnp.einsum('ri,rj->ij', B1, B2)
        y_hat = jnp.tensordot(x, B, axes=2)+ui
        return jnp.mean((self.y-y_hat)**2)

    def _debias(self, i):
        biased_sample = self.biased_samples[i]
        grad_risk = self.jacrev_risk(biased_sample, self.x, self.U[i])
        hessian = self.hessian_risk(biased_sample, self.x, self.U[i])
        indice = jnp.nonzero(biased_sample)[0]

        bias = jnp.dot(jnp.linalg.inv(
            hessian[indice][:, indice]), grad_risk[indice])

        unbiased_sample = jnp.array(biased_sample.copy())
        unbiased_sample = unbiased_sample.at[indice].set(
            unbiased_sample[indice] - bias)

        B1 = unbiased_sample[0:self.p1*self.r].reshape((self.r, self.p1))
        B2 = unbiased_sample[self.p1*self.r:].reshape((self.r, self.p2))
        B_unbiased = jnp.einsum('ri,rj->ij', B1, B2)
        return B_unbiased.flatten()

    def _debias_full(self, biased_sample, ui):
        grad_risk = self.jacrev_risk(biased_sample, self.x, ui)
        hessian = self.hessian_risk(biased_sample, self.x, ui)
        bias = jnp.dot(jnp.linalg.inv(
            hessian), grad_risk)

        unbiased_sample = biased_sample-bias

        B1 = unbiased_sample[0:self.p1*self.r].reshape((self.r, self.p1))
        B2 = unbiased_sample[self.p1*self.r:].reshape((self.r, self.p2))
        B_unbiased = jnp.einsum('ri,rj->ij', B1, B2)
        return B_unbiased.flatten()

    def anal_samples(self, ci_level, B, folder):
        q = (1-ci_level)/2
        ind_tr = jnp.nonzero(B.flatten())

        B_biased_mean = jnp.mean(
            self.samples_biased, axis=0).reshape((self.p1, self.p2))
        B_unbiased_mean = jnp.mean(
            self.samples_unbiased, axis=0).reshape((self.p1, self.p2))

        B_biased_median = jnp.median(
            self.samples_biased, axis=0).reshape((self.p1, self.p2))
        B_unbiased_median = jnp.median(
            self.samples_unbiased, axis=0).reshape((self.p1, self.p2))

        lower_biased = jnp.quantile(self.samples_biased, q=q, axis=0)
        upper_biased = jnp.quantile(self.samples_biased, q=1-q, axis=0)

        prop_biased = (B.flatten() >= lower_biased) * \
            (B.flatten() <= upper_biased)
        width_biased = upper_biased - lower_biased

        prop_biased_sig = prop_biased[ind_tr]
        width_biased_sig = width_biased[ind_tr]

        lower_unbiased = jnp.quantile(self.samples_unbiased, q=q, axis=0)
        upper_unbiased = jnp.quantile(self.samples_unbiased, q=1-q, axis=0)

        prop_unbiased = (B.flatten() >= lower_unbiased) * \
            (B.flatten() <= upper_unbiased)
        width_unbiased = upper_unbiased - lower_unbiased

        prop_unbiased_sig = prop_unbiased[ind_tr]
        width_unbiased_sig = width_unbiased[ind_tr]

        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = folder + '/' + 'B'+'.png'
        plt.imshow(B, cmap=plt.cm.binary)
        plt.axis('off')
        plt.savefig(filename, dpi=300)
        filename = folder + '/' + 'biased_' + 'mean'+'.png'
        plt.imshow(B_biased_mean, cmap=plt.cm.binary)
        plt.axis('off')
        plt.savefig(filename, dpi=300)
        filename = folder + '/' + 'biased_' + 'median'+'.png'
        plt.imshow(B_biased_median, cmap=plt.cm.binary)
        plt.axis('off')
        plt.savefig(filename, dpi=300)
        filename = folder + '/' + 'unbiased_' + 'mean'+'.png'
        plt.imshow(B_unbiased_mean, cmap=plt.cm.binary)
        plt.axis('off')
        plt.savefig(filename, dpi=300)
        filename = folder + '/' + 'unbiased_' + 'median'+'.png'
        plt.imshow(B_unbiased_median, cmap=plt.cm.binary)
        plt.axis('off')
        plt.savefig(filename, dpi=300)

        ans = {'mean_rmse': [np.mean((B_biased_mean-B)**2)**0.5, np.mean((B_unbiased_mean-B)**2)**0.5],
               'median_rmse': [np.mean((B_biased_median-B)**2)**0.5, np.mean((B_unbiased_median-B)**2)**0.5],
               'prop_of_covered': [np.mean(prop_biased), np.mean(prop_unbiased)],
               'ci_width': [np.mean(width_biased), np.mean(width_unbiased)],
               'prop_of_covered_sig': [np.mean(prop_biased_sig), np.mean(prop_unbiased_sig)],
               'ci_width_sig': [np.mean(width_biased_sig), np.mean(width_unbiased_sig)]}

        df = pd.DataFrame(data=ans, index=['biased', 'unbiased'])
        df.to_csv(folder + '/' + 'df.csv', index=True)
        if self.model_sel:
            print('Dataset: %s \nn = %3d, p1 = p2 = %2d, r = %2d, target_level = %.2f with submodel' % (
                folder, self.n, self.p1, self.r, ci_level))
        else:
            print('Dataset: %s \nn = %3d, p1 = p2 = %2d, r = %2d, target_level = %.2f with full model' % (
                folder, self.n, self.p1, self.r, ci_level))
        print(df)
