from lib.full.Fiducial_Sample_Generator import Biased_Fiducial_Sample_Generator, Debiasing
#from lib.full.TensorRegression import TensorRegression
import os
import numpy as np
import pandas as pd
from datetime import datetime
date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
print(f"{date}")

pattern = 'test'  # --------------------------------------changable
folder = pattern+'/'+date
B = np.loadtxt('data/images/'+pattern+'.txt')
p1 = B.shape[0]
p2 = B.shape[1]

if os.path.exists(pattern):
    dics = next(os.walk(pattern))[1]
    df_par = pd.read_csv(pattern+'/'+dics[0]+'/'+'df_par.csv')

    alpha = df_par['alpha'][0]
    r = df_par['r'][0]
    sample_size = df_par['sample_size'][0]
    model_sel = df_par['model_sel'][0]
    n = df_par['n'][0]
    X_seed = df_par['X_seed'][0]
    y_seed = df_par['y_seed'][0]
else:
    alpha = False
    r = 10            # --------------------------------------changable
    sample_size = 100  # --------------------------------------changable
    model_sel = True  # --------------------------------------changable

    n = 100           # --------------------------------------changable
    X_seed = 108
    y_seed = 206

X = np.random.RandomState(X_seed).normal(size=p1*p2*n).reshape((n, p1, p2))
y = np.einsum('nij,ij->n', X, B)+np.random.RandomState(y_seed).normal(size=n)

gfi_generator = Biased_Fiducial_Sample_Generator(
    X, y, r=r, sample_size=sample_size, alpha=alpha)
gfi_generator.get_samples()

result = Debiasing(gfi_generator, model_sel=model_sel)
result.anal_samples(ci_level=0.95, B=B, folder=folder)

dict_par = {'Dataset': pattern, 'p1': p1, 'n': n, 'r': r, 'sample_size': sample_size,
            'model_sel': model_sel, 'X_seed': X_seed, 'y_seed': y_seed, 'alpha': result.alpha}
df_par = pd.DataFrame(data=dict_par, index=['par'])
df_par.to_csv(folder + '/' + 'df_par.csv')

np.save(folder+'/x.npy', result.x)
np.save(folder+'/y.npy', result.y)
np.save(folder+'/biased_samples.npy', result.biased_samples)
np.save(folder+'/samples_biased.npy', result.samples_biased)
np.save(folder+'/U.npy', result.U)
