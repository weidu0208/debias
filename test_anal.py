import os
import numpy as np
from lib.full.Fiducial_Sample_Generator import Biased_Fiducial_Sample_Generator, Debiasing

import numpy as np
import pandas as pd
from datetime import datetime
date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

pattern = 'test'  # --------------------------------------changable
folder = pattern+'_summary'+'/'+date
dics = next(os.walk(pattern))[1]

df_par = pd.read_csv(pattern+'/'+dics[0]+'/'+'df_par.csv')

B = np.loadtxt('data/images/'+pattern+'.txt')
p1 = B.shape[0]
p2 = B.shape[1]

r = df_par['r'][0]  # --------------------------------------changable
# --------------------------------------changable
sample_size = df_par['sample_size'][0]*len(dics)
# --------------------------------------changable
model_sel = df_par['model_sel'][0]

n = df_par['n'][0]  # --------------------------------------changable
X_seed = df_par['X_seed'][0]
y_seed = df_par['y_seed'][0]

X = np.random.RandomState(X_seed).normal(size=p1*p2*n).reshape((n, p1, p2))
y = np.einsum('nij,ij->n', X, B)+np.random.RandomState(y_seed).normal(size=n)

gfi_generator = Biased_Fiducial_Sample_Generator(
    X, y, r=r, sample_size=sample_size, alpha=df_par['alpha'][0])

for i in range(len(dics)):
    if i > 0:
        gfi_generator.biased_samples = np.append(gfi_generator.biased_samples, np.load(
            pattern+'/'+dics[i]+'/'+'biased_samples.npy'), axis=0)
        gfi_generator.biased_B = np.append(gfi_generator.biased_B, np.load(
            pattern+'/'+dics[i]+'/'+'samples_biased.npy'), axis=0)
        gfi_generator.U = np.append(gfi_generator.U, np.load(
            pattern+'/'+dics[i]+'/'+'U.npy'), axis=0)
    else:
        gfi_generator.biased_samples = np.load(
            pattern+'/'+dics[i]+'/'+'biased_samples.npy')
        gfi_generator.biased_B = np.load(
            pattern+'/'+dics[i]+'/'+'samples_biased.npy')
        gfi_generator.U = np.load(pattern+'/'+dics[i]+'/'+'U.npy')

result = Debiasing(gfi_generator, model_sel=model_sel)
result.anal_samples(ci_level=0.95, B=B, folder=folder)
