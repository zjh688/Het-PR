__author__ = 'Haohan Wang'

import scipy.optimize as opt
from sklearn.preprocessing import normalize
import numpy as np

import sys
sys.path.append('../')

from models.helpingMethods import *


class BaseModel:

    def __init__(self, numintervals=1000, ldeltamin=-50, ldeltamax=50, alpha=0.05, fdr=False, lowRankFit=False, tau=None):
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.alpha = alpha
        self.fdr = fdr
        self.lowRankFit = lowRankFit
        self.tau = tau

    def correctData(self, X, y, K, Kva=None, Kve=None):
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'

        if y.ndim == 1:
            y = np.reshape(y, (n_s, 1))

        if self.lowRankFit:
            S, U, ldelta0 = self.train_nullmodel_lowRankFit(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                            ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax,
                                                            p=n_f)
        else:
            S, U, ldelta0 = self.train_nullmodel(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                 ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax, p=n_f)

        delta0 = np.exp(ldelta0)
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = np.sqrt(Sdi)
        SUX = np.dot(U.T, X)
        # SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
        for i in range(n_f):
            SUX[:, i] = SUX[:, i] * Sdi_sqrt.T
        SUy = np.dot(U.T, y)
        SUy = SUy * np.reshape(Sdi_sqrt, (n_s, 1))

        SUX = normalize(SUX, axis=0, norm='l2')
        SUy = normalize(SUy, axis=0, norm='l2')

        return SUX, SUy


    def selectValues(self, Kva):
        r = np.zeros_like(Kva)
        n = r.shape[0]
        tmp = self.rescale(Kva)
        ind = 0
        for i in range(n / 2, n - 1):
            if tmp[i + 1] - tmp[i] > 1.0 / n:
                ind = i + 1
                break
        r[ind:] = Kva[ind:]
        r[n - 1] = Kva[n - 1]
        return r

    def train_nullmodel(self, y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm', p=1):
        ldeltamin += scale
        ldeltamax += scale

        if S is None or U is None:
            S, U = linalg.eigh(K)

        Uy = np.dot(U.T, y)

        # grid search
        nllgrid = np.ones(numintervals + 1) * np.inf
        ldeltagrid = np.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        for i in np.arange(numintervals + 1):
            nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S) # the method is in helpingMethods

        nllmin = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        for i in np.arange(numintervals - 1) + 1:
            if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                              (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                              full_output=True)
                if nllopt < nllmin:
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt

        return S, U, ldeltaopt_glob

    def train_nullmodel_lowRankFit(self, y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm', p=1):
        ldeltamin += scale
        ldeltamax += scale

        if S is None or U is None:
            S, U = linalg.eigh(K)

        Uy = np.dot(U.T, y)

        S = self.selectValues(S)
        nllgrid = np.ones(numintervals + 1) * scipy.inf
        ldeltagrid = np.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        for i in np.arange(numintervals + 1):
            nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)  # the method is in helpingMethods

        nllmin = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        for i in np.arange(numintervals - 1) + 1:
            if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                              (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                              full_output=True)
                if nllopt < nllmin:
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt

        return S, U, ldeltaopt_glob
