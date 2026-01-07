__author__ = 'Haohan Wang'

import scipy.optimize as opt
import numpy.linalg as linalg

import sys
# sys.path.append('../')

from models.helpingMethods import *

class LinearMixedModel:
    def __init__(self, numintervals=100, ldeltamin=-5, ldeltamax=5, scale=0, alpha=0.05, fdr=False):
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.scale = scale
        self.alpha = alpha
        self.fdr = fdr
        self.ldelta0 = None
        self.beta = []

    def estimateDelta(self, y):
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0], 1))
        self.y = y
        self.S, self.U, self.ldelta0= self.train_nullmodel(y, self.K)
        print(self.ldelta0)

    def setK(self, K):
        self.K = K

    def setLDelta(self, delta0):
        self.ldelta0=delta0

    def fit(self, X, y):

        self.beta = []
        self.ldelta0 = None

        [n_s, n_f] = X.shape
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0], 1))
        self.y = y
        self.S, self.U, ldelta0= self.train_nullmodel(y, self.K)

        if self.ldelta0 is None:
            self.ldelta0 = ldelta0

        X0 = np.ones(len(self.y)).reshape(len(self.y), 1)

        delta0 = np.exp(self.ldelta0)
        Sdi = 1. / (self.S + delta0)
        Sdi_sqrt = np.sqrt(Sdi)
        SUX = np.dot(self.U.T, X)
        SUX = SUX * np.tile(Sdi_sqrt, (n_f, 1)).T
        SUy = np.dot(self.U.T, self.y)
        SUy = SUy * np.reshape(Sdi_sqrt, (n_s, 1))

        self.SUX = SUX
        self.SUy = SUy
        SUX0 = np.dot(self.U.T, X0)
        self.SUX0 = SUX0 * np.tile(Sdi_sqrt, (1, 1)).T
        self.X0 = X0
        self.neg_log_p = self.hypothesisTest(self.SUX, self.SUy, X, self.SUX0, self.X0)
        return self.neg_log_p

    def fdrControl(self):
        tmp = np.exp(-self.neg_log_p)
        tmp = sorted(tmp)
        threshold = 1e-8
        n = len(tmp)
        for i in range(n):
            if tmp[i] < (i+1)*self.alpha/n:
                threshold = tmp[i]
        self.neg_log_p[self.neg_log_p<-np.log(threshold)] = 0

    def getNegLogP(self):
        if not self.fdr:
            return self.neg_log_p
        else:
            self.fdrControl()
            return self.neg_log_p

    def hypothesisTest(self, UX, Uy, X, UX0, X0):
        [m, n] = X.shape
        p = []
        for i in range(n):
            if UX0 is not None:
                UXi = np.hstack([UX0, UX[:, i].reshape(m, 1)])
                XX = matrixMult(UXi.T, UXi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, UXi.T), Uy)
                Uyr = Uy - matrixMult(UXi, beta)
                Q = np.dot(Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            else:
                Xi = np.hstack([X0, UX[:, i].reshape(m, 1)])
                XX = matrixMult(Xi.T, Xi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, Xi.T), Uy)
                Uyr = Uy - matrixMult(Xi, beta)
                Q = np.dot(Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            self.beta.append(beta[1][0])
            ts, ps = tstat(beta[1], XX_i[1, 1], sigma, 1, m)
            if -1e100 < ts < 1e100:
                p.append(ps)
            else:
                p.append(1)
        p = np.array(p)
        self.beta = np.array(self.beta)
        return -np.log(p)

    def train_nullmodel(self, y, K, S=None, U=None, numintervals=500):
        self.ldeltamin += self.scale
        self.ldeltamax += self.scale

        if S is None or U is None:
            S, U = linalg.eigh(K)

        Uy = np.dot(U.T, y)

        # grid search
        nllgrid = np.ones(numintervals + 1) * np.inf
        ldeltagrid = np.arange(numintervals + 1) / (numintervals * 1.0) * (self.ldeltamax - self.ldeltamin) + self.ldeltamin
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

    def getBeta(self):
        return self.beta
