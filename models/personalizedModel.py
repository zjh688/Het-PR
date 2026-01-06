__author__ = 'Haohan Wang'

import numpy as np
from model.BaseModel import BaseModel
from model.LMM import LinearMixedModel

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error as mse

from tqdm import tqdm
import sys

class PersonalizedThroughMixedModel():
    def __init__(self, mode_regressionModel='lmm', regWeight=0):

        self.mode_regressionModel = mode_regressionModel
        self.regWeight = regWeight

        self.corrector = BaseModel()

        if self.mode_regressionModel == 'lmm':
            self.regressor = LinearMixedModel(fdr=False)
        elif self.mode_regressionModel == 'lr':
            self.regressor = LinearRegression()
        elif self.mode_regressionModel == 'lasso':
            self.regressor = Lasso(alpha = self.regWeight)
        else:
            sys.exit()

    def fit(self, X, y, C):

        K = np.zeros([X.shape[0], X.shape[0]])
        D = np.diag(np.dot(X, X.T))

        # print (D)

        B = np.zeros_like(X)
        P = np.zeros_like(X)

        count = 0

        for i in tqdm(range(X.shape[0])):
            sig = np.mean(np.square(C - C[i]), 1) # sum follows the derivation in the notes,
                                                 # but an annoying fact is that this scales with the dimension of C
                                                 # so maybe a mean here now?


            ### save computation when the error terms are the same
            idx = np.where(sig==0)[0][0]

            if idx < i:
                B[i] = B[idx]
                P[i] = P[idx]
            else:
                diag = sig*D

                # print (i)

                np.fill_diagonal(K, diag)

                Xc, Yc = self.corrector.correctData(X, y, K)

                if self.mode_regressionModel == 'lmm':
                    self.regressor.setK(np.dot(Xc, Xc.T))
                    self.regressor.fit(Xc, Yc)
                    beta = self.regressor.getBeta()
                    P[i] = np.exp(-self.regressor.getNegLogP())
                else:
                    self.regressor.fit(Xc, Yc)
                    beta = self.regressor.coef_

                B[i] = beta

            count += 1
            # if count >= 2:
            #     break

        return B, P

if __name__ == '__main__':
    pass
    # from dataGeneration.generationData import generatePersonalizedDataIID
    #
    # X, B, y, C = generatePersonalizedDataIID(n=100, p=50, blockNum=10)
    #
    # pl = PersonalizedThroughMixedModel()
    # B_1, P = pl.fit(X, y, np.copy(C))
    #
    # from matplotlib import pyplot as plt
    #
    # plt.imshow(B)
    # plt.show()
    #
    # plt.imshow(B_1)
    # plt.show()
    #
    # plt.imshow(P)
    # plt.show()
    #
    # print ('-----------')
    # print (B)
    # print ('-----------')
    # print (B_1)
    #
    # print (P)
    #
    # print (mse(B, B_1))