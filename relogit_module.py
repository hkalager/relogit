#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a Python class to replicate the Rare Event Logit of 
King and Zeng (2001). 

The original paper is available: https://gking.harvard.edu/files/abs/0s-abs.shtml
    
The R software is available at: https://zeligproject.org/

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 01/07/2022
"""

import numpy as np
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
dot=np.dot
t=np.transpose
inv=np.linalg.inv
diag=np.diag
exp=np.exp


# Define necessary modules

class relogit:
    
    __version__='1.0.0'
    def __init__(self, Y,X,add_const=False):
        '''
        
        Parameters
        ----------
        Y : array_like
            A 1-d endogenous response variable. See statsmodels guidance  
        X : array_like
            A nobs x k array where nobs is the number of observations and k is 
            the number of regressors. An intercept is added by setting add_const
            to True.
        add_const : Boolean, optional
            Whether to add a constant into X. The default is False.

        Returns
        -------
        None.

        '''
        if add_const is not False:
            X=add_constant(X)
        base_model=Logit(Y,X)
        fitted_model=base_model.fit(disp=0)
        params=fitted_model.params
        pred=fitted_model.predict()
        w=diag(pred*(1-pred))
        #Q <- X_matrix %*% solve(t(X_matrix) %*% W %*% X_matrix) %*% t(X_matrix)
        inner_1=inv(dot(dot(t(X),w),X))
        q=dot(dot(X,inner_1),t(X))
        #e <- 0.5 * diag(Q) * (2 * pred - 1)
        e=0.5*diag(q)*(2*pred-1)
        #bias <- (solve(t(X_matrix) %*% W %*% X_matrix) %*% t(X_matrix) %*% W %*% e)
        bias=dot(dot(dot(inv(dot(dot(t(X),w),X)),t(X)),w),e)
        self.base=base_model
        self.X=X
        self.Y=Y
        self.unbiased_params=params-bias
        self.unbiased_pred=base_model.predict(params=self.unbiased_params)
        self.a_c=add_const
    
    def predict(self,X_new=[]):
        if len(X_new)>0:
            X=X_new
            if self.a_c is True:
                X=add_constant(X)
            self.X=X
        else:
            X=self.X
        unbiased_params=self.unbiased_params
        unbiased_pred=(1+exp(-1*dot(X,unbiased_params)))**-1
        self.unbiased_pred=unbiased_pred
        return unbiased_pred
        
