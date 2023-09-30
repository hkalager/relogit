#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:39:02 2022

This example illustrates how RE-Logit compares to Logit for a toy problem.

See the link below for details of this example:
https://blog.methodsconsultants.com/posts/bias-adjustment-for-rare-events-logistic-regression-in-r/

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.
Last review: 30/09/2023
"""

import numpy as np
from relogit import relogit
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
invnorm=norm.ppf
rnorm=np.random.normal
pnorm=norm.pdf



def gen_data(signal_to_noise:float=0.2, n:int=750, prop_ones:float=.03,
             train_sample_frac:float=.66,seed:int=12345):
    np.random.seed(seed=seed)
    beta_1 = (signal_to_noise)**.5
    
    x1=rnorm(size=n)
    
    intercept = invnorm(prop_ones, scale = (beta_1 ** 2 + 1)**.5)
    
    y_signal = beta_1 * x1
    
    y_star = intercept + y_signal + rnorm(size=n)
    
    y = (y_star >= 0).astype(int)
    

    true_prob = pnorm(y_signal + intercept)
    idx_train,idx_test = train_test_split(range(0,x1.shape[0]),
                                          train_size=train_sample_frac, 
                                                        random_state=seed)
    X_train=x1[idx_train]
    X_test=x1[idx_test]
    y_train=y[idx_train]
    y_test=y[idx_test]
    X_train=add_constant(X_train)
    X_test=add_constant(X_test)
    true_prob_train=true_prob[idx_train]
    true_prob_test=true_prob[idx_test]
    

    return X_train,y_train,X_test,y_test,true_prob_train,true_prob_test

def _main():
    
    X_train,y_train,X_test,y_test,true_prob_train,true_prob_test=gen_data()
    relogit_model=relogit(y_train,X_train)
    predicted_relogit,coeffs_unbiased,predicted_logit,coeff_biased=relogit_model.predict(X_test)
    logit_summary=relogit_model.base.summary()
    print(logit_summary)
    mae_logit=np.mean(np.abs(predicted_logit-true_prob_test))
    mae_relogit=np.mean(np.abs(predicted_relogit-true_prob_test))
    print('MAE='+str(np.round(mae_logit,4))+' for logit vs '+str(np.round(mae_relogit,4))+' for relogit')
    print('MAE_logit is '+str(np.round(mae_logit/mae_relogit,2))+' MAE_relogit')

    fig, ax = plt.subplots()
    ax.plot(predicted_logit,'-r',label='Prob Logit')
    ax.plot(predicted_relogit,'-b',label='Prob RE-Logit')
    ax.plot(true_prob_test,'-g',label='True Prob')
    ax.set_yscale('logit')
    ax.set_ylabel('predicted value')
    ax.legend(loc=4)
    plt.show()
    
    # logit_model=Logit(y_train,X_train).fit(disp=0)
    # predicted_logit=logit_model.predict(X_test)
    # coeffs_logit=logit_model.params
    
    return None

if __name__=="__main__":
    _main()
    

    
    




