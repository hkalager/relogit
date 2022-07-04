# relogit
 A simple wrapper class to estimate a Rare Event Logit model of King and Zeng (2001) in Python.

# Warning:
* All codes and analyses are subject to error.

# User guide:

– In a terminal window install the requirements as:

` pip install -r requirements.txt`

– In Python environment import the relogit module as:

` from relogit_module import relogit`

  You will be asked to enter your credentials for accessing WRDS at this stage.

– Specify the function using the following variables:
`
Y : array_like
            A 1-d endogenous response variable. See statsmodels guidance
X : array_like
            A nobs x k array where nobs is the number of observations and k is 
            the number of regressors. An intercept is added by setting add_const
            to True.
add_const : Boolean, optional
            Whether to add a constant into X. The default is False.
disp : Boolean, optional
            Whether to display details for fitting. The default is False.
            See statsmodels guidance  


– Train a RE-Logit model by ` relogit_model=relogit(Y, X, *optional keywords*)`

– Get unbiased probability estimation, unbiased coefficients 

`predicted_relogit,coeffs_unbiased,predicted_logit,coeff_biased=relogit_model.predict(X_test)`

– For more see the accompanying example script `vignette.py`

# Packages 
The following packages are required to use this module:
- Numpy
- statsmodels
