# relogit
 A simple wrapper class to estimate a Rare Event Logit model as in King and Zeng (2001)

# Warning:
* This is an exploratory study with no profitable strategy in sight.
* Provision of codes is not an investment advice.
* All codes and analyses are subject to error.


# Replication:

– In a terminal window install the requirements as:

` pip install -r requirements.txt`

– In Python environment import the relogit module as:

` from relogit_module import relogit`

  You will be asked to enter your credentials for accessing WRDS at this stage.

– Specify the module using `study_period` and `horizon` as:

` a=relogit(Y, X, add_const=False)`



# Packages 
The following packages in Python to run these scripts:
- Numpy
- statsmodels
