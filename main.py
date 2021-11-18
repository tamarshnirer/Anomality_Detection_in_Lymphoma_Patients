import pandas
import sklearn
import numpy

def gender_encoder(gender):
    if gender=='M':
        return 0
    if gender=='F':
        return 1
    
def survival_encoder(survival):
    if survival=="Yes":
        return 1
    if survival=="No":
        return 0
    
