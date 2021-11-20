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

def binary_extranodal_sites(number):
    if number >= 1:
        return 1
    if number == 0:
        return 0
    
def get_mean_ipi(number):
    if number>5:
        digits = [int(d) for d in str(number)]
        return sum(digits)/len(digits)
    else:
        return number
    
def get_min_ipi(number):
    if number>5:
        digits = [int(d) for d in str(number)]
        return min(number)
    else:
        return number
    
def get_max_ipi(number):
    if number>5:
        digits = [int(d) for d in str(number)]
        return max(number)
    else:
        return number
    
def impute_ipi_group(number):
    if 0<=number<2:
        return 'Low'
    elif 2<=number<4:
        return 'Intermediate'
    else:
        return 'High'