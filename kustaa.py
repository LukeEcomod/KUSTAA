# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:41:20 2018

@author: slauniai
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Define parameters for calculation
"""

from export_coefficients import default_coeff as coeff

folder = r'c:\Repositories\Kustaa\data'
# annual operations and other pollution sources
datafile = r'KUSTAA_Lahtotiedot_Esimerkkilaskelma.csv'

area = 6000.0 # catchment land area ha
lakes = 100.0 # water bodies within catchment ha

# import annual loading source data from csv-file, replace NaN with 0.0
data = pd.read_csv(os.path.join(folder, datafile), sep=';', header='infer')
data = data.fillna(0.0)

# relative uncertainties of loading areas are extracted from 1st row and
# converted to dict
err = data[data.columns].iloc[0]
err = err.drop('Year').to_dict() 
err['Background'] = 5.0
err['Deposition'] = 2.0

# now drop that row from data
data = data.drop(index=0)
data.index = data['Year'].values.astype(int)
data = data.drop(columns='Year')


class Kustaa():
    """
    Defines Kustaa -model for computing point-source and diffuse loading for a
    given catchment
    """
    
    def __init__(self, coeffs, area, lakes, err, data, scale=0.4, fmodel='Kalle'):
        """
        coeffs - export coefficients (dict)
        area -   catchment land area (ha, float)
        lakes -  catchemnt water bodies area (ha, float)
        err -    relative uncertainty of areas (%, dict, float)
        data -   annual loading source data (pd.DataFrame)
        scale -  if min and max are not given for coeffs, then they are estimated
                 from scale (float)
        fmodel - 'Kalle' uses Finer et al. 2010 (Suomen Ymparisto) - method (str)
        """              
        self.scale = scale  # (min - max) = scale * ave; used in case min-max not given 
        self.area = area    # catchment area
        self.lakes = lakes  # area of lakes and rivers within catchment
        self.area_err = err # relative uncertainties of areas (dict)
        self.C = coeffs     # export coefficients (dict)
        
        self.ForestryModel = fmodel # forestry model version
        if self.ForestryModel=='Kalle':
            from export_coefficients import forestry_relative as rcoeff  
            self.rC = rcoeff    # to account temporal decay of forestry loading (dict)
        
        # check that all cols in 'data' have export coeff values
        self.groups = list(coeffs.keys()) # groups
        k = [self.groups]
        for j in self.groups:
            k.extend(list(coeffs[j].keys()))
        
        result =  all(elem in k  for elem in data.columns.tolist())
        
        if result is False:
            raise Exception('KUSTAA Error: all columns in data must have'  \
                  + ' corresponding export coefficient value')
  
        if data.index[0] < data.index[-1]:
            raise Exception('KUSTAA Error: annual data must be given in ' \
                            + 'descending order (last year at top)')
        
        self.data = data.copy()
        self.data['Background'] = self.area
        self.data['Deposition'] = self.lakes
        
        # set up results dataframes. for each load source save expectation value and variance
        k = data.columns.tolist()
        
        self.N_load = pd.DataFrame(index=data.index, columns=k)
        self.N_var = pd.DataFrame(index=data.index, columns=k)
        
        self.P_load = pd.DataFrame(index=data.index, columns=k)
        self.P_var = pd.DataFrame(index=data.index, columns=k)
        
        self.SS_load = pd.DataFrame(index=data.index, columns=k)
        self.SS_var = pd.DataFrame(index=data.index, columns=k)
        
        self.results = []

    def compute_loads(self):
        """
        computes annual loads and their std's. Assumes all loading takes
        place on the year of operation
        """
        print('*** Kustaa - computing annual loads ***')
        
        for g in self.groups: # groups ['Natural', 'Forestry', 'Agriculture',...]
            s = list(self.C[g].keys()) # loading sources within group
            s = [x for x in s if x in list(self.data.columns)] # loop only those in data

            for k in s: 
                print(k)
                A = self.data[k] # area or unit
                dA = self.area_err[k] / 100.0 * A # area error
                cn = self.C[g][k]['N'] # export coeff
                cp = self.C[g][k]['P']
                cs = self.C[g][k]['SS']

                if (g == 'Forestry' and self.ForestryModel == 'Kalle'):

                    # print('compute forestry')
                    # account for temporal decay of loading
                    cn = np.array([np.multiply(m, self.rC[k]['N']) for m in cn])
                    cp = np.array([np.multiply(m, self.rC[k]['P']) for m in cp])
                    cs = np.array([np.multiply(m, self.rC[k]['SS']) for m in cs])
                    
                    self.N_load[k], self.N_var[k] = annual_forestry_load(cn, A, dA)
                    self.P_load[k], self.P_var[k] = annual_forestry_load(cp, A, dA)
                    self.SS_load[k], self.SS_var[k] = annual_forestry_load(cs, A, dA)

                else:

                    self.N_load[k], self.N_var[k] = annual_load(cn, A, dA, scale=self.scale)
                    self.P_load[k], self.P_var[k] = annual_load(cp, A, dA, scale=self.scale)
                    self.SS_load[k], self.SS_var[k] = annual_load(cs, A, dA, scale=self.scale)
            
            # sum means and variances to group level
            self.N_load[g] = self.N_load[s].sum(axis=1, skipna=True)
            self.P_load[g] = self.P_load[s].sum(axis=1, skipna=True)
            self.SS_load[g] = self.SS_load[s].sum(axis=1, skipna=True)

            self.N_var[g] = self.N_var[s].sum(axis=1, skipna=True)
            self.P_var[g] = self.P_var[s].sum(axis=1, skipna=True)
            self.SS_var[g] = self.SS_var[s].sum(axis=1, skipna=True)            
            

    def summarize(self, period=None, outfile=None):
        """
        Massages results and computes total loads, uncertainties as well as 
        contributions of different sources over given period.
        Args: 
            period - [start_year, end_year] (list)
        Returns:
            group_shares - pd.DataFrame of group level statistics
            in_groups: dict of pd.DataFrame's of within-group statistics
            columns: 'N', 'Nstd', 'fN', 'fNstd' -->
                    total N load (kg), std(kg), fract. of total(-), fract. std(-)
                    + same for P and SS
        """
        
        if period:
            start_year = period[0]; end_year = period[1]
        else:
            start_year = self.data.index[-1]; end_year = self.data.index[0]
        
        # compute temporary sum tables over time
        ix = (self.N_load.index >=start_year) & (self.N_load.index <= end_year)
        
        xN = self.N_load.iloc[ix].sum(axis=0, skipna=True)
        vN = self.N_var.iloc[ix].sum(axis=0, skipna=True)
        xP = self.P_load.iloc[ix].sum(axis=0, skipna=True)
        vP = self.P_var.iloc[ix].sum(axis=0, skipna=True)
        
        xSS = self.SS_load.iloc[ix].sum(axis=0, skipna=True)
        vSS = self.SS_var.iloc[ix].sum(axis=0, skipna=True)
        
        # container for group-level data
        group_shares = pd.DataFrame(index=self.groups,
                                    columns=['N', 'Nstd', 'fN', 'fNstd', 
                                             'P', 'Pstd', 'fP', 'fPstd',
                                             'SS', 'SSstd', 'fSS', 'fSSstd'])
        # dict container for within-group data
        ingroups = {key: None for key in self.groups}
        
        # summarize loads within groups and in total
        for g in self.groups:
            s = list(self.C[g].keys()) # loading sources within group
            s = [x for x in s if x in list(self.data.columns)] # loop only those in data
                
            # contribution of each group on total load
            tN = xN[self.groups].sum(); varN = vN[self.groups].sum()
            tP = xP[self.groups].sum(); varP = vP[self.groups].sum()   
            tSS = xSS[self.groups].sum(); varSS = vSS[self.groups].sum()
            
            a, b = source_contribution(xN[g], tN, vN[g], varN)
            group_shares.loc[g,['N', 'Nstd', 'fN', 'fNstd']] = [xN[g], vN[g]**0.5, a, b**0.5]
            
            a, b = source_contribution(xP[g], tP, vP[g], varP)
            group_shares.loc[g,['P', 'Pstd', 'fP', 'fPstd']] = [xP[g], vP[g]**0.5, a, b**0.5]
            
            a, b = source_contribution(xSS[g], tSS, vSS[g], varSS)
            group_shares.loc[g,['SS', 'SSstd', 'fSS', 'fSSstd']] = [xSS[g], vSS[g]**0.5, a, b**0.5]

            # contribution of each subgroup on group-level loading
            res = pd.DataFrame(index=s, columns=['N', 'Nstd', 'fN', 'fNstd', 
                                                 'P', 'Pstd', 'fP', 'fPstd',
                                                 'SS', 'SSstd', 'fSS', 'fSSstd'])            
            for k in s:            
                a, b = source_contribution(xN[k], xN[g], vN[k], vN[g])
                res.loc[k,['N', 'Nstd', 'fN', 'fNstd']] = [xN[k], vN[k]**0.5, a, b**0.5]
                
                a, b = source_contribution(xP[k], xP[g], vP[k], vP[g])
                res.loc[k,['P', 'Pstd', 'fP', 'fPstd']] = [xP[k], vP[k]**0.5, a, b**0.5]
                
                a, b = source_contribution(xSS[k], xSS[g], vSS[k], vSS[g])
                res.loc[k,['SS', 'SSstd', 'fSS', 'fSSstd']] = [xSS[k], vSS[k]**0.5, a, b**0.5]            
                
            ingroups[g] = res.copy(); del res
        
        # print results to csv-file
        if outfile:
            with open(outfile, 'a') as f:
                f.write('TOTAL')
                group_shares.to_csv(f, header=True)
                for g in self.groups:
                    f.write('\n' + g)
                    ingroups[g].to_csv(f, header=True)
                    
        return group_shares, ingroups


""" THESE FUNCTIONS DO ALL COMPUTING """

def annual_forestry_load(c, A, dA, width=3.3, scale=0.4):
    """ 
    computes annual load from forestry and using Kalle / Kustaa -methods
    Args:
        c - list of export coeffs [mean, min, max]
        A - area or units [array]
        dA - uncertainty of A    
        width = 3.3 when max-min range equals 90% of normal distribution
        width = 4.0 when max-min range equals 95%
    Returns:
        L - annual load
        V - var of annual load
    """
    N = len(A)
    
    if not None in c:
        varC = (abs(c[2] - c[1]) / width)**2.0
    else:
        varC = (scale * c[0] / width)**2.0
        
    varA = (2.0 * dA / width)**2.0             
    L = np.ones(N)*np.NaN
    V = np.ones(N)*np.NaN
    
    # loop in time. start from last year [0] and include also 9 previous years
    for n in range(N):
        # check edges
        mx = min(n + 10, N)
        nn = min(10, mx-n)

        # mean annual load and its variance
        L[n] = np.nansum(c[0][0:nn]*A[n:mx])
        V[n] = np.nansum(prod_variance(A[n:mx], varA[n:mx], c[0][0:nn], varC[0:nn]))
    
    return L, V

def annual_load(c, A, dA, width=3.3, scale=0.4):
    """ 
    computes annual load and its variance
    Args:
        c - list of export coeffs [mean, min, max]
        A - area or units [array]
        dA - uncertainty of A    
        width = 3.3 when max-min range equals 90% of normal distribution
        width = 4.0 when max-min range equals 95%
    Returns:
        L - annual load
        V - var of annual load
    """
    # variance of export coefficient
    if not None in c:
        varC = (abs(c[2] - c[1]) / width)**2.0
    else:
        varC = (scale * c[0] / width)**2.0
        
    varA = (2.0 * dA / width)**2.0             
    
    # mean annual load and its variance
    L = c[0]*A
    V = prod_variance(A, varA, c[0], varC)
    
    return L, V

def prod_variance(X, varX, Y, varY):
    """ variance of product XY """
    z = X**2.0 * varY + varX * varY + varX * Y**2.0
    return z

def source_contribution(K, L, varK, varL):
    """
    relative contrbution and variance of contribution
    Args:
        K - load
        L - total load
        varK - variance of load
        varL - variance of total load
    Returns:
        q - contribution (-) expectation value
        v - var of contribution (-)
    """    
    q = K / L + varL* K / L**3.0 - varK / L**3
    
    v = (K / L)**2.0 * (varK / K**2.0 + varL / L**2.0 - 2.0 * varK / (K * L))
  
    return q, v