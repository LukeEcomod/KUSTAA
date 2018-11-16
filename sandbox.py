# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:09:12 2018

@author: slauniai
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kustaa import Kustaa, make_summary_fig, make_timeseries_fig

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

# now create instance of Kustaa, do computattions and draw few figures

# import export coefficietns
from export_coefficients import default_coeff as coeff

# create Kustaa -instance for the computation area
# be sure that all columns in data can be found as keys in coeff dict (case sensitive)

ku1 = Kustaa(coeff, area, lakes, err, data, scale=0.4, fmodel='Kalle')

# compute annual loads and summarize. saves results into model instance and into files
ku1.compute_loads()
ku1.summarize(period=[2000, 2010], outfile='Kustaa_test.csv')

# draw few figures:

# summary of all groups and their contributions over period
make_summary_fig(dat=ku1.group_shares, figtitle='Ku1 results')

#summary of forestry operations and their contribution to forestry load
make_summary_fig(dat=ku1.ingroup_shares['FORESTRY'], figtitle='Ku1 results')

# timeseries of group-level N-loads. The range is +/- std
make_timeseries_fig(L=ku1.N_load, V=ku1.N_var, period=[2000, 2010], 
                    cols=ku1.groups, figtitle='N - load')

#%% Repeat same but with simpler set of loading sources

folder = r'c:\Repositories\Kustaa\data'
# annual operations and other pollution sources
datafile2 = r'KUSTAA_simple_example.csv'

area = 6000.0 # catchment land area ha
lakes = 100.0 # water bodies within catchment ha

# import annual loading source data from csv-file, replace NaN with 0.0
data2 = pd.read_csv(os.path.join(folder, datafile2), sep=';', header='infer')
data2 = data2.fillna(0.0)

# relative uncertainties of loading areas are extracted from 1st row and
# converted to dict
err2 = data2[data2.columns].iloc[0]
err2 = err2.drop('Year').to_dict() 
err2['Background'] = 5.0
err2['Deposition'] = 2.0

# now drop that row from data
data2 = data2.drop(index=0)
data2.index = data2['Year'].values.astype(int)
data2 = data2.drop(columns='Year')

# now create instance of Kustaa, do computattions and draw few figures

# import export coefficietns
from export_coefficients import simple_coeff as coeff

# create Kustaa -instance for the computation area. Now assign all forestry load to single year
# be sure that all columns in data can be found as keys in coeff dict (case sensitive)

ku2 = Kustaa(coeff, area, lakes, err2, data2, scale=0.4, fmodel='Annual')

# compute annual loads and summarize. saves results into model instance and into files
ku2.compute_loads()
ku2.summarize(period=[2000, 2010], outfile='Kustaa_test2.csv')

# draw few figures:

# summary of all groups and their contributions over period
make_summary_fig(dat=ku2.group_shares, figtitle='Ku2 results')

#summary of forestry operations and their contribution to forestry load
make_summary_fig(dat=ku2.ingroup_shares['ANTHROPOGENIC'], figtitle='Ku2 results')

# timeseries of group-level N-loads. The range is +/- std
make_timeseries_fig(L=ku2.N_load, V=ku2.N_var, period=[2000, 2010], 
                    cols=ku2.groups, figtitle='N - load')