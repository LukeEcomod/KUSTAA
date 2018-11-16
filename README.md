# KUSTAA

Samuli Launiainen, Luke 15.11.2018

Python implementation of Kustaa -tool for sediment and nutrient loading at catchment scale. Based on KUSTAA in Excel-VBA and Launiainen et al. 2014 Suomen Ympäristö 33. One can find these documents from \excel -folder.

Tested in Python 3.6; requires numpy and pandas

To use:

- download and unzip to folder. 
- in spyder or python prompt set working dir to folder
- sandbox.py gives examples how to use
- example input data (annual areas or polluter units) in .csv-files in \data
- note that all columns in input data must have (case-sensitive) keys in export_coefficient -dict
- export load coefficients defined in export_coefficients - modify according to your need
- note that column units in data (ha, ton, pieces, ...) must meet the way how export coefficients are defined.
- stores results in Kustaa-object fields; and prints them also to -csv -files. The logic is: 1) plain annual loads (and their variances) per source are stored in self.N_load and self.N_var -dataframes (and same for P and SS) by self.compute_loads(). These results can then be massaged in many ways to compute different results, 2) long-term sums and contributions to total load over given period are computed in self.summarize() similarly to Kustaa Excel -tool. 3) simple graphs can be produces also.

- two different approaches to compute forestry load: 1) Kalle - follows Finer et al. 2010 and disributes loading to 10 year period after operation (works now only with default_coeff or at least needs group 'FORESTRY'). 2) 'annual' or whatever: assigns whole 10year load to the operation year 
- for export coefficient values, see Excel-version and pdf-document(s)
- export coeffients for e.g. N are given in list: 'N': [ave, min, max], where min - max is typical variability range and assumed to correspond 90% of total variability later in the code.
