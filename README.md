## KUSTAA - tools to estimate diffuse and point-source nutrient and sediment loads from boreal catchments using specific-export approach

***KUSTAA*** is a simple tool to estimate loading of suspended solids, total nitrogen and phosphorus from diffuse and point-sources in a catchment. The repository containst two versions:

***Easy-to-use spreadsheet-tool*** in MS Excel, located in \excel folder. This follows Launiainen et al. 2015 Suomen Ympäristö 33 (In Finnish)

***Python 3.x implementation*** for more advanced use, without GUI. Used in Bhattacharjee et al. 2021 Sci. Tot. Env. 

***References:***

Launiainen, S., Sarkkola, S., Laurén, A., Puustinen, M., Tattari, S., Mattsson, T., Piirainen, S., Heinonen, J., Alakukku, L. ja Finér, L. 2015. KUSTAA-työkalu valuma-alueen vesistökuormituksen laskentaan. Suomen Ympäristökeskuksen Raportteja 33, 55p., ISBN 978-952-11-4374-8 (PDF)

Palviainen, M., Laurén, A., Launiainen, S. and Piirainen, S. 2016. Predicting the export and concentrations of organic carbon, nitrogen and phosphorus in boreal lakes by catchment characteristics and land use: a practical approach. Ambio (2016), doi:10.1007/s13280-016-0789-2

Bhattacharjee J., Marttila H., Launiainen S., Lepistö A., Kløve B. 2021. Combining Landsat image analysis, land-use statistics and land-use-specific export coefficient to predict river water quality after large scale peatland drainage. Science of the Total Environment, 779 146419, https://doi.org/10.1016/j.scitotenv.2021.146419

***Minimalistic User-guide for the Python-version:***

- Tested in Python 3.6; requires numpy and pandas.
- download and unzip to folder. 
- in spyder or python prompt set working dir to folder
- sandbox.py gives examples how to use
- example input data (annual areas or polluter units) in .csv-files in \data
- note that all columns in input data must have (case-sensitive) keys in export_coefficient -dict
- export load coefficients defined in export_coefficients - modify according to your need
- note that column units in data (ha, ton, pieces, ...) must meet the way how export coefficients are defined.
- stores results in Kustaa-object fields; and prints them also to -csv -files. The logic is: 1) plain annual loads (and their variances) per source are stored in self.N_load and self.N_var -dataframes (and same for P and SS) by self.compute_loads(). These results can then be massaged in many ways to compute different results, 2) long-term sums and contributions to total load over given period are computed in self.summarize() similarly to Kustaa Excel -tool. 3) simple graphs can be produces also.

- two different approaches to compute forestry load: 1) Kalle - follows Finer et al. 2010 Suomen Ympäristö, and disributes loading to 10 year period after operation (works now only with default_coeff or at least needs group 'FORESTRY'). 2) 'annual' or whatever: assigns whole 10year load to the operation year 
- for export coefficient values, see Excel-version and pdf-document(s)
- export coeffients for e.g. N are given in list: 'N': [ave, min, max], where min - max is typical variability range and assumed to correspond 90% of total variability later in the code.
