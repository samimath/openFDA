# Purpose of the project


This is an exercise of exploratoray analysis using data provided by the [openFDA API](https://open.fda.gov/apis/drug/). Specifically, I'm interesting in the drug event end point where information of an adverse events forproducts such as prescription drugs, OTC medication are submitted.


### 1. A bit about the data:

My first goal is to get a sense of what kind of data we are working with. How do we efficiently gather data? What does the data structure look like? What are some data cleaning and formatting tasks we'll need to do before it is in a workable state? 

Forunately, data structure information is made easier from the yaml file provided by OpenFDA:


Based on the `yaml` file provided by openFDA, we can see that these are the attributes of the data from the _drug event_ end point, with those in **bold** indicating nested keys which contain additional arrays of information:


* authoritynumb 

* companynumb 

* duplicate

* fulfillexpeditecriteria

* occurcountry

* **patient**

* **primarysource**

* primarysourcecountry 

* receiptdate

* **receiver** 

* **reportduplicate** 

* reporttype 

* safetyreportid 

* safetyreportversion 

* **sender** 

* serious 

* seriousnesscongenitalanomali 

* seriousnessdeath 

* seriousnessdisabling 

* seriousnesshospitalization 

* seriousnesslifethreatening 

* seriousnessother 

* transmissiondate 

* transmissiondateformat


# OpenFDA exploratory notebook

Sami Furst, May 2020
## Purpose of this notebook : 

### 1. get a sense of the data
### 2. try to find something interesting in the data through feature generation and visualization




```python
import requests
import os
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import time
from joblib import delayed, Parallel
import datetime
# helper functions:
import helpers
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore') 

import folium
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

%matplotlib inline
```

================================================================================================================================================================================================
# 1. Let's get a sense of the sample data:

- The sample of data we pulled has 63800 rows and 85 columns, with event dates ranging from January 2010 to March 2020. Note: the total data set consists of millions of records, and the data we are working with here is only a small part of it.

- There are quite a few columns with incomplete data, we'll need to do some data QC to select more useful features

- While some data is not missing, they may still be non-informative if e.g. there is only 1 unique value



### helper functions:



```python
def country_convert(s):
    
    if s =='YU':
        
        s = 'Yugoslavia'
    try:
        s = coco.convert(s, to ='short_name')
    
    except:
        pass
    
    return(s)




def days_between(d1, d2):
    
    if np.isnan(d1) or np.isnan(d2):
        
        return np.nan

    else:
    
        d1 = str(int(d1))

        d2 = str(int(d2))

    if len(d1) == 6:
        
        d1 = d1+'01'
        
    elif len(d1) == 4:
        
        d1 = d1 +'0101'
        
    if len(d2) == 6:

        d2 = d2+'01'

    elif len(d2) == 4:

        d2 = d2 +'0101'
    
    try:
        
        d1 = datetime.datetime.strptime(d1, "%Y%m%d")
        d2 = datetime.datetime.strptime(d2, "%Y%m%d")
    
        return abs((d2 - d1).days)
    except:
        return np.nan
```

## 1.1 First pass of the data:


```python
data = pd.read_csv('openFDA_data/data/combined_weekly_sample_20200515.csv');
```

    /anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py:3063: DtypeWarning: Columns (10,23,29,31,41,55,62,65,66) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
data.shape
```




    (63800, 81)



###  How many variables are missing data?


```python
plt.figure(figsize=(15,20))
missing_data_count = sns.heatmap(data.isnull().transpose(), cbar = True)
plt.savefig('missing_data_count.png')
#sns.barplot(data_var)
#missing_data_count.savefig("missing_data_count.png")
```


![png](output_8_0.png)


### How many unique levels there are across the variables?


```python
unique_levels = pd.DataFrame(data = {'column':data.columns,'nunique':data.nunique()}).sort_values('nunique')
plt.figure(figsize=(15,20))
sns.barplot(data=unique_levels, y ='column', x='nunique')
plt.savefig('unique_levels.png')
```


![png](output_10_0.png)


================================================================================================================================================================================================

## 1.2 Data cleaning & additional feature generation :

Horizontal (column): by dropping a feature using the following criteria:

- % of missing data > 80%

- has only 1 unique level

- represents a nested key (those were already unested in the previous data gathering function so information should be contained in other columns)




### new features:

- patient age in years
- drug duration (taken as difference between `drugstartdate` and `drugenddate`, if exists
- lat and lon of reporter countries (using `geocode` package)
- count of reports by drugs, country and active substance


```python

countries = data_subset['reportercountry'].unique()

countries_short = [country_convert(val) for val in countries]

country_lookup = pd.DataFrame(data = {'reportercountry':countries,'country_short':countries_short})

from geopy.geocoders import Nominatim
geolocator = Nominatim()
#import numpy as np

locs = [geolocator.geocode(c) for c in country_lookup['country_short']]

lat = [locs[i].latitude if locs[i] is not None else np.NaN for i in range(0,len(country_lookup))]
lon = [locs[i].longitude if locs[i] is not None else np.NaN  for i in range(0,len(country_lookup))]
country_lookup['lat'] = lat
country_lookup['lon'] = lon
```


```python
missing_df = data.isnull()

missing_df_info = pd.DataFrame(data = {'missing_count':missing_df[missing_df==True].count(axis=0)/len(data)})


nested_keys = ['primarysource','sender','patient','receiver','reportduplicate','summary','patientdeath']
informative_vars1 = [val for val in set(missing_df_info[missing_df_info['missing_count']<0.8].index) - set(nested_keys) ]

informative_vars2 = [val for val in unique_levels[unique_levels['nunique']>1].index]

informative_vars = [val for val in informative_vars1  or informative_vars2]

data_subset = data[informative_vars]

data_subset['serious'] = data_subset['serious'].apply(lambda s : 1 if s ==1 else 0)

# add country information:
country_lookup = pd.read_csv('country_lookup.csv')

data_subset = data_subset.merge(country_lookup)

country_count = data_subset['country_short'].value_counts(ascending = False).reset_index()

country_count.columns = ['country_short','country_short_count']


data_subset = data_subset.merge(country_count)

drug_duration = [ days_between(data_subset['drugstartdate'][i],data_subset['drugenddate'][i]) for i in range(0,len(data_subset))]


# add drug duration:
data_subset['drug_duration'] = drug_duration


active_sub_count = data_subset['active_substance'].value_counts().reset_index()

active_sub_count.columns = ['active_substance','active_substance_count']

medicinalproduct_count = data_subset['medicinalproduct'].value_counts().reset_index()

medicinalproduct_count.columns = ['medicinalproduct','medicinalproduct_count']

data_subset = data_subset.merge(active_sub_count).merge(medicinalproduct_count)


```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:15: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app



```python
len(data_subset['patient_age_year'].dropna())
```




    32506



### Let's take a look at the variables that got removed:


```python
set(data.columns) - set(informative_vars)
```




    {'drugadditional',
     'drugcumulativedosagenumb',
     'drugcumulativedosageunit',
     'drugrecurreadministration',
     'drugrecurrence',
     'drugtreatmentduration',
     'drugtreatmentdurationunit',
     'patientagegroup',
     'patientdeath',
     'summary'}



================================================================================================================================================================================================
# 2. Now some data analysis and visualization:


### More helper functions:


```python
def get_sample_rank(data_subset,col,pivot_value = 'serious'):
    
    col_count = str(col)+'_count'
    if col_count in data_subset.columns:
        
        strat_sample = data_subset[(data_subset[col_count]>200) & (data_subset[col]!='None')].groupby(col).apply(lambda d: d.sample(100))


        strat_sample.reset_index(drop=True, inplace = True)

        return strat_sample.pivot_table(values=pivot_value, index=col).reset_index().sort_values(pivot_value,ascending=False)
    else:
        return None
    
def compile_bootstrap(data_subset,col,pivot_value = 'serious',n=100, plot = True):
    
    bootstrap_samples = pd.concat(Parallel(n_jobs= -1)(delayed(get_sample_rank)(data_subset=data_subset,col=col,pivot_value=pivot_value) for i in np.arange(0,n)))
    
    if plot:
        plt.figure(figsize=(20,20))
        sns.boxplot(data = bootstrap_samples,  y = col , x = pivot_value)
        plt.title('Rate of '+ pivot_value + ' occurence based on ' + str(n)+' bootstrap samples of 100')
        plt.ylabel(col)
        plt.xlabel('Bootstrap rate')
        fig_name = col+'_'+pivot_value+'_'+ str(n)
        plt.savefig('figures/'+fig_name+'.png')
    
    bootstrap_median = bootstrap_samples.groupby(by=col).median().reset_index()
    
    bootstrap_median.columns = [col,'median']

    bootstrap_var = bootstrap_samples.groupby(by=col).var().reset_index()
    
    bootstrap_var.columns = [col,'var']
    

    return bootstrap_median.merge(bootstrap_var).sort_values('median',ascending = False)





def folium_map(data2, title = 'Rate of Life Theatening Conditions Across Countries',map_col = 'country', data_col = 'proportion', scale = 1000000):
    # Make an empty map
    m = folium.Map(location=[20, 0], zoom_start=3,width=1500,height=800)
    
    title_html = '''
             <h3 align="left" style="font-size:20px"><b>''' + title + '''</b></h3>'''
    m.get_root().html.add_child(folium.Element(title_html))


    for i in range(0,len(data2)):
           folium.Circle(
              location=[data2['lat'][i], data2['lon'][i]],
              popup= data2[map_col][i] + ': ' + str(round(100*data2[data_col][i],2))+ '%',
              radius=data2[data_col][i]*scale,
              color='crimson',
            fill = True,
            fill_color='crimson').add_to(m)
    return m


```


## 2.1 Types of adverse events:

Let's see the breakdown of productions by adverse events:

From the metadata, we know that the flag `serious` indicates whether or not the adverse events result in serious conditions which are: death, a life threatening condition, hospitalization, disability, congenital anomaly, and other. 

In this sample, more than 50% of reports consist of serious adverse event.



```python
serious_flag = [val for val in data_subset.columns if 'serious' in val]
plt.figure(figsize=(20,10))
serious_pivot = data_subset[['country_short']+serious_flag].groupby(by='country_short').sum().reset_index().sort_values('seriousnesslifethreatening',ascending = False)
sns.set_color_codes("pastel")
g = sns.barplot(x = 'index', y = 'serious', 
                data = data_subset['serious'].value_counts('normalized').reset_index(), 
                hue ='index', alpha = 0.9);
new_title = 'Occurence frequency of serious adverse effects'
plt.title(new_title);
sns.despine(offset=10, trim=True);





plt.figure(figsize=(10,30))
sns.barplot(data = data_subset['country_short'].value_counts().reset_index(), x = 'country_short', y = 'index')
plt.title('Count of country');
```


![png](output_20_0.png)



![png](output_20_1.png)


## 2.2 Compare adverse events by countries and products

In order to visualize adverse events on a map, we need to standardize country names and get geocode (lat/lon), also, since countries are represented in widely different number. A straight-forward comparison may not be fair, so let's try to offset the imbalance by inspecting the rate of serious adverse event using bootstrap samples.  Namely, we'll only show countries that have at least 200 cases, and we'll do a stratified sampling of 100 reports per country, then extract the rate of adverse events  


```python
pivot_val = 'seriousnesslifethreatening'
country_bootstrap = compile_bootstrap(data_subset = data_subset, col ='country_short',pivot_value=pivot_val,n=50,plot=False)

country_bootstrap_map = country_bootstrap.merge(country_lookup[['country_short','lat','lon']])

```


```python
m = folium_map(data2 = country_bootstrap_map, title = 'Rate of serious adverse effects (countries with record > 100) that led to life threatening conditions',map_col='country_short', data_col='median',scale=2000000)

path=pivot_val +'_by_country.html'
m.save(path)
m
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgCiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgTF9OT19UT1VDSCA9IGZhbHNlOwogICAgICAgICAgICBMX0RJU0FCTEVfM0QgPSBmYWxzZTsKICAgICAgICA8L3NjcmlwdD4KICAgIAogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjYuMC9kaXN0L2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY29kZS5qcXVlcnkuY29tL2pxdWVyeS0xLjEyLjQubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5qcyI+PC9zY3JpcHQ+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjYuMC9kaXN0L2xlYWZsZXQuY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9mb250LWF3ZXNvbWUvNC42LjMvY3NzL2ZvbnQtYXdlc29tZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC4yL2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2Nkbi5naXRoYWNrLmNvbS9weXRob24tdmlzdWFsaXphdGlvbi9mb2xpdW0vbWFzdGVyL2ZvbGl1bS90ZW1wbGF0ZXMvbGVhZmxldC5hd2Vzb21lLnJvdGF0ZS5jc3MiLz4KICAgIDxzdHlsZT5odG1sLCBib2R5IHt3aWR0aDogMTAwJTtoZWlnaHQ6IDEwMCU7bWFyZ2luOiAwO3BhZGRpbmc6IDA7fTwvc3R5bGU+CiAgICA8c3R5bGU+I21hcCB7cG9zaXRpb246YWJzb2x1dGU7dG9wOjA7Ym90dG9tOjA7cmlnaHQ6MDtsZWZ0OjA7fTwvc3R5bGU+CiAgICAKICAgICAgICAgICAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwKICAgICAgICAgICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgICAgICAgICAgPHN0eWxlPgogICAgICAgICAgICAgICAgI21hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSB7CiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHJlbGF0aXZlOwogICAgICAgICAgICAgICAgICAgIHdpZHRoOiAxNTAwLjBweDsKICAgICAgICAgICAgICAgICAgICBoZWlnaHQ6IDgwMC4wcHg7CiAgICAgICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgICAgICB0b3A6IDAuMCU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgCjwvaGVhZD4KPGJvZHk+ICAgIAogICAgCiAgICAgICAgICAgICA8aDMgYWxpZ249ImxlZnQiIHN0eWxlPSJmb250LXNpemU6MjBweCI+PGI+UmF0ZSBvZiBzZXJpb3VzIGFkdmVyc2UgZWZmZWN0cyAoY291bnRyaWVzIHdpdGggcmVjb3JkID4gMTAwKSB0aGF0IGxlZCB0byBsaWZlIHRocmVhdGVuaW5nIGNvbmRpdGlvbnM8L2I+PC9oMz4KICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKICAgICAgICAgICAgdmFyIG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSA9IEwubWFwKAogICAgICAgICAgICAgICAgIm1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSIsCiAgICAgICAgICAgICAgICB7CiAgICAgICAgICAgICAgICAgICAgY2VudGVyOiBbMjAuMCwgMC4wXSwKICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3LAogICAgICAgICAgICAgICAgICAgIHpvb206IDMsCiAgICAgICAgICAgICAgICAgICAgem9vbUNvbnRyb2w6IHRydWUsCiAgICAgICAgICAgICAgICAgICAgcHJlZmVyQ2FudmFzOiBmYWxzZSwKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgKTsKCiAgICAgICAgICAgIAoKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl9mYTc5OTI5M2I3Njg0MjlmOGZlODk4YTU1MjZkNTI0ZiA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgImh0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nIiwKICAgICAgICAgICAgICAgIHsiYXR0cmlidXRpb24iOiAiRGF0YSBieSBcdTAwMjZjb3B5OyBcdTAwM2NhIGhyZWY9XCJodHRwOi8vb3BlbnN0cmVldG1hcC5vcmdcIlx1MDAzZU9wZW5TdHJlZXRNYXBcdTAwM2MvYVx1MDAzZSwgdW5kZXIgXHUwMDNjYSBocmVmPVwiaHR0cDovL3d3dy5vcGVuc3RyZWV0bWFwLm9yZy9jb3B5cmlnaHRcIlx1MDAzZU9EYkxcdTAwM2MvYVx1MDAzZS4iLCAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsICJtYXhOYXRpdmVab29tIjogMTgsICJtYXhab29tIjogMTgsICJtaW5ab29tIjogMCwgIm5vV3JhcCI6IGZhbHNlLCAib3BhY2l0eSI6IDEsICJzdWJkb21haW5zIjogImFiYyIsICJ0bXMiOiBmYWxzZX0KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZDYwOTNhZjRkNDA4NDI0YmFmYzA4OTVlNTU2NWQzY2YgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Ni42MDMzNTM5OTk5OTk5OTYsIDEuODg4MzMzNDk5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMDU5NDA0OWU5Y2MwNDVhNjk0ZDQzYWVjZjgxZGJiNTMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzI5OTlkMjZkYTg4MDRlYmZiMDY3ZWRjMWU4NTE1YTVmID0gJChgPGRpdiBpZD0iaHRtbF8yOTk5ZDI2ZGE4ODA0ZWJmYjA2N2VkYzFlODUxNWE1ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RnJhbmNlOiA4LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzA1OTQwNDllOWNjMDQ1YTY5NGQ0M2FlY2Y4MWRiYjUzLnNldENvbnRlbnQoaHRtbF8yOTk5ZDI2ZGE4ODA0ZWJmYjA2N2VkYzFlODUxNWE1Zik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9kNjA5M2FmNGQ0MDg0MjRiYWZjMDg5NWU1NTY1ZDNjZi5iaW5kUG9wdXAocG9wdXBfMDU5NDA0OWU5Y2MwNDVhNjk0ZDQzYWVjZjgxZGJiNTMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZmViZjMwYTk3YzIyNDNjYmEwNWUzMGVjMDg5ZDc3YWEgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Ni42MDMzNTM5OTk5OTk5OTYsIDEuODg4MzMzNDk5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxNjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDhiZTczNmNlNDdjNDM1MTliMjJiODcwYTNiOGJiODMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzBmOTA4YjE4ZWY1YjRiMzM5YjBhMmEzYzg0NWEwMjViID0gJChgPGRpdiBpZD0iaHRtbF8wZjkwOGIxOGVmNWI0YjMzOWIwYTJhM2M4NDVhMDI1YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RnJhbmNlOiA4LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2Q4YmU3MzZjZTQ3YzQzNTE5YjIyYjg3MGEzYjhiYjgzLnNldENvbnRlbnQoaHRtbF8wZjkwOGIxOGVmNWI0YjMzOWIwYTJhM2M4NDVhMDI1Yik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9mZWJmMzBhOTdjMjI0M2NiYTA1ZTMwZWMwODlkNzdhYS5iaW5kUG9wdXAocG9wdXBfZDhiZTczNmNlNDdjNDM1MTliMjJiODcwYTNiOGJiODMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfOGQzZTU3OGU5N2Y2NDkwYmJiODQ5ZTU3MTBhOGU2ZDcgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1MS4wODM0MTk2LCAxMC40MjM0NDY5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDE2MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lN2E3YmNiNmI1OTk0NDQ5YWJiMTRmOTA5NzNjYzg1YiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfODM5YmU1YThmNjE5NGYyNjlmZjZhZGZhZDBmOWNkMjEgPSAkKGA8ZGl2IGlkPSJodG1sXzgzOWJlNWE4ZjYxOTRmMjY5ZmY2YWRmYWQwZjljZDIxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HZXJtYW55OiA4LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2U3YTdiY2I2YjU5OTQ0NDlhYmIxNGY5MDk3M2NjODViLnNldENvbnRlbnQoaHRtbF84MzliZTVhOGY2MTk0ZjI2OWZmNmFkZmFkMGY5Y2QyMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV84ZDNlNTc4ZTk3ZjY0OTBiYmI4NDllNTcxMGE4ZTZkNy5iaW5kUG9wdXAocG9wdXBfZTdhN2JjYjZiNTk5NDQ0OWFiYjE0ZjkwOTczY2M4NWIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMTA5NTcxODBmOGMyNGViZGI0NDQ2OWQ0YzgwNDRiNTAgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1MS4wODM0MTk2LCAxMC40MjM0NDY5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDE2MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mYTc5NzY1MmE0NTg0NjZjOTQ2MWVmYjg3NWJjY2U0NyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfODAxMzkxZmQ1NDU4NDEyNGFhMjQ4MzViMTRhYTUwNDMgPSAkKGA8ZGl2IGlkPSJodG1sXzgwMTM5MWZkNTQ1ODQxMjRhYTI0ODM1YjE0YWE1MDQzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HZXJtYW55OiA4LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2ZhNzk3NjUyYTQ1ODQ2NmM5NDYxZWZiODc1YmNjZTQ3LnNldENvbnRlbnQoaHRtbF84MDEzOTFmZDU0NTg0MTI0YWEyNDgzNWIxNGFhNTA0Myk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8xMDk1NzE4MGY4YzI0ZWJkYjQ0NDY5ZDRjODA0NGI1MC5iaW5kUG9wdXAocG9wdXBfZmE3OTc2NTJhNDU4NDY2Yzk0NjFlZmI4NzViY2NlNDcpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMDE0Y2MxZjY3MDc0NDNiMGIxNWU0OTUxMTE2MGIxYzkgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Mi42Mzg0MjYxLCAxMi42NzQyOTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTQwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2FiNGVhNTYxN2UzYzQ2YjViNGQ3ZGEyNjM1YTZmMjhlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF81Y2ZlYzMyY2ZkMmY0ZDFjODAwY2YxMzZiMTA0NmUyOSA9ICQoYDxkaXYgaWQ9Imh0bWxfNWNmZWMzMmNmZDJmNGQxYzgwMGNmMTM2YjEwNDZlMjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkl0YWx5OiA3LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2FiNGVhNTYxN2UzYzQ2YjViNGQ3ZGEyNjM1YTZmMjhlLnNldENvbnRlbnQoaHRtbF81Y2ZlYzMyY2ZkMmY0ZDFjODAwY2YxMzZiMTA0NmUyOSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8wMTRjYzFmNjcwNzQ0M2IwYjE1ZTQ5NTExMTYwYjFjOS5iaW5kUG9wdXAocG9wdXBfYWI0ZWE1NjE3ZTNjNDZiNWI0ZDdkYTI2MzVhNmYyOGUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMTY4NTQyMTU2NDVhNGE0ZDk1NzcwZThhZjE4NTk1ODIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Mi42Mzg0MjYxLCAxMi42NzQyOTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTQwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzFlYTc3YzY2ZTA3MTQxMWFhMTc1ODkzNDkxMWMwYzM1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF80ODMwMjg2YTYxMjk0OWIzODc4YjZjODQyMTJjOGUyNSA9ICQoYDxkaXYgaWQ9Imh0bWxfNDgzMDI4NmE2MTI5NDliMzg3OGI2Yzg0MjEyYzhlMjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkl0YWx5OiA3LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzFlYTc3YzY2ZTA3MTQxMWFhMTc1ODkzNDkxMWMwYzM1LnNldENvbnRlbnQoaHRtbF80ODMwMjg2YTYxMjk0OWIzODc4YjZjODQyMTJjOGUyNSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8xNjg1NDIxNTY0NWE0YTRkOTU3NzBlOGFmMTg1OTU4Mi5iaW5kUG9wdXAocG9wdXBfMWVhNzdjNjZlMDcxNDExYWExNzU4OTM0OTExYzBjMzUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfN2E2MzM4MzU2MWE5NDFhMTg4NzQxYjA2NWRjNDRjOWIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1NC43MDIzNTQ1MDAwMDAwMDYsIC0zLjI3NjU3NTI5OTk5OTk5OTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTQwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2M3MmUyMDE4NGMxZjRkOTNhZjlmZmRkNmJmM2M1MjVkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hZjU2Yzk0Y2FhM2Y0ZjFkYjc3Njg3OGNiZjgxMWM3ZCA9ICQoYDxkaXYgaWQ9Imh0bWxfYWY1NmM5NGNhYTNmNGYxZGI3NzY4NzhjYmY4MTFjN2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXRlZCBLaW5nZG9tOiA3LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M3MmUyMDE4NGMxZjRkOTNhZjlmZmRkNmJmM2M1MjVkLnNldENvbnRlbnQoaHRtbF9hZjU2Yzk0Y2FhM2Y0ZjFkYjc3Njg3OGNiZjgxMWM3ZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV83YTYzMzgzNTYxYTk0MWExODg3NDFiMDY1ZGM0NGM5Yi5iaW5kUG9wdXAocG9wdXBfYzcyZTIwMTg0YzFmNGQ5M2FmOWZmZGQ2YmYzYzUyNWQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNWIxYzdhZTFmMmRlNDI1OGIzMGM5NWNjMTI1ZTYyNjYgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1NC43MDIzNTQ1MDAwMDAwMDYsIC0zLjI3NjU3NTI5OTk5OTk5OTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTQwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2RlOWIxZGEyNzBjYzQ3ODZiMjZhZjA2NzkyNWY4YjU3ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82OWNiZTlmMWMyOTU0MWIxYTRiZDRmNmUyZmQ0MzgxZiA9ICQoYDxkaXYgaWQ9Imh0bWxfNjljYmU5ZjFjMjk1NDFiMWE0YmQ0ZjZlMmZkNDM4MWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXRlZCBLaW5nZG9tOiA3LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2RlOWIxZGEyNzBjYzQ3ODZiMjZhZjA2NzkyNWY4YjU3LnNldENvbnRlbnQoaHRtbF82OWNiZTlmMWMyOTU0MWIxYTRiZDRmNmUyZmQ0MzgxZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV81YjFjN2FlMWYyZGU0MjU4YjMwYzk1Y2MxMjVlNjI2Ni5iaW5kUG9wdXAocG9wdXBfZGU5YjFkYTI3MGNjNDc4NmIyNmFmMDY3OTI1ZjhiNTcpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMDM4NzYxNTE2MGYzNGE0MjhjNzk3NjQ1ZWZlYjQ0YTUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFszNi41NzQ4NDQxLCAxMzkuMjM5NDE3OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxMjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDc0YTM1YmZiYjliNGM4NTljNDdhNTllODkwZTA3NWUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzQyYTIzMWE0ODEzYjQ5ODE4MzQwYjNiODQyMDk3YWIxID0gJChgPGRpdiBpZD0iaHRtbF80MmEyMzFhNDgxM2I0OTgxODM0MGIzYjg0MjA5N2FiMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SmFwYW46IDYuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZDc0YTM1YmZiYjliNGM4NTljNDdhNTllODkwZTA3NWUuc2V0Q29udGVudChodG1sXzQyYTIzMWE0ODEzYjQ5ODE4MzQwYjNiODQyMDk3YWIxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzAzODc2MTUxNjBmMzRhNDI4Yzc5NzY0NWVmZWI0NGE1LmJpbmRQb3B1cChwb3B1cF9kNzRhMzViZmJiOWI0Yzg1OWM0N2E1OWU4OTBlMDc1ZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV82ZmY4NjEwZTQ4ZWU0YTY1YTNiNDVmNzAxOWE4YmM4YyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM2LjU3NDg0NDEsIDEzOS4yMzk0MTc5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDEyMDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mNjQ5NTE4MzBjZDA0N2Q3ODA0ZDZjNzYyZTBmNWNjMCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzU0MDllODU4MmEyNDczN2FhNjFkNGVmMmYwZDkyNzMgPSAkKGA8ZGl2IGlkPSJodG1sXzM1NDA5ZTg1ODJhMjQ3MzdhYTYxZDRlZjJmMGQ5MjczIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5KYXBhbjogNi4wJTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mNjQ5NTE4MzBjZDA0N2Q3ODA0ZDZjNzYyZTBmNWNjMC5zZXRDb250ZW50KGh0bWxfMzU0MDllODU4MmEyNDczN2FhNjFkNGVmMmYwZDkyNzMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfNmZmODYxMGU0OGVlNGE2NWEzYjQ1ZjcwMTlhOGJjOGMuYmluZFBvcHVwKHBvcHVwX2Y2NDk1MTgzMGNkMDQ3ZDc4MDRkNmM3NjJlMGY1Y2MwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2NiOTNiMmNjNDNmNDRlNjRiYWVjZjRjMjcyNTJhMjQ3ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbNDYuNzk4NTYyNCwgOC4yMzE5NzM1OTk5OTk5OThdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMTIwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzcwZDEwNGVjMzEwYzQzOWI5NTk3NmIwNDM5NTM4NGFkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lMGQzN2E1NmNjZjA0ZDVjYjc2M2U3MDFkN2VlYjRkNyA9ICQoYDxkaXYgaWQ9Imh0bWxfZTBkMzdhNTZjY2YwNGQ1Y2I3NjNlNzAxZDdlZWI0ZDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN3aXR6ZXJsYW5kOiA2LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzcwZDEwNGVjMzEwYzQzOWI5NTk3NmIwNDM5NTM4NGFkLnNldENvbnRlbnQoaHRtbF9lMGQzN2E1NmNjZjA0ZDVjYjc2M2U3MDFkN2VlYjRkNyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9jYjkzYjJjYzQzZjQ0ZTY0YmFlY2Y0YzI3MjUyYTI0Ny5iaW5kUG9wdXAocG9wdXBfNzBkMTA0ZWMzMTBjNDM5Yjk1OTc2YjA0Mzk1Mzg0YWQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMWQ3NTEwYTExZWM0NDE4YTljZDM1MmFlYmRkMmY4ZWEgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Ni43OTg1NjI0LCA4LjIzMTk3MzU5OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxMjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWQxNmNhMjc0YzE3NDA2MmE4ZTdjNWIwYmRhYzg4ZDYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2RjZDA5MTgyODZiNzRhMGRiOTBhZGM4YTliZmRhZjkwID0gJChgPGRpdiBpZD0iaHRtbF9kY2QwOTE4Mjg2Yjc0YTBkYjkwYWRjOGE5YmZkYWY5MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3dpdHplcmxhbmQ6IDYuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNWQxNmNhMjc0YzE3NDA2MmE4ZTdjNWIwYmRhYzg4ZDYuc2V0Q29udGVudChodG1sX2RjZDA5MTgyODZiNzRhMGRiOTBhZGM4YTliZmRhZjkwKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzFkNzUxMGExMWVjNDQxOGE5Y2QzNTJhZWJkZDJmOGVhLmJpbmRQb3B1cChwb3B1cF81ZDE2Y2EyNzRjMTc0MDYyYThlN2M1YjBiZGFjODhkNikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9kMGM1MzU0ZmIwMDg0YTZhYTE2OGI5NjQyOWM4ZTgxMSA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM5LjMyNjIzNDUsIC00LjgzODA2NDkwMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxMDAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfY2NjMzBjMGE5OGY1NDlkYTlmZjQyZGIxNzYyNzU0YzIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2I5MGY2NjY5YWNmZjRmZTFiMDM5ZTUwNDE4NjEwYmFiID0gJChgPGRpdiBpZD0iaHRtbF9iOTBmNjY2OWFjZmY0ZmUxYjAzOWU1MDQxODYxMGJhYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3BhaW46IDUuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfY2NjMzBjMGE5OGY1NDlkYTlmZjQyZGIxNzYyNzU0YzIuc2V0Q29udGVudChodG1sX2I5MGY2NjY5YWNmZjRmZTFiMDM5ZTUwNDE4NjEwYmFiKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2QwYzUzNTRmYjAwODRhNmFhMTY4Yjk2NDI5YzhlODExLmJpbmRQb3B1cChwb3B1cF9jY2MzMGMwYTk4ZjU0OWRhOWZmNDJkYjE3NjI3NTRjMikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV84YjIwODM4NmZjMmY0ZDFiODE1MWQ2NzU2MDg1MzE3YiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM5LjMyNjIzNDUsIC00LjgzODA2NDkwMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiAxMDAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNjFlNDc4NzY1NzgxNDAxNGI1ZDdkM2M2Yjk2NDU1NmEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FkOTc2YmU0MDJhYjQ3ZDliNmY5YThjMTYwY2Q0YTNmID0gJChgPGRpdiBpZD0iaHRtbF9hZDk3NmJlNDAyYWI0N2Q5YjZmOWE4YzE2MGNkNGEzZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3BhaW46IDUuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjFlNDc4NzY1NzgxNDAxNGI1ZDdkM2M2Yjk2NDU1NmEuc2V0Q29udGVudChodG1sX2FkOTc2YmU0MDJhYjQ3ZDliNmY5YThjMTYwY2Q0YTNmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzhiMjA4Mzg2ZmMyZjRkMWI4MTUxZDY3NTYwODUzMTdiLmJpbmRQb3B1cChwb3B1cF82MWU0Nzg3NjU3ODE0MDE0YjVkN2QzYzZiOTY0NTU2YSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV82OWYyNWFjY2YwN2U0ZDJhOTIzNzQxNDYyYTQzOTcwMCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM4Ljk1OTc1OTM5OTk5OTk5NiwgMzQuOTI0OTY1M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA4MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iZTBiNGUwZTY1ODI0ZTY4YTNiNWQyODNmZjBkYmVkOCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzExZWZiZjY2M2RhNDUzODhmOGY0ZjFmZDk3NzQ2YmMgPSAkKGA8ZGl2IGlkPSJodG1sXzMxMWVmYmY2NjNkYTQ1Mzg4ZjhmNGYxZmQ5Nzc0NmJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UdXJrZXk6IDQuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYmUwYjRlMGU2NTgyNGU2OGEzYjVkMjgzZmYwZGJlZDguc2V0Q29udGVudChodG1sXzMxMWVmYmY2NjNkYTQ1Mzg4ZjhmNGYxZmQ5Nzc0NmJjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzY5ZjI1YWNjZjA3ZTRkMmE5MjM3NDE0NjJhNDM5NzAwLmJpbmRQb3B1cChwb3B1cF9iZTBiNGUwZTY1ODI0ZTY4YTNiNWQyODNmZjBkYmVkOCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8xMGM3ZWYzMDVjNDg0ZTc5OWYzNjcyNGI5MzVmMWIxYyA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM4Ljk1OTc1OTM5OTk5OTk5NiwgMzQuOTI0OTY1M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA4MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82YmJjNzAzODFhNDE0NDk0YmI1YjViY2Q1OGE2YWYyNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfN2RkNTM3M2VhNTI2NDgwNTg4MjVmODNjNjkwZTQzYmMgPSAkKGA8ZGl2IGlkPSJodG1sXzdkZDUzNzNlYTUyNjQ4MDU4ODI1ZjgzYzY5MGU0M2JjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UdXJrZXk6IDQuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNmJiYzcwMzgxYTQxNDQ5NGJiNWI1YmNkNThhNmFmMjQuc2V0Q29udGVudChodG1sXzdkZDUzNzNlYTUyNjQ4MDU4ODI1ZjgzYzY5MGU0M2JjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlXzEwYzdlZjMwNWM0ODRlNzk5ZjM2NzI0YjkzNWYxYjFjLmJpbmRQb3B1cChwb3B1cF82YmJjNzAzODFhNDE0NDk0YmI1YjViY2Q1OGE2YWYyNCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV8zYjJjNDM5NTYzMjQ0MmFlODQzMjQxYjZmYjBlZjhmOCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWy0xMC4zMzMzMzMzMDAwMDAwMDEsIC01My4yXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDgwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzkyZjI5ZDk1MDA4MjQzZTFiMjk1YTdmZjExZTViN2Q2ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8zNDQ5YzdhOTU2Njk0MzZiYjMyY2RjZDcyNTk5MTNhMSA9ICQoYDxkaXYgaWQ9Imh0bWxfMzQ0OWM3YTk1NjY5NDM2YmIzMmNkY2Q3MjU5OTEzYTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyYXppbDogNC4wJTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF85MmYyOWQ5NTAwODI0M2UxYjI5NWE3ZmYxMWU1YjdkNi5zZXRDb250ZW50KGh0bWxfMzQ0OWM3YTk1NjY5NDM2YmIzMmNkY2Q3MjU5OTEzYTEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfM2IyYzQzOTU2MzI0NDJhZTg0MzI0MWI2ZmIwZWY4ZjguYmluZFBvcHVwKHBvcHVwXzkyZjI5ZDk1MDA4MjQzZTFiMjk1YTdmZjExZTViN2Q2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzIzNWQ3NGY3MjU0MjQwMzNhZjQyZWM3ZDUwNzU3MTcyID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTEwLjMzMzMzMzMwMDAwMDAwMSwgLTUzLjJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogODAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYWQwNTE3ODI3NjE0NDIzNzlmOWNiMjJjYTQwMWY5NWUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzQ2MDViZTBmODEyMzQwNTI5OTcyNzc4ZjMxNzk0NWVlID0gJChgPGRpdiBpZD0iaHRtbF80NjA1YmUwZjgxMjM0MDUyOTk3Mjc3OGYzMTc5NDVlZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnJhemlsOiA0LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2FkMDUxNzgyNzYxNDQyMzc5ZjljYjIyY2E0MDFmOTVlLnNldENvbnRlbnQoaHRtbF80NjA1YmUwZjgxMjM0MDUyOTk3Mjc3OGYzMTc5NDVlZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8yMzVkNzRmNzI1NDI0MDMzYWY0MmVjN2Q1MDc1NzE3Mi5iaW5kUG9wdXAocG9wdXBfYWQwNTE3ODI3NjE0NDIzNzlmOWNiMjJjYTQwMWY5NWUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZTQ5YjExOGQyODRiNDdhMGEwNjkwOTczNjI2NDFiMWQgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs2MS4wNjY2OTIyLCAtMTA3Ljk5MTcwNzFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogODAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMWE5YjBjOGVlYmM5NGI5NGJmZjNlMjIyZmI4ODQzZGMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzE1NDNkOWQyZTVmZTQ5ZDhiOWZlOGQ0OTkyOTRiNGJkID0gJChgPGRpdiBpZD0iaHRtbF8xNTQzZDlkMmU1ZmU0OWQ4YjlmZThkNDk5Mjk0YjRiZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FuYWRhOiA0LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzFhOWIwYzhlZWJjOTRiOTRiZmYzZTIyMmZiODg0M2RjLnNldENvbnRlbnQoaHRtbF8xNTQzZDlkMmU1ZmU0OWQ4YjlmZThkNDk5Mjk0YjRiZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9lNDliMTE4ZDI4NGI0N2EwYTA2OTA5NzM2MjY0MWIxZC5iaW5kUG9wdXAocG9wdXBfMWE5YjBjOGVlYmM5NGI5NGJmZjNlMjIyZmI4ODQzZGMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMGJkNjVhYjg1MTE0NGQyODgzYjIwYWRmNTFjMDYwNmUgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs2MS4wNjY2OTIyLCAtMTA3Ljk5MTcwNzFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogODAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMGQyY2RlMDQ0OGVjNDZhMjg0MjcwZGNhYjhlMTJiNzMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzE1OTNiMDhjZWZhYjRhNWZhZjc1MGI2NTg1YWNiOGYyID0gJChgPGRpdiBpZD0iaHRtbF8xNTkzYjA4Y2VmYWI0YTVmYWY3NTBiNjU4NWFjYjhmMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FuYWRhOiA0LjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzBkMmNkZTA0NDhlYzQ2YTI4NDI3MGRjYWI4ZTEyYjczLnNldENvbnRlbnQoaHRtbF8xNTkzYjA4Y2VmYWI0YTVmYWY3NTBiNjU4NWFjYjhmMik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8wYmQ2NWFiODUxMTQ0ZDI4ODNiMjBhZGY1MWMwNjA2ZS5iaW5kUG9wdXAocG9wdXBfMGQyY2RlMDQ0OGVjNDZhMjg0MjcwZGNhYjhlMTJiNzMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNzExNGQyNDRkYTgyNDgwOWFkNjg0Y2VjNjg0ZmQ1NTIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsyMi4zNTExMTQ4LCA3OC42Njc3NDI4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDYwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2FlZDAyYzBmY2Q4YzQzYTM4NTFmMmYyNTQ3MGJkMGZkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF80MTE5ZDVkOTM3OTg0Mjg1YWE1N2NmY2ViZTE3M2IzZSA9ICQoYDxkaXYgaWQ9Imh0bWxfNDExOWQ1ZDkzNzk4NDI4NWFhNTdjZmNlYmUxNzNiM2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluZGlhOiAzLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2FlZDAyYzBmY2Q4YzQzYTM4NTFmMmYyNTQ3MGJkMGZkLnNldENvbnRlbnQoaHRtbF80MTE5ZDVkOTM3OTg0Mjg1YWE1N2NmY2ViZTE3M2IzZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV83MTE0ZDI0NGRhODI0ODA5YWQ2ODRjZWM2ODRmZDU1Mi5iaW5kUG9wdXAocG9wdXBfYWVkMDJjMGZjZDhjNDNhMzg1MWYyZjI1NDcwYmQwZmQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfYzlhZWZlMjMwNDQ2NGI0NWIxZTkxMzZhYzQ4MjFkMzIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFsyMi4zNTExMTQ4LCA3OC42Njc3NDI4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDYwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzgxM2IxMjM4MzM2NzRmMzQ4ODg4MjdhZDRjOWUxODI5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kMDdmMTYxMjRlM2E0Mzc1YTIxNzQ4YTNhNGFjYmYzNyA9ICQoYDxkaXYgaWQ9Imh0bWxfZDA3ZjE2MTI0ZTNhNDM3NWEyMTc0OGEzYTRhY2JmMzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluZGlhOiAzLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzgxM2IxMjM4MzM2NzRmMzQ4ODg4MjdhZDRjOWUxODI5LnNldENvbnRlbnQoaHRtbF9kMDdmMTYxMjRlM2E0Mzc1YTIxNzQ4YTNhNGFjYmYzNyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9jOWFlZmUyMzA0NDY0YjQ1YjFlOTEzNmFjNDgyMWQzMi5iaW5kUG9wdXAocG9wdXBfODEzYjEyMzgzMzY3NGYzNDg4ODgyN2FkNGM5ZTE4MjkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfN2ZhNWQwZDk4Yzk0NGFhYWEyZjk1NzE4YWI4NmFkY2EgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1Mi41MDAxNjk3OTk5OTk5OTUsIDUuNzQ4MDgyMDk5OTk5OTk5NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA2MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9kNTRmMDE5OWFlMjA0MjZjOGQyZWM3MTNhZThiNmY4YSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMTA5ZmQzNzBjYjY1NGJkYzg0NDcwMGNhYjM5MTFlYTEgPSAkKGA8ZGl2IGlkPSJodG1sXzEwOWZkMzcwY2I2NTRiZGM4NDQ3MDBjYWIzOTExZWExIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OZXRoZXJsYW5kczogMy4wJTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9kNTRmMDE5OWFlMjA0MjZjOGQyZWM3MTNhZThiNmY4YS5zZXRDb250ZW50KGh0bWxfMTA5ZmQzNzBjYjY1NGJkYzg0NDcwMGNhYjM5MTFlYTEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfN2ZhNWQwZDk4Yzk0NGFhYWEyZjk1NzE4YWI4NmFkY2EuYmluZFBvcHVwKHBvcHVwX2Q1NGYwMTk5YWUyMDQyNmM4ZDJlYzcxM2FlOGI2ZjhhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2M1NTQ3ZjBkYzA1NTRhZDdhZWM4MjViM2YwNjU3ODgwID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbNTIuNTAwMTY5Nzk5OTk5OTk1LCA1Ljc0ODA4MjA5OTk5OTk5OTVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDczNWU2MWY4OGE0NGRiODlkZmVhYTZlZGRkNjYxZGQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzk4NjE0YTYyMjIzNzRjMzBiZGFkYWQxZThmYjI3MzIyID0gJChgPGRpdiBpZD0iaHRtbF85ODYxNGE2MjIyMzc0YzMwYmRhZGFkMWU4ZmIyNzMyMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV0aGVybGFuZHM6IDMuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZDczNWU2MWY4OGE0NGRiODlkZmVhYTZlZGRkNjYxZGQuc2V0Q29udGVudChodG1sXzk4NjE0YTYyMjIzNzRjMzBiZGFkYWQxZThmYjI3MzIyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2M1NTQ3ZjBkYzA1NTRhZDdhZWM4MjViM2YwNjU3ODgwLmJpbmRQb3B1cChwb3B1cF9kNzM1ZTYxZjg4YTQ0ZGI4OWRmZWFhNmVkZGQ2NjFkZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9kMWM4ZmUyYTBjYTE0NjlmOGY2OTM1NzkzMDM0N2UzOCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM2LjYzODM5MTk5OTk5OTk5NiwgMTI3LjY5NjExODgwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDYwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2U1MGY3N2UyZTlhYjRlZmViNGRmNWJlYTQxYzRhNGRjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yMDU0OWNjZjgwN2Q0MTliOWU3NDZlMzNiMzNiODdmNyA9ICQoYDxkaXYgaWQ9Imh0bWxfMjA1NDljY2Y4MDdkNDE5YjllNzQ2ZTMzYjMzYjg3ZjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNvdXRoIEtvcmVhOiAzLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2U1MGY3N2UyZTlhYjRlZmViNGRmNWJlYTQxYzRhNGRjLnNldENvbnRlbnQoaHRtbF8yMDU0OWNjZjgwN2Q0MTliOWU3NDZlMzNiMzNiODdmNyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9kMWM4ZmUyYTBjYTE0NjlmOGY2OTM1NzkzMDM0N2UzOC5iaW5kUG9wdXAocG9wdXBfZTUwZjc3ZTJlOWFiNGVmZWI0ZGY1YmVhNDFjNGE0ZGMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNmUxZTZkMjI0NDIyNDIwNGI4ODA1MjUwZTllOTE5M2IgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFszNi42MzgzOTE5OTk5OTk5OTYsIDEyNy42OTYxMTg4MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA2MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9kOWY3NTc4ZDUzMzE0ODJkYTUzNjM5NGZlYzZhOTQ5ZSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZWIxNjJlZjY3ZmI4NDA0NGJkNmQzZjUzNzQzMDIzMjUgPSAkKGA8ZGl2IGlkPSJodG1sX2ViMTYyZWY2N2ZiODQwNDRiZDZkM2Y1Mzc0MzAyMzI1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Tb3V0aCBLb3JlYTogMy4wJTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9kOWY3NTc4ZDUzMzE0ODJkYTUzNjM5NGZlYzZhOTQ5ZS5zZXRDb250ZW50KGh0bWxfZWIxNjJlZjY3ZmI4NDA0NGJkNmQzZjUzNzQzMDIzMjUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfNmUxZTZkMjI0NDIyNDIwNGI4ODA1MjUwZTllOTE5M2IuYmluZFBvcHVwKHBvcHVwX2Q5Zjc1NzhkNTMzMTQ4MmRhNTM2Mzk0ZmVjNmE5NDllKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX2VkZGYwOGU2MDIzZTQ5MzJiMmM3M2RmN2Y2MzM0NDMyID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbNTkuNjc0OTcxMjAwMDAwMDEsIDE0LjUyMDg1ODRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzk0NjNiYTZhZDJhNGM2NTkwMmI5MmIzODcwMmZkNDUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzc0MmIzMDMzZjgyZjRmMDU5NjRmNzk2YTlhMTU1YTY4ID0gJChgPGRpdiBpZD0iaHRtbF83NDJiMzAzM2Y4MmY0ZjA1OTY0Zjc5NmE5YTE1NWE2OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3dlZGVuOiAzLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M5NDYzYmE2YWQyYTRjNjU5MDJiOTJiMzg3MDJmZDQ1LnNldENvbnRlbnQoaHRtbF83NDJiMzAzM2Y4MmY0ZjA1OTY0Zjc5NmE5YTE1NWE2OCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9lZGRmMDhlNjAyM2U0OTMyYjJjNzNkZjdmNjMzNDQzMi5iaW5kUG9wdXAocG9wdXBfYzk0NjNiYTZhZDJhNGM2NTkwMmI5MmIzODcwMmZkNDUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfYmE0M2I0YWRjYjlkNDVkNWFhOWY5NDIwYTZkMzNmNTMgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs1OS42NzQ5NzEyMDAwMDAwMSwgMTQuNTIwODU4NF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA2MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF85ZGVmMzI0M2IyMWE0NTgyODNhYTljYzMzMjcxMGMxOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMDIwNDg1ZDFkOGU5NDAxYjliODc4N2NhZGYxMWI5NTMgPSAkKGA8ZGl2IGlkPSJodG1sXzAyMDQ4NWQxZDhlOTQwMWI5Yjg3ODdjYWRmMTFiOTUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Td2VkZW46IDMuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOWRlZjMyNDNiMjFhNDU4MjgzYWE5Y2MzMzI3MTBjMTkuc2V0Q29udGVudChodG1sXzAyMDQ4NWQxZDhlOTQwMWI5Yjg3ODdjYWRmMTFiOTUzKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2JhNDNiNGFkY2I5ZDQ1ZDVhYTlmOTQyMGE2ZDMzZjUzLmJpbmRQb3B1cChwb3B1cF85ZGVmMzI0M2IyMWE0NTgyODNhYTljYzMzMjcxMGMxOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9lMzI4ZGJjODYzMzM0MjcwYWViOTBiMmViYmM0MTQyMiA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM5Ljc4MzczMDQsIC0xMDAuNDQ1ODgyNV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA0MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8wMjlkNzY5NTU2NzM0OGEwYmVhOTg0YWY5NjNiMmEwOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYzRlMmRmMjQ1NjkzNDljMmE0ZTM3M2QzN2FiNWUxMmMgPSAkKGA8ZGl2IGlkPSJodG1sX2M0ZTJkZjI0NTY5MzQ5YzJhNGUzNzNkMzdhYjVlMTJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml0ZWQgU3RhdGVzOiAyLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzAyOWQ3Njk1NTY3MzQ4YTBiZWE5ODRhZjk2M2IyYTA5LnNldENvbnRlbnQoaHRtbF9jNGUyZGYyNDU2OTM0OWMyYTRlMzczZDM3YWI1ZTEyYyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9lMzI4ZGJjODYzMzM0MjcwYWViOTBiMmViYmM0MTQyMi5iaW5kUG9wdXAocG9wdXBfMDI5ZDc2OTU1NjczNDhhMGJlYTk4NGFmOTYzYjJhMDkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNjY3MzMyZWRlZjhmNDdlYTgyNTBhNDcyMzEwZGY4MmMgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFszOS43ODM3MzA0LCAtMTAwLjQ0NTg4MjVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNDAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNjZhY2Q3NWUyNjIyNDI5OGExNTc1OTE2NjMyOGQ1Y2YgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzhlMWU3NmNkNDczOTRjMGVhY2E1OGZjZTAxZjllMWJiID0gJChgPGRpdiBpZD0iaHRtbF84ZTFlNzZjZDQ3Mzk0YzBlYWNhNThmY2UwMWY5ZTFiYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdGVkIFN0YXRlczogMi4wJTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82NmFjZDc1ZTI2MjI0Mjk4YTE1NzU5MTY2MzI4ZDVjZi5zZXRDb250ZW50KGh0bWxfOGUxZTc2Y2Q0NzM5NGMwZWFjYTU4ZmNlMDFmOWUxYmIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfNjY3MzMyZWRlZjhmNDdlYTgyNTBhNDcyMzEwZGY4MmMuYmluZFBvcHVwKHBvcHVwXzY2YWNkNzVlMjYyMjQyOThhMTU3NTkxNjYzMjhkNWNmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzU0MmExZWFmNjcxZTQ1YjBiMTBiODA1YmY0YWJjMzc2ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTM0Ljk5NjQ5NjMsIC02NC45NjcyODE3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzhiNmUyMzNhNjMzZDRlODdiY2VmMzg2MzBlODFjZDNmID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82MjljZjY4OWU5OWI0YjQyYTc2NDQ2NWM1ODY2YWFlYiA9ICQoYDxkaXYgaWQ9Imh0bWxfNjI5Y2Y2ODllOTliNGI0MmE3NjQ0NjVjNTg2NmFhZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFyZ2VudGluYTogMi4wJTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF84YjZlMjMzYTYzM2Q0ZTg3YmNlZjM4NjMwZTgxY2QzZi5zZXRDb250ZW50KGh0bWxfNjI5Y2Y2ODllOTliNGI0MmE3NjQ0NjVjNTg2NmFhZWIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfNTQyYTFlYWY2NzFlNDViMGIxMGI4MDViZjRhYmMzNzYuYmluZFBvcHVwKHBvcHVwXzhiNmUyMzNhNjMzZDRlODdiY2VmMzg2MzBlODFjZDNmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzViMzczNzA4N2Y5YTQ0MDliNzc5MDI4N2E0YjExNzFlID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTM0Ljk5NjQ5NjMsIC02NC45NjcyODE3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzYwYWM5YjVkMDZiMjRiNWRhNmE4ZjlmMDljODNjZGMxID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF85OTJmOTFlMDQyMjQ0NjJhYjk5MzJhMDE0YmI0NDAxMiA9ICQoYDxkaXYgaWQ9Imh0bWxfOTkyZjkxZTA0MjI0NDYyYWI5OTMyYTAxNGJiNDQwMTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFyZ2VudGluYTogMi4wJTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82MGFjOWI1ZDA2YjI0YjVkYTZhOGY5ZjA5YzgzY2RjMS5zZXRDb250ZW50KGh0bWxfOTkyZjkxZTA0MjI0NDYyYWI5OTMyYTAxNGJiNDQwMTIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfNWIzNzM3MDg3ZjlhNDQwOWI3NzkwMjg3YTRiMTE3MWUuYmluZFBvcHVwKHBvcHVwXzYwYWM5YjVkMDZiMjRiNWRhNmE4ZjlmMDljODNjZGMxKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzRkZTAzNWE5ZTViZjQyODdiZTY3NjcwZDlhNzVmNDc3ID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbLTI0Ljc3NjEwODYsIDEzNC43NTVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNDAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNjgxNDM4ZGZkMjkyNDYyMWE4NWI1MTg2NGNmYjg2OTYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2NmNzViMmM1NTk3MjQ2M2I5MjJkYmY3YzNjMjAyYzliID0gJChgPGRpdiBpZD0iaHRtbF9jZjc1YjJjNTU5NzI0NjNiOTIyZGJmN2MzYzIwMmM5YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QXVzdHJhbGlhOiAyLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzY4MTQzOGRmZDI5MjQ2MjFhODViNTE4NjRjZmI4Njk2LnNldENvbnRlbnQoaHRtbF9jZjc1YjJjNTU5NzI0NjNiOTIyZGJmN2MzYzIwMmM5Yik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV80ZGUwMzVhOWU1YmY0Mjg3YmU2NzY3MGQ5YTc1ZjQ3Ny5iaW5kUG9wdXAocG9wdXBfNjgxNDM4ZGZkMjkyNDYyMWE4NWI1MTg2NGNmYjg2OTYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfZjU2YjgwNTM4ODVmNDljOGJkZTExZDhmNzhhYzU5MjcgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFstMjQuNzc2MTA4NiwgMTM0Ljc1NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA0MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9kY2YzMjRlNmZhOWU0NDg3OTdkYWM1YmM1OWJlNTRhNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzk3YWM1MDQ3M2I2NGNjMTk1MmRiNWNkMmU4ZTE5NGUgPSAkKGA8ZGl2IGlkPSJodG1sXzM5N2FjNTA0NzNiNjRjYzE5NTJkYjVjZDJlOGUxOTRlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BdXN0cmFsaWE6IDIuMCU8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZGNmMzI0ZTZmYTllNDQ4Nzk3ZGFjNWJjNTliZTU0YTQuc2V0Q29udGVudChodG1sXzM5N2FjNTA0NzNiNjRjYzE5NTJkYjVjZDJlOGUxOTRlKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX2Y1NmI4MDUzODg1ZjQ5YzhiZGUxMWQ4Zjc4YWM1OTI3LmJpbmRQb3B1cChwb3B1cF9kY2YzMjRlNmZhOWU0NDg3OTdkYWM1YmM1OWJlNTRhNCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9kMTVkYTJhOGYwNDY0ODkwYTFhZmQ3ZTkzYjAxODE0MCA9IEwuY2lyY2xlKAogICAgICAgICAgICAgICAgWzM1LjAwMDA3NCwgMTA0Ljk5OTkyN10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiY3JpbXNvbiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJjcmltc29uIiwgImZpbGxPcGFjaXR5IjogMC4yLCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA0MDAwMC4wLCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2JlN2FkOTc4ODYyNTRkNDFhY2E3NTAwYzM0NTBhMTllKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF81NjgwYTQ5MjE4NzQ0Y2VlOTExNDM0MjVhN2VlNTI3YiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNDc0YzQwZjk4MzBjNDkwMjg3Y2U0NDk1NDU1YWExZDYgPSAkKGA8ZGl2IGlkPSJodG1sXzQ3NGM0MGY5ODMwYzQ5MDI4N2NlNDQ5NTQ1NWFhMWQ2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaGluYTogMi4wJTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF81NjgwYTQ5MjE4NzQ0Y2VlOTExNDM0MjVhN2VlNTI3Yi5zZXRDb250ZW50KGh0bWxfNDc0YzQwZjk4MzBjNDkwMjg3Y2U0NDk1NDU1YWExZDYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfZDE1ZGEyYThmMDQ2NDg5MGExYWZkN2U5M2IwMTgxNDAuYmluZFBvcHVwKHBvcHVwXzU2ODBhNDkyMTg3NDRjZWU5MTE0MzQyNWE3ZWU1MjdiKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlXzExNGIxNmYxYTczYjRiYWI5M2RhYjNlMTAyNWI5ZjRkID0gTC5jaXJjbGUoCiAgICAgICAgICAgICAgICBbMzUuMDAwMDc0LCAxMDQuOTk5OTI3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJjcmltc29uIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogImNyaW1zb24iLCAiZmlsbE9wYWNpdHkiOiAwLjIsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDQwMDAwLjAsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmU3YWQ5Nzg4NjI1NGQ0MWFjYTc1MDBjMzQ1MGExOWUpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzE4MGFkYjhlY2MxODRjYWI5ZTRkZTlmNzM0OTkyY2E4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jZjJmZDFlYjUxZDY0MjU2YTc5YjlhZjg2MTA1Y2EzZiA9ICQoYDxkaXYgaWQ9Imh0bWxfY2YyZmQxZWI1MWQ2NDI1NmE3OWI5YWY4NjEwNWNhM2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoaW5hOiAyLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzE4MGFkYjhlY2MxODRjYWI5ZTRkZTlmNzM0OTkyY2E4LnNldENvbnRlbnQoaHRtbF9jZjJmZDFlYjUxZDY0MjU2YTc5YjlhZjg2MTA1Y2EzZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8xMTRiMTZmMWE3M2I0YmFiOTNkYWIzZTEwMjViOWY0ZC5iaW5kUG9wdXAocG9wdXBfMTgwYWRiOGVjYzE4NGNhYjllNGRlOWY3MzQ5OTJjYTgpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfYzUyODdkOWI1MzJkNDgwNTgzYWE1MWUwN2ZiZGU2ZjAgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Ni4zODQ5NTE3OTk5OTk5OTYsIDEzLjI1NjA3NTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfN2FkMmIwM2FhNjgyNDM3MWE0YzQ2NjQ2YTQxMjg3Y2YgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzMwZWMzMTMzMTQxYTQ1ODI4Y2ZhOTRkYzBkYjg3ZTY4ID0gJChgPGRpdiBpZD0iaHRtbF8zMGVjMzEzMzE0MWE0NTgyOGNmYTk0ZGMwZGI4N2U2OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+bm90IGZvdW5kOiAxLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzdhZDJiMDNhYTY4MjQzNzFhNGM0NjY0NmE0MTI4N2NmLnNldENvbnRlbnQoaHRtbF8zMGVjMzEzMzE0MWE0NTgyOGNmYTk0ZGMwZGI4N2U2OCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9jNTI4N2Q5YjUzMmQ0ODA1ODNhYTUxZTA3ZmJkZTZmMC5iaW5kUG9wdXAocG9wdXBfN2FkMmIwM2FhNjgyNDM3MWE0YzQ2NjQ2YTQxMjg3Y2YpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfMjQ0NjAwYWIxMjQxNDgwODk2NmI4NDFlMTUyNmEzZmYgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Ni4zODQ5NTE3OTk5OTk5OTYsIDEzLjI1NjA3NTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfOWQ5MmRkZTdiYWIwNDk5N2I4YTZkYWZjMTRkNTA5YTkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Q3NTc0OWFmZTkwNzRiYmNhNjlmMjRlZDdkYzc1MzFmID0gJChgPGRpdiBpZD0iaHRtbF9kNzU3NDlhZmU5MDc0YmJjYTY5ZjI0ZWQ3ZGM3NTMxZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+bm90IGZvdW5kOiAxLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzlkOTJkZGU3YmFiMDQ5OTdiOGE2ZGFmYzE0ZDUwOWE5LnNldENvbnRlbnQoaHRtbF9kNzU3NDlhZmU5MDc0YmJjYTY5ZjI0ZWQ3ZGM3NTMxZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV8yNDQ2MDBhYjEyNDE0ODA4OTY2Yjg0MWUxNTI2YTNmZi5iaW5kUG9wdXAocG9wdXBfOWQ5MmRkZTdiYWIwNDk5N2I4YTZkYWZjMTRkNTA5YTkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfNmU5OGVhZTNhOTRiNDVmMDkyODhkOGIxZWM5YTJjNDIgPSBMLmNpcmNsZSgKICAgICAgICAgICAgICAgIFs0Ni4zODQ5NTE3OTk5OTk5OTYsIDEzLjI1NjA3NTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImNyaW1zb24iLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiY3JpbXNvbiIsICJmaWxsT3BhY2l0eSI6IDAuMiwgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogMjAwMDAuMCwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9iZTdhZDk3ODg2MjU0ZDQxYWNhNzUwMGMzNDUwYTE5ZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWIyN2ExZGIxMjMzNDNlNWJkNmM5MjdmMzU0YTc0N2QgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2ViODc4ZjZkYjA3ZDQzOGViNTQ5NGIyMTBkOWQzN2MzID0gJChgPGRpdiBpZD0iaHRtbF9lYjg3OGY2ZGIwN2Q0MzhlYjU0OTRiMjEwZDlkMzdjMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+bm90IGZvdW5kOiAxLjAlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzViMjdhMWRiMTIzMzQzZTViZDZjOTI3ZjM1NGE3NDdkLnNldENvbnRlbnQoaHRtbF9lYjg3OGY2ZGIwN2Q0MzhlYjU0OTRiMjEwZDlkMzdjMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV82ZTk4ZWFlM2E5NGI0NWYwOTI4OGQ4YjFlYzlhMmM0Mi5iaW5kUG9wdXAocG9wdXBfNWIyN2ExZGIxMjMzNDNlNWJkNmM5MjdmMzU0YTc0N2QpCiAgICAgICAgOwoKICAgICAgICAKICAgIAo8L3NjcmlwdD4= onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Medicinal products and active substance vs adverse events:

Are there certain active substances that appear more often with serious events? 


```python
active_substance_bootstrap = compile_bootstrap(data_subset=data_subset, col ='active_substance',pivot_value='serious',n=50)
```


![png](output_25_0.png)



```python
medicinal_product_bootstrap = compile_bootstrap(data_subset=data_subset, col ='medicinalproduct',pivot_value='serious',n=50)
```


![png](output_26_0.png)


## 2.3 Inspecting similarity between drugs based on the reactions associated with them (as provided by the `reactionmeddrapt` attribute)

In here, we'll create a new data frame containing information about drugs with at least 200 adverse events reported in our sample of data.


```python
drug_reactions = pd.DataFrame()
    
for drug in medicinal_product_bootstrap['medicinalproduct'].unique():
    
    subset = data_subset[data_subset['medicinalproduct']==drug]
    event_count = medicinalproduct_count['medicinalproduct_count'][medicinalproduct_count['medicinalproduct']==drug]
    reactions = subset['reactionmeddrapt'].apply(lambda s : s.lower())
    top5_reaction = str(reactions.value_counts().reset_index().sort_values('reactionmeddrapt',ascending = False).head(5)['index'].unique())
    reaction_list = str([r.lower() for r in subset['reactionmeddrapt']])
    
    drug_duration = subset['drug_duration'].dropna().median()
    drug_reactions = drug_reactions.append(pd.DataFrame(data = {'medicinalproduct':[drug], 'unique_reaction':[reaction_list],\
                                                                'top5_reaction':top5_reaction,'drug_duration':drug_duration,\
                                                               'medicinalproduct_count':event_count}))

drug_reactions['drug_duration_disc'] = [str(s) for s in pd.cut(reaction_df2['drug_duration'],10)]    
    
drug_reactions.reset_index(inplace = True)    
drug_reactions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>medicinalproduct</th>
      <th>unique_reaction</th>
      <th>top5_reaction</th>
      <th>drug_duration</th>
      <th>medicinalproduct_count</th>
      <th>drug_duration_disc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>44</td>
      <td>ZEMPLAR</td>
      <td>['cardiac failure congestive', 'disorientation...</td>
      <td>['adverse event' 'death' 'cardiac disorder' 's...</td>
      <td>363.0</td>
      <td>204</td>
      <td>(292.2, 438.3]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35</td>
      <td>CLOZARIL</td>
      <td>['hypothyroidism', 'death', 'tachycardia', 'gr...</td>
      <td>['death' 'mental impairment' 'malaise' 'hospit...</td>
      <td>571.0</td>
      <td>230</td>
      <td>(438.3, 584.4]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>ARANESP</td>
      <td>['death', 'atrial fibrillation', 'hospitalisat...</td>
      <td>['death' 'hospitalisation' 'investigation' 'ha...</td>
      <td>259.0</td>
      <td>366</td>
      <td>(146.1, 292.2]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31</td>
      <td>GLIVEC</td>
      <td>['chronic graft versus host disease', 'myocard...</td>
      <td>['death' 'blast crisis in myelogenous leukaemi...</td>
      <td>653.0</td>
      <td>245</td>
      <td>(584.4, 730.5]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>EXJADE</td>
      <td>['blood creatinine increased', 'blood creatini...</td>
      <td>['death' 'transplant' 'bone marrow transplant'...</td>
      <td>154.5</td>
      <td>307</td>
      <td>(146.1, 292.2]</td>
    </tr>
  </tbody>
</table>
</div>



#### Generate text similarity from drug reactions using TFIDF:

The TFIDF transformation treats each record as a document (in this case, combination of different drug reaction texts for each drug) and calculates the term frequency of each word adjusted by how often it appears in the rest of the documents. The output is a TFIDF weight matrix, and the inner product provides us with the cosine similarity of each drug as a function of their reaction texts


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer().fit_transform(drug_reactions['unique_reaction'])

reaction_sim = tfidf * tfidf.T.toarray()


reaction_df = pd.DataFrame(reaction_sim, index = drug_reactions['medicinalproduct'],columns=drug_reactions['medicinalproduct']).reset_index()

reaction_df = reaction_df.merge(drug_reactions)

reaction_df.to_csv('openFDA_data/drug_df.csv',index = '')

reaction_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>medicinalproduct</th>
      <th>ZEMPLAR</th>
      <th>CLOZARIL</th>
      <th>ARANESP</th>
      <th>GLIVEC</th>
      <th>EXJADE</th>
      <th>ACLASTA</th>
      <th>ZOMETA</th>
      <th>SENSIPAR</th>
      <th>PROLIA</th>
      <th>FORTEO</th>
      <th>REMICADE</th>
      <th>AFINITOR</th>
      <th>XARELTO</th>
      <th>REBIF</th>
      <th>ELIQUIS</th>
      <th>REVLIMID</th>
      <th>ABILIFY</th>
      <th>LIPITOR</th>
      <th>PRADAXA</th>
      <th>CRESTOR</th>
      <th>HUMIRA</th>
      <th>AVONEX</th>
      <th>DIANEAL LOW CALCIUM PERITONEAL DIALYSIS SOLUTION WITH DEXTROSE</th>
      <th>CHANTIX</th>
      <th>GILENYA</th>
      <th>BYETTA</th>
      <th>NEXIUM</th>
      <th>SOLIRIS</th>
      <th>ENTRESTO</th>
      <th>TECFIDERA</th>
      <th>TYSABRI</th>
      <th>NEULASTA</th>
      <th>NUVARING</th>
      <th>COSENTYX</th>
      <th>COPAXONE</th>
      <th>ENBREL</th>
      <th>MIRENA</th>
      <th>PRISTIQ</th>
      <th>SPIRIVA</th>
      <th>AMPYRA</th>
      <th>NEXPLANON</th>
      <th>ALEVE (CAPLET)</th>
      <th>OTEZLA</th>
      <th>PLAN B ONE-STEP</th>
      <th>NIASPAN</th>
      <th>BOTOX COSMETIC</th>
      <th>index</th>
      <th>unique_reaction</th>
      <th>top5_reaction</th>
      <th>drug_duration</th>
      <th>medicinalproduct_count</th>
      <th>drug_duration_disc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ZEMPLAR</td>
      <td>1.000000</td>
      <td>0.252599</td>
      <td>0.483496</td>
      <td>0.472707</td>
      <td>0.394881</td>
      <td>0.375798</td>
      <td>0.137793</td>
      <td>0.037087</td>
      <td>0.431160</td>
      <td>0.203442</td>
      <td>0.172403</td>
      <td>0.456525</td>
      <td>0.025581</td>
      <td>0.125768</td>
      <td>0.478030</td>
      <td>0.292765</td>
      <td>0.107369</td>
      <td>0.163400</td>
      <td>0.139908</td>
      <td>0.187436</td>
      <td>0.086077</td>
      <td>0.166355</td>
      <td>0.239349</td>
      <td>0.074171</td>
      <td>0.098731</td>
      <td>0.045203</td>
      <td>0.146913</td>
      <td>0.106298</td>
      <td>0.430105</td>
      <td>0.178925</td>
      <td>0.107591</td>
      <td>0.052789</td>
      <td>0.054577</td>
      <td>0.082578</td>
      <td>0.039709</td>
      <td>0.061407</td>
      <td>0.011784</td>
      <td>0.021203</td>
      <td>0.007192</td>
      <td>0.109155</td>
      <td>0.095180</td>
      <td>0.448031</td>
      <td>0.082341</td>
      <td>0.006227</td>
      <td>0.009502</td>
      <td>0.001168</td>
      <td>44</td>
      <td>['cardiac failure congestive', 'disorientation...</td>
      <td>['adverse event' 'death' 'cardiac disorder' 's...</td>
      <td>363.0</td>
      <td>204</td>
      <td>(292.2, 438.3]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CLOZARIL</td>
      <td>0.252599</td>
      <td>1.000000</td>
      <td>0.428577</td>
      <td>0.403398</td>
      <td>0.433050</td>
      <td>0.406319</td>
      <td>0.175758</td>
      <td>0.160283</td>
      <td>0.427193</td>
      <td>0.311573</td>
      <td>0.267973</td>
      <td>0.480242</td>
      <td>0.052258</td>
      <td>0.200514</td>
      <td>0.334922</td>
      <td>0.454905</td>
      <td>0.209765</td>
      <td>0.335444</td>
      <td>0.158351</td>
      <td>0.242956</td>
      <td>0.181290</td>
      <td>0.233469</td>
      <td>0.262073</td>
      <td>0.216603</td>
      <td>0.349085</td>
      <td>0.236472</td>
      <td>0.183227</td>
      <td>0.302412</td>
      <td>0.350962</td>
      <td>0.293369</td>
      <td>0.214015</td>
      <td>0.083363</td>
      <td>0.054784</td>
      <td>0.140706</td>
      <td>0.070383</td>
      <td>0.113911</td>
      <td>0.015383</td>
      <td>0.150585</td>
      <td>0.043888</td>
      <td>0.164596</td>
      <td>0.027155</td>
      <td>0.054803</td>
      <td>0.103513</td>
      <td>0.009472</td>
      <td>0.055249</td>
      <td>0.069706</td>
      <td>35</td>
      <td>['hypothyroidism', 'death', 'tachycardia', 'gr...</td>
      <td>['death' 'mental impairment' 'malaise' 'hospit...</td>
      <td>571.0</td>
      <td>230</td>
      <td>(438.3, 584.4]</td>
    </tr>
  </tbody>
</table>
</div>



### 

## 2.4 Let's see if we can visualize it using an interactive network:


```python
def create_graph(reaction_df):
    
    g = nx.Graph()
    g.inspection_policy=NodesAndLinkedEdges()
    
    # set some attributes for graph labeling:
    count = []
    duration = []
    top5_reaction = []
    nx.set_node_attributes(g, count, 'count')
    nx.set_node_attributes(g, duration, 'duration_bracket')
    nx.set_node_attributes(g, top5_reaction, 'top5_reaction')
    
    # list of drugs:
    drugs = [d for d in reaction_df['medicinalproduct']]

    for i in range(0,len(reaction_df)):
        
        drug = reaction_df['medicinalproduct'][i]
        duration = reaction_df['drug_duration_disc'][i]
        top5 = reaction_df['top5_reaction'][i]        
        
        # make sure the attributes are assigned to exisiting nodes:
        if drug in g:
            
            g.nodes[drug]["count"] = reaction_df['medicinalproduct_count'][i]
            g.nodes[drug]["duration_bracket"] = duration
            g.nodes[drug]['top5_reactions'] = top5
            
        else:

            g.add_node(drug)
            g.nodes[drug]["count"] = reaction_df['medicinalproduct_count'][i]
            g.nodes[drug]["duration_bracket"] = duration
            g.nodes[drug]['top5_reactions'] = top5
            print(g)

        for j in range(1,len(drugs)):

            # assign weight from text similarity of reactions:
            weight = reaction_df[reaction_df.columns[j]][i]

            drug1 = reaction_df['medicinalproduct'][i]
            drug2 = reaction_df['medicinalproduct'][j]
            # create an edge between two drugs if the similarity is at least 0.3 (1 = max, 0 = min)
            if (drug1 != drug2) & (float(weight) > 0.3) :
                
                if g.has_edge(drug1, drug2):

                    pass

                else:

                    g.add_edge(drug1, drug2, attr_dict={"weight":weight},length = 10*weight)
                    
                    
    neighbourless_drugs = []
    
    for drug in g.nodes():

        if len(list(g.neighbors(drug))) == 0:

            neighbourless_drugs.append(drug)

    for drug in neighbourless_drugs:
        
        g.remove_node(drug)
                       


    return g

g = create_graph(reaction_df)
```


```python
from bokeh.plotting import *
from bokeh.models import HoverTool, BoxSelectTool, TapTool
from bokeh.io import output_notebook, save
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models.graphs import from_networkx
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.events import Tap
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes

def hex_generator(palette):
    
    try:
    
        hex_code = ['#%02x%02x%02x' % (int(255*val[0]), int(255*val[1]), int(255*val[2])) for val in palette]
        
    except:
        hex_code = []
    
    return hex_code
    

def network_viz(g, view_in_notebook = True, output_html = 'drug_netwrok.html', title = "Network visualization of drugs based on similarity of reactions and median duration until adverse events"):
    # color by duration of drug taken:
    g.inspection_policy=NodesAndLinkedEdges()
    color_att = set([g.nodes[node]['duration_bracket'] for node in g.nodes])

    palette = sns.color_palette("BuGn_r", len(color_att))
    sizes = []
    drug_labels = []
    reaction_labels =[]
    count_labels = []
    # The above palette is given in RGB [0-1] range
    # bokeh requires colours be defined in hex RGB
    hex_palette = hex_generator(palette)
    
    # set palette map in hex code:
    palette_map = {duration:hex_palette[i] for i, duration in enumerate(color_att)}

    #pos = nx.layout.spring_layout(g, iterations=1000)
    
    pos = nx.layout.fruchterman_reingold_layout(g, iterations = 1000)
    
    # ------------------------------------------------

    # Determine appearance of nodes, representing drugs
    
    xs, ys, colours, labels, sizes = [], [], [], [], []

    for i, (node_name, coords) in enumerate(pos.items()):
        
        # X and Y co-ordinates of the drug
        xs.append(coords[0])
        ys.append(coords[1])


        d = g.nodes[node_name]['duration_bracket']
        
        colours.append(palette_map[d])

        r = g.nodes[node_name]['top5_reactions']
                
        count = g.nodes[node_name]["count"]
        
        drug_labels.append(node_name)
        
        reaction_labels.append(r)
        
        count_labels.append(str(count))
        
        labels.append(label)

        size = min(0.05*count,30)

        sizes.append(size)
        
    # set data for nodes on the graph:
    node_source = ColumnDataSource(data=dict(x=xs, y=ys, label=labels, color=colours, size=sizes,drug = drug_labels, top5_reactions = reaction_labels, count_labels = count_labels))

    xlist, ylist, weight = [], [], []

    for node_A, node_B, data in g.edges(data=True):

        x1, y1 = pos[node_A]
        x2, y2 = pos[node_B]

        xlist += [x1, x2, float("NaN")]
        ylist += [y1, y2, float("NaN")]


    line_source = ColumnDataSource(data=dict(xs=xlist, ys=ylist))

    # ---------------------------

    if view_in_notebook:
        
        output_notebook(hide_banner=True)

    f1 = figure(plot_width=1200, plot_height=1000, tools="pan,wheel_zoom,box_zoom,reset,hover")

    f1.grid.grid_line_width = 0
    f1.axis.visible = False

    # Draw the lines between nodes
    f1.line(x="xs", y="ys", line_width=0.5, source=line_source, color="#000000", line_alpha=0.35)

    # Draw the nodes
    f1.circle("x", "y", source=node_source, name="node", size="size", color="color", line_width=0.5, line_alpha=0.75, line_color="#000000")

    # Attach the HoverTool to the drug nodes to display their label
    
    tooltips = [("drug","@drug"),("top reactions","@top5_reactions"),("adverse event count","@count_labels")]
    hover = f1.select(dict(type=HoverTool))
    hover.tooltips = tooltips 
    hover.point_policy = "snap_to_data"
    
    
    

    hover.names = ["node"]

    # Legend

    f1.title.text = "Network visualization of drugs based on similarity of reactions and median duration until adverse events"
    f1.title.align = "center"
    f1.title.text_color = "grey"
    f1.title.text_font_size = "14px"

    text = [index for _,index in enumerate(palette_map)]

    colors = [palette_map[index] for _,index in enumerate(palette_map)]

    x_pos = [-1+0.05*i for i in range(0,len(colors))] 
    y_pos = [-1 for i in range(0,len(colors))] 
    size_legend = [18 for i in range(0,len(colors))]

    f1.circle(x_pos, y_pos, size=size_legend, line_width=0.5, line_alpha=0.75, line_color="#000000", color=colors,)
    f1.text(x_pos, y_pos, text=text, text_align="center", text_font_size="8pt",angle = 45)
    
    
    # color:
    
    #f1.background_fill_color = "beige"
    #f1.background_fill_alpha = 0.6

    #show(f1)
    if view_in_notebook:
        show(f1)
    else:
        filename = save(f1, filename=output_html, title=title)
        
    return f1


  
    
    
```


```python
network_viz(g)
```










<div class="bk-root" id="3c65ea1a-2b5e-49f7-aeb8-2a177c579ec6" data-root-id="18452"></div>





# 3. Take-away:

In this exploratory data analysis, we utilized several tools to understand a bit more about the adverse event database from FDA. In here, we have learned how to gather data using API calls, understand the data structure behind the outputs, did a ton of data wrangling to flatten the results, and analyzing the data to see what's out there. There are a few things I found pretty interesting: 


- **On mapping the adverse events** : The sample of data represents 125 different countries, in which US has taken up the majority of the records (perhaps to be expected). But, it's surprising that the ratio of _serious_ adverse events from the US are about 40%, compared to closer to 90% for most other countries. Perhaps this could be a result of differences in reporting standards or metrics? 

- **Next Step** : In this exercise, I worked on using unsupervised learning analysis to understand the relationship of the drugs, it would be really interesting to start some work on supervised learning as well. For example, 

     - predict types of serious adverse reactions based on the drug, active ingredient, and patient information such as their illness, how long they've taken the drug, age, gender and potentially other biomarkers?
     - predict quality problem of batches of drugs based on the time and loation these events were reported 
     - monitor rare drug reaction or anomaliy based on text analysis 

