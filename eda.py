#!/usr/bin/env python
# coding: utf-8

# # OpenFDA exploratory notebook
# 
# Sami Furst, May 2020
# ## Purpose of this notebook : 
# 
# ### 1. get a sense of the data
# ### 2. try to find something interesting in the data through feature generation and visualization
# 
# 

# In[15]:


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

get_ipython().run_line_magic('matplotlib', 'inline')


# ================================================================================================================================================================================================
# # 1. Let's get a sense of the sample data:
# 
# - The sample of data we pulled has 63800 rows and 85 columns, with event dates ranging from January 2010 to March 2020. Note: the total data set consists of millions of records, and the data we are working with here is only a small part of it.
# 
# - There are quite a few columns with incomplete data, we'll need to do some data QC to select more useful features
# 
# - While some data is not missing, they may still be non-informative if e.g. there is only 1 unique value
# 
# 
# 
# ### helper functions:
# 

# In[ ]:


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


# ## 1.1 First pass of the data:

# In[3]:


data = pd.read_csv('openFDA_data/data/combined_weekly_sample_20200515.csv');


# In[4]:


data.shape


# ###  How many variables are missing data?

# In[5]:


plt.figure(figsize=(15,20))
missing_data_count = sns.heatmap(data.isnull().transpose(), cbar = True)
plt.savefig('missing_data_count.png')
#sns.barplot(data_var)
#missing_data_count.savefig("missing_data_count.png")


# ### How many unique levels there are across the variables?

# In[6]:


unique_levels = pd.DataFrame(data = {'column':data.columns,'nunique':data.nunique()}).sort_values('nunique')
plt.figure(figsize=(15,20))
sns.barplot(data=unique_levels, y ='column', x='nunique')
plt.savefig('unique_levels.png')


# ================================================================================================================================================================================================
# 
# ## 1.2 Data cleaning & additional feature generation :
# 
# Horizontal (column): by dropping a feature using the following criteria:
# 
# - % of missing data > 80%
# 
# - has only 1 unique level
# 
# - represents a nested key (those were already unested in the previous data gathering function so information should be contained in other columns)
# 
# 
# 
# 
# ### new features:
# 
# - patient age in years
# - drug duration (taken as difference between `drugstartdate` and `drugenddate`, if exists
# - lat and lon of reporter countries (using `geocode` package)
# - count of reports by drugs, country and active substance

# In[ ]:



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


# In[7]:


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


# In[8]:


len(data_subset['patient_age_year'].dropna())


# ### Let's take a look at the variables that got removed:

# In[9]:


set(data.columns) - set(informative_vars)


# ================================================================================================================================================================================================
# # 2. Now some data analysis and visualization:
# 
# 
# ### More helper functions:

# In[10]:


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


# 
# ## 2.1 Types of adverse events:
# 
# Let's see the breakdown of productions by adverse events:
# 
# From the metadata, we know that the flag `serious` indicates whether or not the adverse events result in serious conditions which are: death, a life threatening condition, hospitalization, disability, congenital anomaly, and other. 
# 
# In this sample, more than 50% of reports consist of serious adverse event.
# 

# In[12]:


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


# ## 2.2 Compare adverse events by countries and products
# 
# In order to visualize adverse events on a map, we need to standardize country names and get geocode (lat/lon), also, since countries are represented in widely different number. A straight-forward comparison may not be fair, so let's try to offset the imbalance by inspecting the rate of serious adverse event using bootstrap samples.  Namely, we'll only show countries that have at least 200 cases, and we'll do a stratified sampling of 100 reports per country, then extract the rate of adverse events  

# In[344]:


pivot_val = 'seriousnesslifethreatening'
country_bootstrap = compile_bootstrap(data_subset = data_subset, col ='country_short',pivot_value=pivot_val,n=50,plot=False)

country_bootstrap_map = country_bootstrap.merge(country_lookup[['country_short','lat','lon']])


# In[347]:


m = folium_map(data2 = country_bootstrap_map, title = 'Rate of serious adverse effects (countries with record > 100) that led to life threatening conditions',map_col='country_short', data_col='median',scale=2000000)

path=pivot_val +'_by_country.html'
m.save(path)
m


# ### Medicinal products and active substance vs adverse events:
# 
# Are there certain active substances that appear more often with serious events? 

# In[27]:


active_substance_bootstrap = compile_bootstrap(data_subset=data_subset, col ='active_substance',pivot_value='serious',n=50)


# In[28]:


medicinal_product_bootstrap = compile_bootstrap(data_subset=data_subset, col ='medicinalproduct',pivot_value='serious',n=50)


# ## 2.3 Inspecting similarity between drugs based on the reactions associated with them (as provided by the `reactionmeddrapt` attribute)
# 
# In here, we'll create a new data frame containing information about drugs with at least 200 adverse events reported in our sample of data.

# In[382]:


drug_reactions = pd.DataFrame()
    
for drug in medicinal_product_bootstrap['medicinalproduct'].unique():
    
    subset = data_subset[data_subset['medicinalproduct']==drug]
    event_count = medicinalproduct_count['medicinalproduct_count'][medicinalproduct_count['medicinalproduct']==drug]
    reactions = subset['reactionmeddrapt'].apply(lambda s : s.lower())
    top5_reaction = str(reactions.value_counts().reset_index().sort_values('reactionmeddrapt',ascending = False).head(5)['index'].unique())
    reaction_list = str([r.lower() for r in subset['reactionmeddrapt']])
    
    drug_duration = subset['drug_duration'].dropna().median()
    drug_reactions = drug_reactions.append(pd.DataFrame(data = {'medicinalproduct':[drug], 'unique_reaction':[reaction_list],                                                                'top5_reaction':top5_reaction,'drug_duration':drug_duration,                                                               'medicinalproduct_count':event_count}))

drug_reactions['drug_duration_disc'] = [str(s) for s in pd.cut(reaction_df2['drug_duration'],10)]    
    
drug_reactions.reset_index(inplace = True)    
drug_reactions.head()


# #### Generate text similarity from drug reactions using TFIDF:
# 
# The TFIDF transformation treats each record as a document (in this case, combination of different drug reaction texts for each drug) and calculates the term frequency of each word adjusted by how often it appears in the rest of the documents. The output is a TFIDF weight matrix, and the inner product provides us with the cosine similarity of each drug as a function of their reaction texts

# In[564]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer().fit_transform(drug_reactions['unique_reaction'])

reaction_sim = tfidf * tfidf.T.toarray()


reaction_df = pd.DataFrame(reaction_sim, index = drug_reactions['medicinalproduct'],columns=drug_reactions['medicinalproduct']).reset_index()

reaction_df = reaction_df.merge(drug_reactions)

reaction_df.to_csv('openFDA_data/drug_df.csv',index = '')

reaction_df.head(2)


# ### 

# ## 2.4 Let's see if we can visualize it using an interactive network:

# In[556]:


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


# In[562]:


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


  
    
    


# In[484]:


network_viz(g)


# # 3. Take-away:
# 
# In this exploratory data analysis, we utilized several tools to understand a bit more about the adverse event database from FDA. In here, we have learned how to gather data using API calls, understand the data structure behind the outputs, did a ton of data wrangling to flatten the results, and analyzing the data to see what's out there. There are a few things I found pretty interesting: 
# 
# 
# - **On mapping the adverse events** : The sample of data represents 125 different countries, in which US has taken up the majority of the records (perhaps to be expected). But, it's surprising that the ratio of _serious_ adverse events from the US are about 40%, compared to closer to 90% for most other countries. Perhaps this could be a result of differences in reporting standards or metrics? 
# 
# - **Next Step** : In this exercise, I worked on using unsupervised learning analysis to understand the relationship of the drugs, it would be really interesting to start some work on supervised learning as well. For example, 
# 
#      - predict types of serious adverse reactions based on the drug, active ingredient, and patient information such as their illness, how long they've taken the drug, age, gender and potentially other biomarkers?
#      - predict quality problem of batches of drugs based on the time and loation these events were reported 
#      - monitor rare drug reaction or anomaliy based on text analysis 
# 
