import os
import requests
import constant
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import time
from joblib import delayed, Parallel
import datetime
from ruamel.yaml import YAML
from pathlib import Path
path = Path('fields.yaml')
yaml = YAML(typ='safe')
fields = yaml.load(path)

path = Path('fields.yaml')
yaml = YAML(typ='safe')
fields = yaml.load(path)

keys = constant.keys

def get_active_sub_name(row):
    val = 'NA'
    try:
        val = row['activesubstancename']
    except:
        pass
    return(val)


def fda_api(key,sd,ed):
    
    time.sleep(1)
    
    # initialize output dataframe
    output = pd.DataFrame()
    
    base_url = 'https://api.fda.gov/drug/event.json?api_key='+key
    
    search_term = '&search=receivedate:%5B'+str(sd)+'+TO+'+str(ed)+'%5D&sort=receivedate:desc&limit=100'
    
    url = base_url + search_term
    
    print(url)
    
    r = requests.get(url)
    
    if r.status_code == 200:   
        
        print('query succesful, getting data now:')
        
        data = r.json()
            

        for result in data.get('results', []):
            

            output = output.append(json_normalize(result))  
            
        if ('patient.reaction' in output.columns) and ('patient.drug' in output.columns)  :
            
            reaction_list = [val[0] for val in output['patient.reaction']]
            
            drug_list = [val[0] for val in output['patient.drug']]
            
            print('unlisting nested information about drug and patient reaction')

            output2 = output.join(pd.DataFrame(data = reaction_list)).join(pd.DataFrame(data=drug_list))

            return data
        
        else:
            print('no information about patient reaction and drug')
            return data
    
    else:
        print('Query unsuccesful, status_code: '+str(r.status_code))
        return None

def get_dates_df(start_date, end_date = None,message = True, freq = 'M'):
    
    if not end_date:
    
        today = datetime.datetime.today().strftime('%Y%m%d')
        
        end = today
        
        print('end date is '+ str(today))
    else:
        end = end_date
    
    date_list = [str(s) for s in pd.date_range(start = start_date, end = end,freq=freq).strftime('%Y%m%d')]
    
    #print(date_list)
    
    dates_df = pd.DataFrame(data = {'sd_list':date_list[0:-1],'ed_list':date_list[1:len(date_list)]})
        
    if message:
        
        min_sd = dates_df['sd_list'].min()
        
        max_ed = dates_df['ed_list'].max()

        if freq == 'M':
            
            interval = 'monthly'
            
        elif freq == 'W':
            
            interval = 'weekly'
            
        elif freq == 'D':
            
            interval = 'daily'

        print('List contains '+ str(len(dates_df))+ ' '+interval +' intervals between '+ str(min_sd)+' and '+str(max_ed))
    return dates_df


def get_key_type(fields):
    
    try:
        
        field_key_list = [s for s in fields.get('properties',[]).keys()]

    except:
        
        raise Exception('Invalid input')
        
        field_key_list = []
    
    if field_key_list:
        
        field_prop = [fields.get('properties',[]).get(s,'None') for s in field_key_list]


        field_types = [field_prop[i]['type'] for i in range(0,len(field_prop))]
        
        #print(field_types)


    
        field_objs = [i for i in range(len(field_types)) if (field_types[i] == 'array') or (field_types[i] == 'object')]

        print('The following keys are arrays or objects:')
        print([field_key_list[i] for i in field_objs])
        #print([field_prop[i] for i in field_objs])

        return {'field_key_list':field_key_list,'nested_keys':field_objs,'nested_key_names':[field_key_list[i] for i in field_objs]}
    
    else:
        return None

def get_openfda_ind(row0,s):
#    ind = 0
    output = []
    row = row0['patient']['drug']
    ind = len(row)
    for i in range(0,ind):
        try:
            if 'openfda' in row0['patient']['drug'][i].keys():
                output = row0['patient']['drug'][i].get('openfda','None').get(s,'None') 

                return(output)
            else:
                return None
        except:
            return None

def replace_square_bracket(s):
    
    return str(s).replace('[','').replace(']','')



class get_fda_keys:
    

    def __init__(self,fields):
        
        self.key_types = get_key_type(fields=fields)
        field_key_list = self.key_types['field_key_list']
        nested_keys = self.key_types['nested_key_names']        
    # nested level  = 0 
        self.single_keys = [i for i in set(field_key_list) - set(nested_keys)]


        # nested level = 1
        self.patient_keys = [val for val in set(fields['properties']['patient']['properties'].keys())- {'drug','reaction'}]

        self.reaction_keys = [val for val in set(fields['properties']['patient']['properties']['reaction']['items']['properties'].keys())]
        # nested level = 2
        self.drug_keys = [val for val in set(fields['properties']['patient']['properties']['drug']['items']['properties'].keys()) - {'openfda'}]

        # nested level = 3
        self.openfda_keys = [val for val in set(fields['properties']['patient']['properties']['drug']['items']['properties']['openfda']['properties'].keys())]
        
        self.primsource_keys = [val for val in set(fields['properties']['primarysource']['properties'].keys()) - {'literaturereference'}]

        
def get_pdf(sd,ed,fields = fields, keys = keys):
    
    json_out = fda_api(keys['API_KEY'],sd = sd,ed = ed)

    json_out2 = json_out.get('results',[])
    
    fda_keys = get_fda_keys(fields = fields)
    
    single_keys = fda_keys.single_keys
    
    patient_keys = fda_keys.patient_keys
    
    reaction_keys = fda_keys.reaction_keys
    
    drug_keys = fda_keys.drug_keys
    
    openfda_keys = fda_keys.openfda_keys
    
    primsource_keys = fda_keys.primsource_keys
    
    print(primsource_keys)
    
    
    try:
    #patient_val = [json_out2[0]['patient'].get(s) for s in patient_keys]
    
        #qualification = [json_out2[i]['primarysource'].get('qualification','None') for i in range(0, len(json_out2))]
        #reportercountry = [json_out2[i]['primarysource'].get('reportercountry','None') for i in range(0, len(json_out2))]
        
        single_pdf = pd.DataFrame()
        primsource_pdf = pd.DataFrame()
        patient_pdf = pd.DataFrame()
        reaction_pdf = pd.DataFrame()
        drug_pdf = pd.DataFrame()
        openfda_pdf = pd.DataFrame()
     

        for key in single_keys:
            
            value = [json_out2[i].get(key,'None') for i in range(0,len(json_out2))]
            
            single_pdf[key] = value
           # single_pdf = single_pdf.append(pd.DataFrame(data = {key:value}))   
        print('df from unested keys')
        print(single_pdf.shape)
       
        for key in fda_keys.primsource_keys:
    
            primsource_val = [json_out2[i]['primarysource'].get(key,'None') if json_out2[i]['primarysource'] is not None else 'None' for i in range(0,len(json_out2))]
    
            primsource_pdf[key] = primsource_val
           
        for key in patient_keys:
            
            patient_val = [json_out2[i]['patient'].get(key) for i in range(0,len(json_out2))] 
            
            patient_pdf[key] = patient_val#[patient_keys.index(key)]#patient_pdf.append(pd.DataFrame(data = {key:[json_out2[0]['patient'].get(key)]}))   

        # some hard coded stuff until I fully figured out the data structure
        case_event_date = [patient_pdf['summary'][i].get('narrativeincludeclinical','None') if patient_pdf['summary'][i] is not None else 'None' for i in range(0,len(patient_pdf)) ]
        patient_pdf['case_event_date'] = case_event_date
        
        for key in reaction_keys:
            
            reaction_val = [json_out2[i]['patient']['reaction'][0].get(key,'None') for i in range(0, len(json_out2))]
            reaction_pdf[key] = reaction_val
#        reaction = [json_out2[i]['patient']['reaction'][0].get('reactionmeddrapt','None') for i in range(0, len(json_out2))]
#        reaction_outcome = [json_out2[i]['patient']['reaction'][0].get('reactionoutcome','None') for i in range(0, len(json_out2))]
        
        

#        patient_pdf['reaction'] = reaction
#        patient_pdf['reaction_outcome'] = reaction_outcome
        
        
        print('df from nested patient keys')
        print(patient_pdf.shape)
        
        

        for key in drug_keys:
            drug_val = [json_out2[i]['patient']['drug'][0].get(key) for i in range(0,len(json_out2)) ]# for s in drug_keys]
            drug_pdf[key] = drug_val#[drug_keys.index(key)]

        active_substance = [drug_pdf['activesubstance'][i].get('activesubstancename','None') if drug_pdf['activesubstance'][i] is not None else 'None' for i in range(0,len(drug_pdf))  ]
        drug_pdf['active_substance'] = active_substance
        
        print('df from nested drug keys')
        print(drug_pdf.shape)
        
        

        for key in openfda_keys:
            openfda_val = [get_openfda_ind(json_out2[i],key) for i in range(0,len(json_out2))] 
            openfda_pdf[key] = openfda_val#[openfda_keys.index(key)]
            
        print('df from nested openfda keys')
        print(openfda_pdf.shape)
        total_pdf = pd.concat([single_pdf,patient_pdf,reaction_pdf,drug_pdf,openfda_pdf,primsource_pdf],axis = 1)
    

        return total_pdf
    except:
        return None
    
    
            
