# helper functions:
import helpers
import argparse
from ruamel.yaml import YAML
from pathlib import Path
path = Path('fields.yaml')
yaml = YAML(typ='safe')
fields = yaml.load(path)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-sd','--start-date', help='start date for querying based on receivedate, must be in the form of YYYYMMDD', required=False,default='20190101')
parser.add_argument('-ed','--end-date', help='end date for querying based on receivedate, must be in the form of YYYYMMDD', required=False,default = '20190201')

args = vars(parser.parse_args())

print(args)

if __name__ == "__main__":
    print('getting a test set of data:')
    test_df = helpers.get_pdf(sd = args['start_date'],ed = args['end_date'],fields = fields)
    
    if len(test_df)>2:
        min_date = test_df['receivedate'].min()
        max_date = test_df['receivedate'].max()    
        test_df.to_csv('test_df.csv',index = '')
        print('Saved data with receiptdates ranging from '+ min_date + ' to '+ max_date + ' with '+str(len(test_df))+' records')
    