# Purpose of the project


This is an exercise of exploratoray analysis using data provided by the [openFDA API](https://open.fda.gov/apis/drug/). Specifically, I'm interesting in the drug event end point where information of an adverse events forproducts such as prescription drugs, OTC medication are submitted.


### A bit about the data:

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

### Data gathering:

[Here's the script](https://github.com/samimath/openFDA/blob/master/get_raw_data.py) to pull data from the API and returns a flattened dataframe with some additional features. API keys are provided by `openFDA.gov` after submitting a request.

### Data analysis:
[Here's a jupyter notebook](https://github.com/samimath/openFDA/blob/master/eda.ipynb) which documents some of the exploratory work I've tried for this dataset. 

