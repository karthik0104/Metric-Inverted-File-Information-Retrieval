import time
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
import collections
import heapq
from operator import itemgetter

# Define parameter variables
sample_size = 1000
data_size = 1000000

mylist = []

for chunk in  pd.read_csv('dataset/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv', sep=',', chunksize=20000):
    mylist.append(chunk)

hp_data = pd.concat(mylist, axis= 0)[1:data_size+1]
del mylist

#print(hp_data.head())
hp_data.to_csv('dataset/hp_condensed.csv')

###########################################   PRE-PROCESSING    ####################################################

df = pd.read_csv('dataset/hp_condensed.csv', delimiter=',', )

# Drop numeric ID-type columns
df = df.drop(['Operating Certificate Number', 'Facility Id', 'Facility Name', 'Zip Code - 3 digits', 'CCS Diagnosis Code',
                        'CCS Diagnosis Description', 'CCS Procedure Code', 'CCS Procedure Description',
              'APR MDC Code', 'APR MDC Description', 'APR DRG Code', 'APR DRG Description', 'APR Severity of Illness Code',
              'Payment Typology 2', 'Length of Stay',
                        'Payment Typology 3', 'Attending Provider License Number', 'Operating Provider License Number',
                        'Other Provider License Number' ], axis=1)

# Considering followinf columns
print('Considering columns for analysis: ')
print(list(df.columns))

# Fill nan values with string: Unknown
df = df.fillna('Unknown')

# Encode required columns
label_encoder = preprocessing.LabelEncoder()

df['Health Service Area']= label_encoder.fit_transform(df['Health Service Area'])
df['Hospital County']= label_encoder.fit_transform(df['Hospital County'])
df['Age Group']= label_encoder.fit_transform(df['Age Group'])
df['Gender']= label_encoder.fit_transform(df['Gender'])
df['Race']= label_encoder.fit_transform(df['Race'])
df['Ethnicity']= label_encoder.fit_transform(df['Ethnicity'])
df['Type of Admission']= label_encoder.fit_transform(df['Type of Admission'])
df['Patient Disposition']= label_encoder.fit_transform(df['Patient Disposition'])
df['Discharge Year']= label_encoder.fit_transform(df['Discharge Year'])
df['APR Severity of Illness Description']= label_encoder.fit_transform(df['APR Severity of Illness Description'])
df['APR Risk of Mortality']= label_encoder.fit_transform(df['APR Risk of Mortality'])
df['APR Medical Surgical Description']= label_encoder.fit_transform(df['APR Medical Surgical Description'])
df['Payment Typology 1']= label_encoder.fit_transform(df['Payment Typology 1'])
df['Abortion Edit Indicator']= label_encoder.fit_transform(df['Abortion Edit Indicator'])
df['Emergency Department Indicator']= label_encoder.fit_transform(df['Emergency Department Indicator'])

# Handle Total Charges and Total Costs
df['Total Charges'] = df['Total Charges'].replace('[\$,]', '', regex=True).astype(float)
df['Total Costs'] = df['Total Costs'].replace('[\$,]', '', regex=True).astype(float)

#print(df['Total Charges'].min())
#print(df['Total Charges'].max())
#print(df['Total Costs'].min())
#print(df['Total Costs'].max())

df['Total Charges'] = ((df['Total Charges']/10000).apply(np.floor)).astype(int)
df['Total Costs'] = ((df['Total Costs']/10000).apply(np.floor)).astype(int)

# Rename identifier column
df.rename(columns = {"Unnamed: 0": "ID"}, inplace=True)
df = df.drop('ID', axis=1)
print(list(df.columns))

df.to_csv('dataset/hp_intermediate.csv')

###########################################   CREATING METRIC INVERTED FILE    ####################################################
start = time.time()

# Take sample
o_sample, sample = train_test_split(df,test_size=sample_size)

# Iterate over o_sample and store distances over all sample points
dist_map_list = []
i = 0

o_sample_index = o_sample.index
sample_index = sample.index
columns = list(sample.columns)

for ind1 in o_sample_index:
    j = 0
    dist_map = {}
    for ind2 in sample_index:
        sum = 0
        for col in columns:
            sum = sum+abs(int(o_sample[col][ind1]) - int(sample[col][ind2]))
        dist_map[j] = sum
        j = j+1
    #sorted_x = sorted(dist_map.items(), key=lambda kv: kv[1])
    #sorted_dist_map = dict(collections.OrderedDict(sorted_x[:10]))
    sorted_dist_map = dict(heapq.nsmallest(10, dist_map.items(), key=itemgetter(1))) # Using this for optimization
    s = [str(i) for i in sorted_dist_map.keys()]
    res = " ".join(s)
    dist_map_list.append(res)
    i = i+1

doc_list = dist_map_list

MIF = {}
for doc_id,doc in enumerate(doc_list):
        for word_pos,word in enumerate(doc.split()):
            MIF.setdefault(word,[]).append((doc_id,word_pos))

print("MIF:")
print(MIF)

end = time.time()

print('Time taken to create MIF is:')
print(end-start)
#dist_df.to_csv('dataset/MIF.csv')

###########################################   FINDING SIMILAR OBJECTS TO QUERY OBJECT    ########################################

# Take query object from user
print('Enter your query object:')
query_obj = input()
query_cols = query_obj.split(",")

# Prepare query ordered list
j = 0
dist_map = {}
for ind in sample.index:
    sum = 0
    col_num = 0
    for col in list(sample.columns):
        sum = sum + abs(int(query_cols[col_num]) - int(sample[col][ind]))
        col_num = col_num+1
    dist_map[j] = sum
    j = j + 1

sorted_x = sorted(dist_map.items(), key=lambda kv: kv[1])
sorted_dist_map = dict(collections.OrderedDict(sorted_x[:10]))
query_ordered_list = [str(i) for i in sorted_dist_map.keys()]

print("Ordered List Of Query:")
print(query_ordered_list)

# Create Accumulator
accumulator = {}

i=0
for query_reference in query_ordered_list:
    for doc_pos in MIF[query_reference]:
        accumulator[doc_pos[0]] = abs(int(doc_pos[1]) - i)
    i = i+1

print('Accumulator:')
print(accumulator)

sorted_x = sorted(dist_map.items(), key=lambda kv: kv[1])
sorted_dist_map = dict(collections.OrderedDict(sorted_x[:10]))
similar_docs = [str(i) for i in sorted_dist_map.keys()]

print('Similar Objects:')
print(similar_docs)