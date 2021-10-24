# -*- coding: utf-8 -*-
"""
Created on Sat Sep  24 19:25:06 2021
@author: Turan
"""

import pandas as pd 

leads = pd.read_csv('Leads.csv')


#handling missing values

for i in leads.columns:
    if leads[i].isna().sum()>3000:
        leads.drop(i, axis=1, inplace=True)
        
leads['Lead Source'].value_counts()
leads['Lead Source'].fillna('Google', inplace=True)
leads['Lead Source'] = leads['Lead Source'].str.replace('google', 'Google')
leads['Lead Source'].unique()

leads['Last Activity'].value_counts()
leads['Last Activity'].fillna('Other', inplace=True)

leads['City'].value_counts()
leads['City'].fillna('Select', inplace=True)

leads['Specialization'].value_counts()
leads['Specialization'].fillna('Select', inplace=True)

leads['How did you hear about X Education'].value_counts()
leads['How did you hear about X Education'].fillna('Select', inplace=True)

leads['Country'].value_counts()
leads[(leads['Country'].isna())&(leads['City'].isna()==False)]['City'].unique()
leads.loc[(leads['Country'].isna())&(leads['City']=='Mumbai'), 'Country'] = 'India'
leads.loc[(leads['Country'].isna())&(leads['City']=='Other Cities of Maharashtra'), 'Country'] = 'India'
leads['Country'].fillna('unknown', inplace=True)

leads['What is your current occupation'].value_counts()
leads['What is your current occupation'].fillna('Other', inplace=True)

leads['Lead Profile'].value_counts()
leads['Lead Profile'].fillna('Select', inplace=True)

leads['What matters most to you in choosing a course'].value_counts()
leads['What matters most to you in choosing a course'].fillna('Other', inplace=True)

leads.describe()
leads['TotalVisits'].fillna(3, inplace=True)
leads['Page Views Per Visit'].fillna(2, inplace=True)

leads.columns = leads.columns.str.lower().str.replace(' ','_')
string_columns = list(leads.dtypes[leads.dtypes=='Object'].index)

#make all column names follow the same nameing convention
for col in string_columns:
    leads[col] = leads[col].str.lower().str.replace(' ', '_')


leads.loc[(leads['Country'].isna())&(leads['City']=='Mumbai'), 'Country'] = 'India'

leads[(leads['Country'].isna())&(leads['City'].isna()==False)]['City'].unique()
leads.loc[(leads['Country'].isna())&(leads['City']=='Other Cities of Maharashtra'), 'Country'] = 'India'
leads['Country'].fillna('unknown', inplace=True)
