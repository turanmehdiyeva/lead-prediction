categorical_df = leads.select_dtypes(include='object')
categorical_df.drop(['prospect_id'], axis=1, inplace=True)
categorical = list(categorical_df.columns)

numerical_df = leads.select_dtypes(exclude='object')
numerical_df.drop(['lead_number', 'converted'], axis=1, inplace=True)
numerical = list(numerical_df)

train_X['converted'].value_counts()

#finding converting rate
global_mean = round(train_X['converted'].mean(),3)
print(f'Only {global_mean*100}% of leads have been successfully converted.')

#creating function to analize categorical variables
def converted_stat(column):
    data_group = train_X.groupby(column).converted.agg(['mean'])
    data_group['diff'] = data_group['mean'] - global_mean
    data_group['reliability'] = data_group['mean']/global_mean
    return data_group
  
for col in categorical:
  display(converted_stat(col))

#Dependency between categorical variables and target variable  
from sklearn.metrics import mutual_info_score

def calculate_mi(series):
    return mutual_info_score(series, leads['converted'])

data_mi = leads[categorical].apply(calculate_mi)
data_mi = data_mi.sort_values(ascending=False).to_frame(name='Mi')
data_mi
    
#Correcaltion Coefficient
leads[numerical].corrwith(leads['converted']).to_frame('Correlations')
