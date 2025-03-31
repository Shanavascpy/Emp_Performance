import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

data=pd.read_excel("INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls")

data=pd.DataFrame(data,columns=data.columns)

pd.set_option('display.max_columns',None)

data.head()

#data.info()

print(data['Gender'].unique(),"\n")
print(data['EducationBackground'].unique(),"\n")
print(data['MaritalStatus'].unique(),"\n")
print(data['EmpDepartment'].unique(),"\n")
print(data['EmpJobRole'].unique(),"\n")
print(data['BusinessTravelFrequency'].unique(),"\n")

#Encoding the data manually
"""
data['Gender'].replace({'Male':1,'Female':0},inplace=True)
data['EducationBackground'].replace({'Human Resources':1,'Other':2,'Technical Degree':3,'Marketing':4,'Medical':5,'Life Sciences':6},inplace=True)
data['MaritalStatus'].replace({'Single':1, 'Married':2, 'Divorced':0},inplace=True)
data['EmpDepartment'].replace({'Sales':6,'Development':5,'Research & Development':4,'Human Resources':3,'Finance':2,'Data Science':1},inplace=True)
data['BusinessTravelFrequency'].replace({'Non-Travel':1,'Travel_Frequently':2,'Travel_Rarely':3},inplace=True)
data['OverTime'].replace({'Yes':0,'No':1},inplace=True)
data['Attrition'].replace({'Yes':0,'No':1},inplace=True)
"""
def apply_mappings(data):
  
  if isinstance(data, np.ndarray):
    column_names=['EmpNumber', 'Age', 'Gender', 'EducationBackground', 'MaritalStatus',
    'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency',
    'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction',
    'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel',
    'EmpJobSatisfaction', 'NumCompaniesWorked', 'OverTime',
    'EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction',
    'TotalWorkExperienceInYears', 'TrainingTimesLastYear',
    'EmpWorkLifeBalance', 'ExperienceYearsAtThisCompany',
    'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'Attrition', 'PerformanceRating']
    data = pd.DataFrame(data, columns=column_names)
  Gender_mapping={'Male':1,'Female':0}
  EducationBackground_mapping={'Human Resources':1,'Other':2,'Technical Degree':3,'Marketing':4,'Medical':5,'Life Sciences':6}
  MaritalStatus_mapping={'Single':1, 'Married':2, 'Divorced':0}
  EmpDepartment_mapping={'Sales':6,'Development':5,'Research & Development':4,'Human Resources':3,'Finance':2,'Data Science':1}
  BusinessTravelFrequency_mapping={'Non-Travel':1,'Travel_Frequently':2,'Travel_Rarely':3}
  OverTime_mapping={'Yes':0,'No':1}
  Attrition_mapping={'Yes':0,'No':1}

# Function for apply the mappings
#def apply_mappings(data):
  data['Gender']=data.Gender.map(Gender_mapping)
  data['EducationBackground']=data.EducationBackground.map(EducationBackground_mapping)
  data['MaritalStatus']=data.MaritalStatus.map(MaritalStatus_mapping)
  data['EmpDepartment']=data.EmpDepartment.map(EmpDepartment_mapping)
  data['BusinessTravelFrequency']=data.BusinessTravelFrequency.map(BusinessTravelFrequency_mapping)
  data['OverTime']=data.OverTime.map(OverTime_mapping)
  data['Attrition']=data.Attrition.map(Attrition_mapping)
  return data

# Create a transformer using FunctionTransformer
#mapping_transformer = FunctionTransformer(apply_mappings)
mapping_transformer = FunctionTransformer(lambda X: apply_mappings(pd.DataFrame(X, columns=feature_names)))

def encode_and_drop_columns(data):
  if 'EmpJobRole' in data.columns:  
#Initialise label encoder
    labelencoder_EmpjobRole=LabelEncoder()
#Fit and transform data
    encoded_data=labelencoder_EmpjobRole.fit_transform(data['EmpJobRole'])
#Add encoded data to dataframe
    data['EmpJobRoleEncod']=encoded_data
  if 'EmpJobRole' in data.columns:
    data.drop(['EmpJobRole'],inplace=True,axis=1)
  #else:
    #print("Warning: Column 'EmpJobRole' not found in data")

  if 'EmpNumber' in data.columns:
#Remove EmpNumber which is not so important in prediction
    data.drop(['EmpNumber'],inplace=True,axis=1)
  #else:
    #print("Warning: Column 'EmpNumber' not found in data")
  return data

# Create transformer
label_encoding_transformer = FunctionTransformer(encode_and_drop_columns)

data.head()

"""
data_for_scaling=data[['Age','EducationBackground','EmpDepartment','DistanceFromHome','EmpEducationLevel','EmpEnvironmentSatisfaction',
                        'EmpHourlyRate','EmpJobInvolvement','EmpJobLevel','EmpJobSatisfaction','NumCompaniesWorked',
                        'EmpLastSalaryHikePercent','EmpRelationshipSatisfaction','TotalWorkExperienceInYears','TrainingTimesLastYear',
                        'EmpWorkLifeBalance','ExperienceYearsAtThisCompany','ExperienceYearsInCurrentRole','YearsSinceLastPromotion',
                        'YearsWithCurrManager','EmpJobRoleEncod']]

data_not_scaling=data.drop(data_for_scaling,axis=1)
"""

def scale_features(data):
  
    
  data_for_scaling=data[['Age','EducationBackground','EmpDepartment','DistanceFromHome','EmpEducationLevel','EmpEnvironmentSatisfaction',
                        'EmpHourlyRate','EmpJobInvolvement','EmpJobLevel','EmpJobSatisfaction','NumCompaniesWorked',
                        'EmpLastSalaryHikePercent','EmpRelationshipSatisfaction','TotalWorkExperienceInYears','TrainingTimesLastYear',
                        'EmpWorkLifeBalance','ExperienceYearsAtThisCompany','ExperienceYearsInCurrentRole','YearsSinceLastPromotion',
                        'YearsWithCurrManager','EmpJobRoleEncod']]
  data = pd.DataFrame(data, columns=data.columns)  
  data_not_scaling=data.drop(data_for_scaling,axis=1)

#Object Creation
  scaler=MinMaxScaler()

#Fit and transform data
  scaled_data=scaler.fit_transform(data_for_scaling)

  scaled_df=pd.DataFrame(scaled_data,columns=data_for_scaling.columns)
    
#Joining the scaled and non scaled data
  new_data=pd.concat([scaled_df,data_not_scaling],axis=1)
    
  final_df=pd.DataFrame(new_data,columns=new_data.columns)
  return final_df


"""
#Creating final processed data
final_df=pd.DataFrame(new_data,columns=new_data.columns)
final_df.head()
"""

#Save data for modelling into directory
#final_df.to_excel('Processed_data.xlsx',index=False)

pipeline = Pipeline(steps=[
    ('mapping_transformer', FunctionTransformer(apply_mappings, validate=False)),  # Apply mappings
    ('label_encoding_transformer', FunctionTransformer(encode_and_drop_columns, validate=False)),  # Label encode and drop columns
    ('scaling', FunctionTransformer(scale_features, validate=False))  # Scale selected features
])
pipeline.set_output(transform='pandas')
#Save data for modelling into directory
##final_df.to_excel('Processed_data.xlsx',index=False)
final_data=pipeline.fit_transform(data)

final_data.to_excel('Processed_data.xlsx',index=False)

import joblib
joblib.dump(pipeline,'processed_pipeline.pkl')