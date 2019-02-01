N# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:07:14 2019

@author: 64414
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import hashlib

#1. Get the Data

currPath=os.getcwd()

def getData(fileName,dataPath=None):
    datasetPath=os.path.join(currPath,dataPath)
    dataset=pd.read_csv(os.path.join(datasetPath,fileName))
    return dataset

housing=getData('housing.csv',dataPath="dataset/housing")

#2. Take a look at the data structure
housing.info()

housing['ocean_proximity'].value_counts()

housingDescr=housing.describe()

#3. Plot the histogram of the attributes
housing.hist(bins=50,figsize=(20,15))
plt.show()


#4. Split the training and testing dataset
def split_train_test(dataset,test_ratio):
    np.random.seed(42)
    shuffledIdx=np.random.permutation(len(dataset))
    test_size=int(test_ratio*len(dataset))
    testIdx=shuffledIdx[:test_size]
    trainIdx=shuffledIdx[test_size:]
    return dataset.loc[trainIdx],dataset.loc[testIdx]


# 5. Split the dataset based on the data identifiers
def test_set_identifier(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1]<test_ratio*256

def split_train_test_by_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids=data[id_column]
    in_test_set=ids.apply(lambda id_:test_set_identifier(id_,test_ratio,hash))
    return data.loc[~in_test_set],data.loc[in_test_set]

# 6. hist of Median_income
housing['median_income'].hist()
plt.show()

# 7. in order to let the testing data have the information about important data
# like median_income, we should change the continuously numerical values into 
# the categorical values
housing['income_cat']=np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat']<5.0,5.0,inplace=True)

housing['income_cat'].hist()
plt.imshow()

# 8. Spliting the dataset based on the housing['income_cat']

from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,
                             test_size=0.2,
                             random_state=42)

for trainIdx,testIdx in split.split(housing,housing['income_cat']):
    strat_train_set=housing.loc[trainIdx]
    strat_test_set=housing.loc[testIdx]
    

strat_train_set['income_cat'].value_counts()/len(strat_train_set)

housing['income_cat'].value_counts()/len(housing)

# 9. delete the income_cat to recover the original dataset
for set in (strat_train_set,strat_test_set):
    set.drop(['income_cat'],axis=1,inplace=True)
    #set.set_index('index',inplace=True)


# 10. Discover and Visualize the data to gain insights
housing=strat_train_set.copy()
housing.plot(kind='scatter',x='longitude',y='latitude')
plt.show()

# another way to show data density
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=.1)
plt.show()

# to show the relationships between the x,y, population and housing_value 
# with color map
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,
             s=housing['population']/100,label='population',
             c=housing['median_house_value'],cmap=plt.get_cmap('jet'),
             figsize=(10,7),colorbar=True,sharex=False)
plt.legend()
plt.show()

# to use an existing image as background
import matplotlib.image as mpimg
california_img=mpimg.imread('images/end_to_end project/california.png')
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,
             s=housing['population']/100,label='population',
             c=housing['median_house_value'],cmap=plt.get_cmap('jet'),
             figsize=(10,7),sharex=False,colorbar=False)
plt.imshow(california_img,extent=[-124.55,-113.80,32.45,42.05])

prices=housing['median_house_value']
tick_values=np.linspace(prices.min(),prices.max(),11)

cbar=plt.colorbar()
cbar.ax.set_yticklabels(["$%d K"%(round(v/1000)) for v in tick_values])
cbar.set_label('Median House Value')
plt.show()

# 11. explore the correlations between the attributes
corr_matrix=housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# another way to show the correlationships between the attributes
from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()

# check the relationships betweeen the median income and median house values
housing.plot(kind='scatter',x='median_income',y='median_house_value',figsize=(12,8))
plt.show()

# 12. Feature Engineering
housing['rooms_per_household']=housing['total_rooms']/housing['households']
housing['population_per_household']=housing['population']/housing['households']
housing['bedrooms_per_room']=housing['total_bedrooms']/housing['total_rooms']

# check the correlationships between the attributes
corr_matrix=housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

housing.plot(kind='scatter',x='rooms_per_household',y='median_house_value')
plt.axis([0,5,0,520000])
plt.show()

# Preparing data for mechine learning
housing=strat_train_set.drop(['median_house_value'],axis=1)
housing_labels=strat_train_set['median_house_value'].copy()

# get the incomplete rows in the dataframe
sample_incomplete_rows=strat_train_set[strat_train_set.isnull().any(axis=1)]

# Data Cleaning
housing_num=housing.drop(['ocean_proximity'],axis=1)

from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='median')
imputer.fit(housing_num)

X=imputer.transform(housing_num)

housing_tr=pd.DataFrame(data=X,columns=housing_num.columns,index=list(housing.index.values))

housing_tr=pd.DataFrame(data=X,columns=housing_num.columns)

housing_cat=housing[['ocean_proximity']]

# Category encoder
from sklearn.preprocessing import OneHotEncoder
cat_Encoder=OneHotEncoder(sparse=False)
cat_Encoder.fit(housing_cat)
housing_cat_encoded=cat_Encoder.transform(housing_cat)

# define a class that manuplate the features of class
from sklearn.base import BaseEstimator,TransformerMixin

rooms_ix,bedrooms_ix,population_ix,households_ix=3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        #cols=housing.columns.values.tolist()
        rooms_per_household=X[:,rooms_ix]/X[:,households_ix]
        population_per_household=X[:,population_ix]/X[:,households_ix]
        
        if self.add_bedrooms_per_room:
            #addedAttribs=['rooms_per_household','population_per_household','bedrooms_per_room']
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            X=np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            #addedAttribs=['rooms_per_household','population_per_household']
            X=np.c_[X,rooms_per_household,population_per_household]
        return X

attribAdder=CombinedAttributesAdder()
housing_extra_attribs=attribAdder.transform(housing_num.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline=Pipeline([('imputer',Imputer(strategy='median')),
                       ('attribAdder',CombinedAttributesAdder()),
                       ('stdScaler',StandardScaler()),
                       ])
housing_num_tr=num_pipeline.fit_transform(housing_num.values)

from sklearn.compose import ColumnTransformer

num_attribs=list(housing_num)
cat_attribs=['ocean_proximity']

full_pipeline=ColumnTransformer([('num',num_pipeline,num_attribs),
                                 ('cat',OneHotEncoder(),cat_attribs)])

housing_prepared=full_pipeline.fit_transform(housing)

# DataFrame Selector
from sklearn.base import BaseEstimator,TransformerMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribNames):
        self.attribNames=attribNames
    def fit(self, X,y=None):
        return self
    def transform(self,X):
        return X[self.attribNames].values
    

# another pipeline with feature selector
num_pipeline_with_selector= Pipeline([('featureSelector',DataFrameSelector(num_attribs)),
                                      ('imputer',Imputer(strategy='median')),
                                      ('attribsAdder',CombinedAttributesAdder()),
                                      ('stdScaler',StandardScaler())])
housing_num_tr1=num_pipeline_with_selector.fit_transform(housing)

cat_pipeline_with_selector=Pipeline([('featureSelector',DataFrameSelector(cat_attribs)),
                                     ('oneHotEncoder',OneHotEncoder(sparse=False)),])
housing_cat_tr1=cat_pipeline_with_selector.fit_transform(housing)

# combinePipeLine

from sklearn.pipeline import FeatureUnion
wholePipeline=FeatureUnion(transformer_list=[('num_pipeline',num_pipeline_with_selector),
                                             ('cat_pipeline',cat_pipeline_with_selector)])    

wholeHousingPrepared=wholePipeline.fit_transform(housing)

# How to check whether the two arrays are the same
np.allclose(housing_prepared,wholeHousingPrepared)


# Train a model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

lin_reg=LogisticRegression()
lin_reg.fit(wholeHousingPrepared,housing_labels)

some_data=housing.iloc[:20]
some_data=pd.concat([some_data,housing[housing.index==8318]])
some_labels=housing_labels.iloc[:20]
some_labels=pd.concat([some_labels,housing_labels[housing.index==8313]])
some_data_prepared=wholePipeline.fit_transform(some_data)

some_data_labels_pred=lin_reg.predict(some_data_prepared)

# Evaluate 
rmse=np.sqrt(mean_squared_error(some_labels,some_data_labels_pred))
mae=mean_absolute_error(some_labels,some_data_labels_pred)

dt_reg=DecisionTreeRegressor()
dt_reg.fit(wholeHousingPrepared,housing_labels)

some_data_pred_dt=dt_reg.predict(some_data_prepared)
rmseDt=np.sqrt(mean_squared_error(some_data_pred_dt,some_labels))
maeDt=mean_absolute_error(some_data_pred_dt,some_labels)


scores=cross_val_score(dt_reg,wholeHousingPrepared,housing_labels,cv=10,scoring='neg_mean_squared_error')
rmse_scores=np.sqrt(-scores)

# Scores displaying method
def display_scores(scores):
    print('Scores:',scores)
    print('Mean Scores:',np.mean(scores))
    print('Standard Deviation Scores:',np.std(scores))
    

from sklearn.svm import SVR

svm_reg=SVR(kernel='linear')
svm_reg.fit(wholeHousingPrepared,housing_labels)

housing_pred_svm=svm_reg.predict(some_data_prepared)

scores=cross_val_score(svm_reg,wholeHousingPrepared,housing_labels,cv=10)

display_scores(scores)

forest_reg=RandomForestRegressor(random_state=42)
forest_reg.fit(wholeHousingPrepared,housing_labels)
forest_reg_pred=forest_reg.predict(some_data_prepared)
scores=cross_val_score(forest_reg,wholeHousingPrepared,housing_labels,cv=10)
display_scores(scores)

# Fine Tune the model
from sklearn.model_selection import GridSearchCV
param_grid=[{'n_estimators':[3,10,30],
             'max_features':[2,4,6,8]},
            {'bootstrap':[False],
             'n_estimators':[3,10],
             'max_features':[2,3,4]},]

grid_search=GridSearchCV(forest_reg,param_grid,cv=5,
                         scoring='neg_mean_squared_error',
                         return_train_score=True)

grid_search.fit(wholeHousingPrepared,housing_labels)

cvres=grid_search.cv_results_

for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
    
    
feature_importances=grid_search.best_estimator_.feature_importances_

# get important features as well as the corrsponding column names

extra_attribs=['rooms_per_household','population_per_household','bedrooms_per_room']
num_attribs=list(housing_num)

catEncoder=cat_pipeline_with_selector.named_steps['oneHotEncoder']
cat_attribs=list(catEncoder.categories_[0])
cat_attribs[0]='1H OCEAN'

attributes=num_attribs+extra_attribs+cat_attribs
# =============================================================================
# idx=attributes.index('<1H OCEAN')
# 
# attributes[idx]='1H OCEAN'
# =============================================================================

sorted(zip(feature_importances,attributes),reverse=True)



final_model=grid_search.best_estimator_

X_test=strat_test_set.drop(['median_house_value'],axis=1)
y_test=strat_test_set['median_house_value'].copy()


X_test_prepared=wholePipeline.fit_transform(X_test)

final_predictions=final_model.predict(X_test_prepared)

final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)


# A full pipeline with both preparation and prediction
full_pipeline_with_prediction=Pipeline([('preparation',wholePipeline),
                                        ('linearPredictor',LogisticRegression())])

full_pipeline_with_prediction.fit(housing,housing_labels)

y_pred_with_full_pipeline_with_prediction=full_pipeline_with_prediction.predict(some_data)

rmse_full_pipeline=np.sqrt(mean_squared_error(some_labels,y_pred_with_full_pipeline_with_prediction))

#Exercise
grid_params=[{'kernel':['linear'],'C':[10.,30.,100.,300.,1000.,3000.,10000.,30000.]},
             {'kernel':['rbf'],
              'C':[1.0,3.0,10.,30.,100.,300.,1000.,3000.],
              'gamma':[0.01,0.03,0.1,0.3,1.0,3.0,10.,30.]}]
    
svm_reg=SVR()

gridSearch=GridSearchCV(svm_reg,grid_params,scoring='neg_mean_squared_error',
                        cv=5,verbose=2,n_jobs=4)
gridSearch.fit(wholeHousingPrepared,housing_labels)

negative_best_score=gridSearch.best_score_
best_rmse=np.sqrt(-negative_best_score)
best_params=gridSearch.best_params_

# 2. RandomizedSearch
from scipy.stats import expon,reciprocal
from sklearn.model_selection import RandomizedSearchCV
rnd_params={'kernel':['linear','rbf'],
            'C':reciprocal(20,20000),
            'gamma':expon(scale=1.0)}

svr_regRnd=SVR()
rndSearch=RandomizedSearchCV(svr_regRnd,param_distributions=rnd_params,
                            cv=5,scoring='neg_mean_squared_error',
                            verbose=2,n_jobs=4,random_state=42)
rndSearch.fit(wholeHousingPrepared,housing_labels)

neg_rndSearch_best_score=rndSearch.best_score_
r_neg_rndSearch_best_score=np.sqrt(-neg_rndSearch_best_score)
best_model=rndSearch.best_estimator_
y_some_data_pred=best_model.predict(some_data_prepared)
rnd_best_params=rndSearch.best_params_

# =============================================================================
# The reciprocal distribution is useful when you have no idea about what scale of the hyperparameter
# should be,whereas the exponential distribution is best when you know the what the scale of the 
# hyperparameter should be.
# =============================================================================

#the distribution plot of expon
expon_dis=expon(scale=1.0)
samples=expon_dis.rvs(10000,random_state=42)
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('Exponential distribution (scale=1.0)')
plt.hist(samples,bins=50)
plt.subplot(122)
plt.title('Log of this distribution')
plt.hist(np.log(samples),bins=50)
plt.show()

#the distribution plot of reciprocal
reciprocal_distrib=reciprocal(20,200000)
samples=reciprocal_distrib.rvs(10000,random_state=42)
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('Reciprocal distribution (scale=1.0)')
plt.hist(samples,bins=50)
plt.subplot(122)
plt.title('Log of this distribution')
plt.hist(np.log(samples),bins=50)
plt.show()


# 3. adding a transformer to select the most important features in the dataset
from sklearn.base import BaseEstimator,TransformerMixin

def indices_of_top_k(arr,k):
    return np.sort(np.argpartition(np.array(arr),-k)[-k:])
class TopKFeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self,feature_importances,k):
        self.feature_importances=feature_importances
        self.k=k
    def fit(self,X,y=None):
        self.feature_indices_=indices_of_top_k(self.feature_importances,self.k)
        return self
    def transform(self,X,y=None):
        return X[:,self.feature_indices_]



k=5
top_k_features_indices=indices_of_top_k(feature_importances,k)


preparation_and_feature_selection_pipeline=Pipeline([('preparation',wholePipeline),
                                                     ('featureSelection',TopKFeatureSelector(feature_importances,k))])
    
prepared_housing_with_top_k_features=preparation_and_feature_selection_pipeline.fit_transform(housing)



prepared_housing_with_top_k_features[0:3]

housing_prepared[0:3,top_k_features_indices]

np.allclose(prepared_housing_with_top_k_features[0:3],housing_prepared[0:3,top_k_features_indices])

# Trying to build a pipeline with datapreparation, data important feature selection as well as the final prediction
prepare_select_and_predict_pipeline=Pipeline([('preparison',wholePipeline),
                                              ('featureSelection',TopKFeatureSelector(feature_importances,k)),
                                              ('svm_reg',SVR(**rndSearch.best_params_))])


prepared_final_clf=prepare_select_and_predict_pipeline.fit(housing,housing_labels)

prepared_final_clf_pred=prepared_final_clf.predict(some_data)

print("Predictions:\t",prepared_final_clf_pred)
print("Labels:\t\t",list(some_labels))

# trying to automatic some steps in using GridSearchCV
grid_params=[{'preparation__num__imputer__strategy':['mean','median','most_frequent'],
             'feature_selection__k':list(range(1,len(feature_importances)+1))}]

grid_search_prep=GridSearchCV(prepare_select_and_predict_pipeline,grid_params,cv=5,
                              scoring='neg_mean_squared_error',verbose=2,
                              n_jobs=4)
grid_search_prep.fit(housing,housing_labels)
