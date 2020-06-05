
# coding: utf-8

# ###  BIG MART SALES PREDICTION
# 
# #### Big Mart is a big supermarket chain, having stores all around the country.The management wants to predict sales per product for each store. The shop has collected sales data of products accross 10 sroes in different cities over a given period of time.
# 
# #### Problem Statement : This is supervised machine learning problem with a target lebel as item_outlet_sales, we will be implementing regression techniques to design prediction model
# 
# #### BigMart Sale Analysis divided into below Categories
# 1. Exploratory data analysis (EDA);
# 2. Data Pre-processing;
# 3. Feature engineering;
# 4. Feature Transformation;
# 5. Modeling;
# 6. Hyperparameter tuning
# 7. Compare the Accuracy of the models.

# #### Step 1: Exploratory data analysis (EDA);

# #### Import all the required Libraries

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# #### Load the Dataset to Pandas DataFrame

# In[129]:


df = pd.read_csv('C:\\Users\\Anup\\Desktop\\BigMart_DataSet\\Bigmart_Train.csv')


# #### Display Random samples of Data set

# In[75]:


# Display random samples
df.sample(5).style.set_table_styles(
    [{'selector': 'tr:hover',
      'props': [('background-color', 'grey')]}]
)


# #### Display top 5 observations

# In[76]:


# Display top 5 observations
df.head(5).style.set_properties(**{'background-color': 'black',                                                   
                                    'color': 'lawngreen', 
                                    'border-color': 'white'})


# #### Display Last 5 observations

# In[77]:


# Display Last 5 observations
df.tail(5).style.set_table_styles(
[{'selector': 'tr:nth-of-type(odd)',
  'props': [('background', '#eee')]}, 
 {'selector': 'tr:nth-of-type(even)',
  'props': [('background', 'white')]},
 {'selector': 'th',
  'props': [('background', '#606060'), 
            ('color', 'white'),
            ('font-family', 'verdana')]},
 {'selector': 'td',
  'props': [('font-family', 'verdana')]},
]
)


# #### If we look at variable Item_Identifier , we can see different group of letters per each product such as ‘FD’ (Food), ‘DR’(Drinks) and ‘NC’ (Non-Consumable).

# In[79]:


# Check the dimension of the dataset.
print("The train data set size is : {} ".format(df.shape))
print("Above DataSet contains {} rows and {} features or columns ".format(df.shape[0],df.shape[1]))


# #### Display Feature Types and No. of non-null entries for each features

# In[80]:


# Display Feature Types and No of non-null entries for each features
df.info()
print("\n*************************************** \n Item_Weight & Outlet_Size contains Null Values")


# #### Display Statistical Summary of Numeric Features

# In[81]:


df.describe().transpose()


# ### Step 2: Data PreProcessing 

# #### Check Null or Missing Values  

# In[82]:


print(df.isnull().sum())
print("\n*************************************** \n Item_Weight & Outlet_Size contains Null Values")


# #### Check missing values ratio for the features

# In[83]:


all_data_na = (df.isnull().sum() / len(df)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# #### Missing Ratio Tells for Outlet_Size feature approxmate 28% are Missing values and for Item_Weight approxmate 17% are missing values	 

# In[84]:


f, ax = plt.subplots(figsize=(10, 8))
plt.xticks(rotation='50')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# ### Univariate Analysis

# #### Features having multiple class or levels, convert them to Category Type

# In[86]:


# select all columns except float and integer based 
df.select_dtypes(exclude =['float64', 'int64']).dtypes 


# #### Check Feature Distribution

# In[87]:


df.hist(figsize=(15,12))


# #### Check total count of each class or level for all categorical features 

# In[130]:


print("Categories and Count of each class in Item Fat Content:\n*************************\n{}".format(df.Item_Fat_Content.value_counts()))


# In[131]:


## LF, low fat belong to same category that is Low Fat and reg belong to Regular category so replacing LF, low fat and reg to thier category by
df.Item_Fat_Content=df.Item_Fat_Content.replace('LF','Low Fat')
df.Item_Fat_Content=df.Item_Fat_Content.replace('reg','Regular')
df.Item_Fat_Content=df.Item_Fat_Content.replace('low fat','Low Fat')


# In[132]:


df.Item_Fat_Content.value_counts()


# In[96]:


df.Item_Type.value_counts()


# #### Display the counts of each items in Item_Type

# In[168]:


plt.figure(figsize=(15,10))
sns.countplot(df.Item_Type)
plt.xticks(rotation=50)


# In[117]:


df.Outlet_Size.value_counts()


# #### Distribution of Outlet Size

# In[169]:


plt.figure(figsize=(10,8))
sns.countplot(df.Outlet_Size)


# In[119]:


df.Outlet_Type.value_counts()


# In[170]:


plt.figure(figsize=(10,8))
sns.countplot(df.Outlet_Type)


# In[121]:


df.Outlet_Location_Type.value_counts()


# In[171]:


plt.figure(figsize=(10,8))
sns.countplot(df.Outlet_Location_Type)


# In[133]:


df.groupby('Outlet_Location_Type').count()


# In[156]:


# group by Item_Type and Outlet_Type
grouped = df.groupby(['Item_Type', 'Outlet_Type']).agg({'Item_Outlet_Sales': ['mean', 'count']})
grouped = grouped.reset_index()
grouped


# In[155]:


# group by Item_Type and Outlet_Type and disply Item_Outlet_Sales count 
grp = df.groupby(['Outlet_Size', 'Outlet_Type']).agg({'Item_Outlet_Sales': 'count'})
grp = grp.reset_index()
grp


# In[154]:


twowaytable=pd.crosstab(df['Outlet_Size'],df['Outlet_Type'])
twowaytable


# In[160]:


# group by Outlet_Size and Outlet_Location_Type and disply Item_Outlet_Sales count 
grpdata = df.groupby(['Outlet_Size', 'Outlet_Location_Type']).agg({'Item_Outlet_Sales': 'count'})
grpdata = grpdata.reset_index()
grpdata


# In[159]:


twowaytable=pd.crosstab(df['Outlet_Size'],df['Outlet_Location_Type'])
twowaytable


# #### Display distribution of numeric features

# In[162]:


for i in df.describe().columns:
    sns.distplot(df[i].dropna())
    plt.show()


# #### Display distribution of numeric features as Box Plot

# In[163]:


for i in df.describe().columns:
    sns.boxplot(df[i].dropna())
    plt.show()


# ### Bivariate Analysis

# #### Need to understand the relationship between our target variable and predictors as well as the relationship among predictors.

# In[8]:


fig,axes=plt.subplots(1,1,figsize=(10,8))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)
plt.plot([69,69],[0,5000])
plt.plot([137,137],[0,5000])
plt.plot([203,203],[0,9000])


# #### Item_Weight vs Item_Outlet_Sales Analysis

# In[9]:


plt.figure(figsize=(13,9))
plt.xlabel("Item_Weight ")
plt.ylabel(" Item_Outlet_Sales")
plt.title("Item_Weight vs Item_Outlet_Sales Analysis")
sns.scatterplot(x="Item_Weight",y="Item_Outlet_Sales",hue="Item_Type",size="Item_Weight",data=df)


# #### As the plots are random and scattered, so there is no good Co-rellation between Item_Weight vs Item_Outlet_Sales. Item Weight doesn't affect more on item sales

# #### Check Co-rellation between Item_Visibility vs Item_Outlet_Sales

# In[10]:


plt.figure(figsize=(13,9))
plt.xlabel("Item_Visibility")
plt.ylabel(" Item_Outlet_Sales")
plt.title("Item_Visibility vs Item_Outlet_Sales Analysis")
sns.scatterplot(x="Item_Visibility",y="Item_Outlet_Sales",hue="Item_Type",size="Item_Weight",data=df)


# #### From the above plot, most of the items are clustered within 0.15 visibility. As per the data, lower the visibility more the sales. This might be due to the fact that a great number of daily use products, which do not need high visibility

# #### Check the co-rellation between Item_Visibility vs Item_MRP

# In[11]:


plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Maximum Retail Price")
plt.title("Item_Visibility vs Maximum Retail Price Analysis")
plt.plot(df.Item_Visibility,df.Item_MRP,".", alpha=0.3)


# #### Above graph tells, less visible items are having Maximum Retail Price 

# #### Outlet_Establishment_Year and Item_Outlet_Sales analysis

# In[18]:


Outlet_Establishment_Year_pivot = df.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# #### There seems to be no significant meaning between the year of store establishment and the sales for the items. 1998 has low values but thet might be due to the fact the few stores opened in that year.

# #### Item_Type and Item_Visibility Analysis

# In[22]:


Outlet_Establishment_Year_pivot = df.pivot_table(index='Item_Type', values="Item_Visibility", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(20,7))
plt.xlabel("Item_Type")
plt.ylabel("Item_Visibility")
plt.title("Impact of Item_Type on Item_Visibility")
plt.xticks(rotation=50)
plt.show()


# In[23]:


Outlet_Establishment_Year_pivot = df.pivot_table(index='Item_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(20,7))
plt.xlabel("Item_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Type on Item_Outlet_Sales")
plt.xticks(rotation=50)
plt.show()


# ### Bivariate Analysis for Categorical Variables

# #### Impact of Outlet_Identifier on Item_Outlet_Sales

# In[26]:


Outlet_Identifier_pivot = df.pivot_table(index="Outlet_Identifier", values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# #### we see that thr groceries (“OUT010”, “OUT019”) have the lowest sales results which is expected followed by the Supermarket Type 2 (“OUT018”). The best results belong to “Out027” which is a “Medium” size Supermarket Type 3.

# #### Impact of Outlet_Type on Item_Outlet_Sales

# In[39]:


Outlet_Type_pivot = df.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# ### Step 3: Feature Engineering
# #### Check Co-rellation between the features

# In[19]:


#Correlation map to see how features are correlated with Item Outlet Sales
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), vmax=1, square=True, annot=True, cmap='viridis')
plt.title("Co-rellation Between Different Features")
plt.show()


# In[17]:


# Lets see correlation b/w target and features
corr_matrix=df.corr()
corr_matrix['Item_Outlet_Sales'].sort_values(ascending=False)


# #### Item_Visibility is the feature with the lowest correlation with our target variable. Therefore, the less visible the product is in the store the higher the price will be. ITEM_MRP seems to have a good correlation with targeted ITEM_OUTLET_SALES and other columns are not very useful for prediction of target value

# In[ ]:


## Above co-rrelation matrix says there is good relationship between Item_MRP and Item_Outlet_Sales
## Since ITEM_WEIGHT column correlation strength is very low so we can drop it


# ### Missing Value Treatments

# In[133]:


df.isnull().sum()


# #### Item_Weight contains 1463 no. of missing values. As Item_Weight is approxmately Normally Distributed and there is no outliers so we can use the mean of this column for missing value treatment. 

# In[134]:


df['Item_Weight'].mean()


# In[135]:


df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)


# #### For Outlet_Size the missing will be treated as mode, medium size outlets are more 

# In[136]:


df['Outlet_Size'].fillna('Medium',inplace=True)


# In[137]:


df.isnull().sum()


# #### Now there is no Null or missing values

# #### Treat Item_Visibility Feature. There are many zeros for item_Visiblity which not possible, all the items needs to be visible to the customers. 

# In[138]:


df[df['Item_Visibility']==0]['Item_Visibility'].count()


# In[139]:


df['Item_Visibility'] = df['Item_Visibility'].replace(0.000000,np.nan)#first fill by nam for simplicity
df['Item_Visibility'].fillna(df['Item_Visibility'].median(),inplace=True)


# In[140]:


df[df['Item_Visibility']==0]['Item_Visibility'].count()


# #### Determine the years of operation of a store

# #### The outlet i established from 1985 to 2009. 

# In[141]:


df.Outlet_Establishment_Year.unique()


# In[142]:


# We are subtracting from 2010 to get yeas of establishment
df['Outlet_Years'] = 2010 - df['Outlet_Establishment_Year']
df['Outlet_Years'].describe()


# #### Create a broad category of Item_Type

# #### Item_Type variable has 16 categories which might not prove to be very useful in our analysis. So it’s a good idea to combine them. If we look closely to the Item_Identifier of each item we see that each one starts with either “FD” (Food), “DR” (Drinks) or “NC” (Non-Consumables). Therefore, we can group the items within these 3 categories

# In[143]:


#Get the first two characters of ID:
df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
df['Item_Type_Combined'].value_counts()


# #### there were some non-consumables as well and a fat-content should not be specified for them. So we can also create a separate category for such kind of observations.

# In[144]:


df.Item_Fat_Content.value_counts()


# In[145]:


#Mark non-consumables as separate category in low_fat:
df.loc[df['Item_Type_Combined']=="Non-Consumable","Item_Fat_Content"] = "Non-Edible"


# In[146]:


df.Item_Fat_Content.value_counts()


# #### Creating variable Item_Visibility_Avg

# #### we hypothesized that products with higher visibility are likely to sell more. But along with comparing products on absolute terms, we should look at the visibility of the product in that particular store as compared to the mean visibility of that product across all stores. This will give some idea about how much importance was given to that product in a store as compared to other stores. We can use the ‘visibility_avg’ variable made above to achieve this.

# In[147]:


Item_Visibility_Avg = df.pivot_table(values="Item_Visibility", index="Item_Identifier")
Item_Visibility_Avg


# In[108]:


func = lambda x: x['Item_Visibility']/Item_Visibility_Avg['Item_Visibility'][Item_Visibility_Avg.index == x['Item_Identifier']][0]
df['Item_Visibility_MeanRatio'] = df.apply(func,axis=1).astype(float)
df['Item_Visibility_MeanRatio'].describe()


# In[148]:


df.head()


# ### Step 4: Feature Transformation
# #### Categorical Variables — One Hot Encoding

# #### Since scikit-learn only accepts numerical variables, we need to convert all categories of nominal variables into numeric types. Let’s start with turning all categorical variables into numerical values using LabelEncoder() (Encode labels with value between 0 and n_classes-1). After that, we can use get_dummies to generate dummy variables from these numerical categorical variables

# In[150]:


#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']


# In[152]:


for i in var_mod:
    df[i] = le.fit_transform(df[i])


# #### One-Hot-Coding refers to creating dummy variables, one for each category of a categorical variable. For example, the Item_Fat_Content has 3 categories — LowFat,Regular,Non-Edible. One hot coding will remove this variable and generate 3 new variables. Each will have binary numbers — 0 (if the category is not present) and 1(if category is present). This can be done using get_dummies function of Pandas.

# In[153]:


#Dummy Variables:
df = pd.get_dummies(df, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
df.dtypes


# In[155]:


df.head()


# #### Here we can see that all variables are now float and each category has a new variable. Lets look at the 3 columns formed from Item_Fat_Content.

# In[156]:


df[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


# #### You can notice that each row will have only one of the columns as 1 corresponding to the category in the original variable.

# In[157]:


#Drop the columns which have been converted to different types:
df.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)


# ### Split Data into Training and Testing set

# In[163]:


X = df.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier', 'Item_Weight'],axis=1)


# In[165]:


Y = df['Item_Outlet_Sales']


# #### Splitting the dataset into 80 and 20 ratio. *0% data will be used for training the model and remaining 20% of the data will be used for evaluating the model

# In[172]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[272]:


print("Shape of training Data is {}".format(X_train.shape))


# In[273]:


print("Shape of test Data is {}".format(X_test.shape))


# ### Step 5: Modeling
# ### Linear Regression Model

# In[183]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
lr.fit(X_train, Y_train)


# In[185]:


lr_pred = lr.predict(X_test)


# #### Check the difference between the actual value and predicted value.

# In[189]:


finaldf = pd.DataFrame({'Actual': Y_test, 'Predicted': lr_pred})
df1 = finaldf.head(25)
finaldf.head(20)


# #### Now let's plot the comparison of Actual and Predicted values

# In[190]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# #### The final step is to evaluate the performance of the algorithm. We’ll do this by finding the values for MAE, MSE, and RMSE. Execute the following script:

# In[281]:


from sklearn import metrics
lr_rmse = np.sqrt(metrics.mean_squared_error(Y_test, lr_pred))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(Y_test, lr_pred),2))  
print('Mean Squared Error:', round(metrics.mean_squared_error(Y_test, lr_pred),2))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(Y_test, lr_pred)),2))


# In[284]:


lr_accuracy = lr.score(X_train,Y_train)
print("Linear Model Accuracy is ", round(lr_accuracy*100,2),'%')


# In[286]:


# R squared value
print("R squared value is  " , metrics.explained_variance_score(Y_test, lr_pred))


# In[255]:


# calculating coefficients
coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(lr.coef_)
coeff


# In[261]:


#checking the magnitude of coefficients
predictors = X_train.columns
coef = pd.Series(lr.coef_,predictors).sort_values()
plt.figure(figsize=(15,15))
coef.plot(kind='bar', title='Modal Coefficients')


# ### Ridge Regression

# In[287]:


from sklearn.linear_model import Ridge
## training the model
ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(X_train,Y_train)
rr_pred = ridgeReg.predict(X_test)
rr_rmse = np.sqrt(metrics.mean_squared_error(Y_test, rr_pred))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(Y_test, rr_pred),2))  
print('Mean Squared Error:', round(metrics.mean_squared_error(Y_test, rr_pred),2))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(Y_test, rr_pred)),2))


# In[290]:


## calculating score 
rr_accuracy = ridgeReg.score(X_train,Y_train)
print("Ridge Regression Model Accuracy is ", round(rr_accuracy*100,2),'%')


# ### Lasso regression

# #### LASSO (Least Absolute Shrinkage Selector Operator), is quite similar to ridge, but lets understand the difference them by implementing it in our big mart problem.

# In[291]:


from sklearn.linear_model import Lasso
lassoReg = Lasso(alpha=0.3, normalize=True)
lassoReg.fit(X_train,Y_train)
la_pred = lassoReg.predict(X_test)

la_rmse = np.sqrt(metrics.mean_squared_error(Y_test, rr_pred))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(Y_test, la_pred),2))  
print('Mean Squared Error:', round(metrics.mean_squared_error(Y_test, la_pred),2))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(Y_test, la_pred)),2))


# In[292]:


## calculating score 
la_accuracy = lassoReg.score(X_test,Y_test)
print("Lasso Regression Model Accuracy is ", round(la_accuracy*100,2),'%')


# #### So, we can see that there is a slight improvement in our model because the value of the R-Square has been increased. Note that value of alpha, which is hyperparameter of Ridge, which means that they are not automatically learned by the model instead they have to be set manually.

# ### RandomForestRegressor

# In[307]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=400,max_depth=6,min_samples_leaf=100,n_jobs=4)
rf.fit(X_train,Y_train)
rf_pred =rf.predict(X_test)


# In[308]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rf_rmse = np.sqrt(metrics.mean_squared_error(Y_test, rf_pred))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(Y_test, rf_pred),2))  
print('Mean Squared Error:', round(metrics.mean_squared_error(Y_test, rf_pred),2))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(Y_test, rf_pred)),2))


# In[309]:


rf_accuracy = rf.score(X_train,Y_train)
print("Random Forest model accuracy is  ", round(rf_accuracy*100,2),'%')


# ### DecisionTreeRegression

# In[298]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=6,min_samples_leaf=100)
dt.fit(X_train,Y_train)
dt_pred = dt.predict(X_test)


# In[299]:


dt_rmse = np.sqrt(metrics.mean_squared_error(Y_test, dt_pred))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(Y_test, dt_pred),2))  
print('Mean Squared Error:', round(metrics.mean_squared_error(Y_test, dt_pred),2))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(Y_test, dt_pred)),2))


# In[300]:


dt_accuracy = dt.score(X_test,Y_test)
print("Decision Tree Model Accuracy is " , dt_accuracy*100,'%')


# ### XGBoost Regressior

# In[230]:


from xgboost import XGBRegressor


# In[231]:


XGB = XGBRegressor(n_estimators=1000,learning_rate=0.05)
XGB.fit(X_train,Y_train)


# In[232]:


XGB_pred = XGB.predict(X_test)


# In[310]:


xgb_rmse = np.sqrt(metrics.mean_squared_error(Y_test, XGB_pred))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(Y_test, XGB_pred),2))  
print('Mean Squared Error:', round(metrics.mean_squared_error(Y_test, XGB_pred),2))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(Y_test, XGB_pred)),2))


# In[311]:


XGB_accuracy = XGB.score(X_train,Y_train)
print("XGBoost Model Accuracy is ",round(XGB_accuracy*100,2),'%')


# ### Step 7: Check Model Accuracy

# In[317]:


rmse = {'Linear Regression': round(lr_rmse,2),
        'Ridge Regression' : round(rr_rmse,2),
        'Lasso Regression' : round(la_rmse,2),
        'Decision Tree' : round(dt_rmse,2),
        'Random Forest' : round(rf_rmse,2),
        'XGBoost Model' : round(xgb_rmse,2)}


# In[323]:


from pandas import DataFrame

df_rmse = DataFrame(list(rmse.items()),columns = ['Models','RMSE'])

print (df_rmse)


# #### From the Above models Random Forest having less RMSE(Root Mean Square Error) value

# In[324]:


rfdf = pd.DataFrame({'Actual': Y_test, 'Predicted': rf_pred})
df2 = rfdf.head(25)
df2.head(25)


# In[325]:


df2.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ### Conclusion :
# 
# #### Random Forest Model is comparatively performed better among all other model.
