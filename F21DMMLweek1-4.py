#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
assert sys.version_info >= (3, 5)


import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os
import tarfile
import urllib
import pandas as pd
import urllib.request

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


get_ipython().system('pip install opendatasets')
import opendatasets as od


# In[3]:


od.download("https://www.kaggle.com/datasets/yasserh/loan-default-dataset")


# In[4]:


import os
import pandas as pd
my_download= os.path.join("loan-default-dataset", "Loan_Default.csv")
my_dir = os.getcwd()

loans = pd.read_csv(os.path.join(my_dir,my_download))


# In[5]:


loans.info()


# In[6]:


loans.head() # used to display the table


# In[7]:


correlations = loans.corr()  # how each column are releated to each other and corr() ignore non numeric column
correlations


# In[8]:


correlations['Credit_Score'].sort_values(ascending=False) #sorting the Credit_score column in descending order ---high to low


# In[9]:


correlations['property_value'].sort_values(ascending=False)


# In[10]:


loans[loans.duplicated(keep=False)] #If False, it consider all of the same values as duplicates. 


# In[11]:


loans.isna().sum()


# In[12]:


NAN_loans = loans[loans.isnull().any(axis=1)]
NAN_loans[['rate_of_interest','Status']]


# In[13]:


loans['Status'].mean()


# In[14]:


#cutting down the data set to just 1000 rows for quiker plotting
sample_loans = loans.sample(n=1000)


# In[15]:


import seaborn as sns


# In[16]:


sns.boxplot(data=sample_loans,x='Gender' ,y='income', hue='Credit_Worthiness')


# In[17]:


sns.scatterplot(data=sample_loans,x='Gender',y='income',hue='Status')


# In[18]:


sns.pairplot(sample_loans)


# In[19]:


sns.scatterplot(data=sample_loans,x='Gender',y='income',hue='Status')


# In[20]:


loans['Status'].max()


# In[21]:


from pandas.plotting import scatter_matrix

attributes = ["property_value", "rate_of_interest", "income", "term"]
scatter_matrix(loans[attributes], figsize=(12, 8))


# In[22]:


sample_loans['Neg_ammortization'].value_counts()


# In[23]:


fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(14,10))
sns.boxplot(data=sample_loans,x='Gender' ,y='income', hue='Neg_ammortization', ax=axes[0,0])
sns.boxplot(data=sample_loans,x='Gender' ,y='income', hue='Credit_Worthiness', ax=axes[0,1])
sns.boxplot(data=loans,x='Gender' ,y='income', hue='Neg_ammortization', ax=axes[1,0])
sns.boxplot(data=loans,x='Gender' ,y='income', hue='Credit_Worthiness', ax=axes[1,1])


# In[24]:


clean_loans =  loans.drop(['year'],axis=1)
clean_loans['loan_limit'] = clean_loans[['loan_limit']].fillna('cf')
clean_loans['approv_in_adv'] = clean_loans[['approv_in_adv']].fillna('nopre')
clean_loans = clean_loans.dropna(axis=0, subset=['loan_purpose','term','age','submission_of_application','Neg_ammortization'])
clean_loans['income'] = clean_loans['income'].fillna(clean_loans['income'].median())
clean_loans = clean_loans = clean_loans[clean_loans['property_value'] != 8000].reset_index()
clean_loans = clean_loans[clean_loans['property_value'] <= 5000000].reset_index()

########
conditions = [clean_loans['dtir1'].isnull() & clean_loans['Status'].eq(1),
              clean_loans['dtir1'].isnull() & clean_loans['Status'].eq(0),
              clean_loans['dtir1'].isnull() == False]
choices = [clean_loans['dtir1'][clean_loans['Status']==1].median(),clean_loans['dtir1'][clean_loans['Status']==0].median(), clean_loans['dtir1']]
clean_loans['dtir1'] = np.select(conditions, choices, default=0)

########
conditions = [clean_loans['property_value'].isnull() & clean_loans['Status'].eq(1),
              clean_loans['property_value'].isnull() & clean_loans['Status'].eq(0),
              clean_loans['property_value'].isnull() == False]
choices = [clean_loans['property_value'][clean_loans['Status']==1].median(),clean_loans['property_value'][clean_loans['Status']==0].median(), clean_loans['property_value']]
clean_loans['property_value'] = np.select(conditions, choices, default=0)

########
conditions = [clean_loans['LTV'].isnull() & clean_loans['Status'].eq(1),
              clean_loans['LTV'].isnull() & clean_loans['Status'].eq(0),
              clean_loans['LTV'].isnull() == False]
choices = [clean_loans['LTV'][clean_loans['Status']==1].median(),clean_loans['LTV'][clean_loans['Status']==0].median(), clean_loans['LTV']]
clean_loans['LTV'] = np.select(conditions, choices, default=0)

########
conditions = [clean_loans['Upfront_charges'].isnull() & clean_loans['Status'].eq(1),
              clean_loans['Upfront_charges'].isnull() & clean_loans['Status'].eq(0),
              clean_loans['Upfront_charges'].isnull() == False]
choices = [clean_loans['Upfront_charges'][clean_loans['Status']==0].median(),
           clean_loans['Upfront_charges'][clean_loans['Status']==0].median(), 
           clean_loans['Upfront_charges']]
clean_loans['Upfront_charges'] = np.select(conditions, choices, default=0)

########
conditions = [clean_loans['rate_of_interest'].isnull() & clean_loans['Status'].eq(1),
              clean_loans['rate_of_interest'].isnull() == False]
choices = [clean_loans['rate_of_interest'][clean_loans['Status']==0].median(), clean_loans['rate_of_interest']]
clean_loans['rate_of_interest'] = np.select(conditions, choices)

########           
conditions = [clean_loans['Interest_rate_spread'].isnull() & clean_loans['Status'].eq(1),
              clean_loans['Interest_rate_spread'].isnull() == False]
choices = [clean_loans['Interest_rate_spread'][clean_loans['Status']==0].median(),clean_loans['Interest_rate_spread']]
clean_loans['Interest_rate_spread'] = np.select(conditions, choices, default=0)


# In[25]:


fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(14,10))
sns.histplot(data=sample_loans, x="rate_of_interest", hue='Status',binwidth=0.2, ax=axes[0,0])
sns.histplot(data=sample_loans, x="income", hue='Status',binwidth=1000, ax=axes[0,1])
sns.histplot(data=sample_loans, x="dtir1", hue='Status',binwidth=10, ax=axes[1,0])
sns.histplot(data=sample_loans, x="LTV", hue='Status',binwidth=10, ax=axes[1,1])
sns.histplot(data=sample_loans, x="Upfront_charges", hue='Status',binwidth=1000, ax=axes[2,0])
sns.histplot(data=sample_loans, x="Neg_ammortization", hue='Status',ax=axes[2,1])


# In[26]:


empty_data = pd.isna(loans['loan_limit'])
loans[empty_data].isna().sum()


# In[27]:



loans[empty_data].head()


# In[28]:


loans['loan_limit'].value_counts()


# In[29]:


pip install association-metrics


# In[30]:


# Import association_metrics  
import association_metrics as am
# Convert you str columns to Category columns
cram_mat_loans = loans.apply(
        lambda x: x.astype("category") if x.dtype == "O" else x)

# Initialize a CamresV object using you pandas.DataFrame
cramersv = am.CramersV(cram_mat_loans) 
# will return a pairwise matrix filled with Cramer's V, where columns and index are 
# the categorical variables of the passed pandas.DataFrame
my_cramer_matrix = cramersv.fit().sort_values(by=['loan_limit'], ascending=False)
my_cramer_matrix


# In[31]:


fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
sns.heatmap(data=my_cramer_matrix,vmin=0., vmax=1, square=True, ax=axes[0], annot=False)
sns.heatmap(data=sample_loans.corr(),vmin=0, vmax=1, square=True, ax=axes[1], annot=False)


# In[32]:


# The year collmn is dropped as it is clear from the pair plots that it is always the same value
clean_loans =  loans.drop(['year'],axis=1)


# In[33]:


# The strategy used for loan_limit nan values is is replace by most freuent value without any additional weighting
clean_loans['loan_limit'] = clean_loans[['loan_limit']].fillna('cf')


# In[34]:


clean_loans.isna().sum()


# In[35]:


sns.heatmap(data=clean_loans[pd.isna(clean_loans['approv_in_adv'])].corr(),vmin=-0.2, vmax=1, square=True)
clean_loans[pd.isna(clean_loans['approv_in_adv'])].corr()


# In[36]:


clean_loans['approv_in_adv'].value_counts()


# In[37]:


# we will replace the nan values with the most common value for approv_in_advance
clean_loans['approv_in_adv'] = clean_loans[['approv_in_adv']].fillna('nopre')


# In[38]:


clean_loans['loan_purpose'].value_counts()


# In[39]:


clean_loans['loan_purpose'].isna().sum()


# Since this is only a small amount of data we will chose to drop a few rows which have a low number of NaN values and we use a len function to show the limited impact on the overall row count.
# 

# In[40]:


len(clean_loans)


# In[41]:


clean_loans = clean_loans.dropna(axis=0, subset=['loan_purpose'])
len(clean_loans)


# In[42]:


#we will do the same for term, Neg_ammortization, age and submission_of_application
clean_loans = clean_loans.dropna(axis=0, subset=['term','age','submission_of_application','Neg_ammortization'])
len(clean_loans)


# In[43]:


sns.heatmap(data=clean_loans[pd.isna(clean_loans['rate_of_interest'])].corr(),vmin=-0.2, vmax=1, square=True, ax=axes[0])
clean_loans[pd.isna(clean_loans['rate_of_interest'])].corr()


# In[44]:


clean_loans['dtir1'] .mean()


# In[45]:


fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(14,5))

sns.histplot(data=clean_loans, x="dtir1", hue='Status', bins=10, ax=axes[0])
sns.histplot(data=clean_loans, x="rate_of_interest", hue='Status', bins=10, ax=axes[1])


# In[46]:


clean_loans.isna().sum()


# In[47]:


sns.displot(data=clean_loans, x="income", binwidth=10000)


# In[48]:


clean_loans['income'].describe()


# In[49]:


sns.boxplot(data=clean_loans, y="income", x='Status')
plt.ylim(0, 30000)


# In[50]:


clean_loans['income'] = clean_loans['income'].fillna(clean_loans['income'].median())


# In[51]:


sns.boxplot(data=clean_loans, y="income", x='Status')
plt.ylim(0, 30000)
#Note there is no clear change in the boxplots by doing this replacement.


# In[52]:


clean_loans.isna().sum()


# In[53]:


sns.boxplot(data=clean_loans, y="dtir1", x='Status')


# In[54]:


print(clean_loans['dtir1'][clean_loans['Status']==1].median())
print(clean_loans['dtir1'][clean_loans['Status']==0].median())


# In[55]:


#this cell is replacing the Nan values for the average mean dtir1 based on status.

conditions = [
   clean_loans['dtir1'].isnull() & clean_loans['Status'].eq(1),
    clean_loans['dtir1'].isnull() & clean_loans['Status'].eq(0),
    clean_loans['dtir1'].isnull() == False
]

choices = [clean_loans['dtir1'][clean_loans['Status']==1].median(),clean_loans['dtir1'][clean_loans['Status']==0].median(), clean_loans['dtir1']]

clean_loans['dtir1'] = np.select(conditions, choices, default=0)


# In[56]:


sns.boxplot(data=clean_loans, y="LTV", x='Status')
plt.ylim(0, 200)
clean_loans['LTV'].describe()


# In[57]:


sns.boxplot(data=clean_loans, y="property_value", x='Status') 
plt.ylim(100000, 1000000)
clean_loans['property_value'].describe()


# In[58]:


fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
sns.histplot(data=clean_loans, x="property_value", hue='Status',binwidth=50000, ax=axes[0]) 
sns.histplot(data=clean_loans, x="LTV", hue='Status',binwidth=20, ax=axes[1]) 


# In[ ]:




