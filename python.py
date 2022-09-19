import pandas as pd
import numpy as np
from pandas import plotting
from scipy import stats
from statsmodels.formula.api import ols
import seaborn
from matplotlib import pyplot as plt


data = pd.read_csv('csv/brain_size.csv', header=0, delim_whitespace=True)
data

###dataframe from numpy arrays
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

####data manipulation
data.shape
data.columns

########BROKE HERE -- error 'Gender' is not a column name
print(data['Gender'])

data[data['Gender'] == 'Female']['VIQ'].mean()

groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender,value.mean()))

groupby_gender.mean()

#####PLOTTING 

plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])   

plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])
####recieving error that objects arent in the columns


####Hypothesis testing


stats.ttest_1samp(data['VIQ'], 0)


female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender']== 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)


stats.ttest_ind(data['FSIQ'], data['PIQ'])   


stats.ttest_rel(data['FSIQ'], data['PIQ'])   


stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)   


stats.wilcoxon(data['FSIQ'], data['PIQ'])   



#####Linear Models 

x = np.linspace(-5, 5, 20)
np.random.seed(1)

y = -5 + 3*x + 4 * np.random.normal(size=x.shape)

data = pd.DataFrame({'x': x, 'y': y})


###OLS model

model = ols("y ~ x", data).fit()

print(model.summary())  


data = pd.read_csv('csv/brain_size.csv', sep=';', na_values=".")

###comparison

model = ols("VIQ ~ Gender + 1", data).fit()

print(model.summary())

stats.ttest_ind(data['FSIQ'], data['PIQ'])   


####multiple regression

data_2 = pd.read_csv('csv/iris.csv')

model = ols('sepal_width ~ name + petal_length', data_2).fit()

print(model.summary())

print(model.f_test([0, 1, -1, 0]))  


###seaborn pairplot

seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],kind='reg')

seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],kind='reg', hue='SEX') 


####matplotlib
plt.rcdefaults()

seaborn.lmplot(y='WAGE', x='EDUCATION', data=data)

result = data_2.ols(formula='wage ~ education + gender + education * gender',data=data).fit()
print(result.summary())    
