import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix

data = pd.read_csv('brain_size.csv', header=0, delim_whitespace=True)
data

t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

data.shape
data.columns

########BROKE HERE -- error 'Gender' is not a column name
print(data['Gender'])

data[data['Gender'] == 'Female']['VIQ'].mean()

groupby_gender = data.groupby('Gender')

for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))

#####PLOTTING 

scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])   
####recieving error that objects arent in the columns


