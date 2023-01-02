import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

house_df=pd.read_csv('house_clean.csv')
house_df.head()

fig, ax=plt.subplots(figsize=(20,8))
sns.boxplot(data=house_df);

house_df1 =house_df.drop(columns='SalePrice')

fig, ax=plt.subplots(figsize=(20,8))
sns.boxplot(data=house_df1);

house_df1.columns

house_df1.head()

fig, ax=plt.subplots(figsize=(20,8))
sns.heatmap(house_df.corr(), annot=False)

#X_h3=house_df[['LotArea','YearBuilt',]]
X_h3=house_df[['LotArea', 'YearBuilt', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'GarageCars']]
y_h=house_df.SalePrice

#print (house_df['SalePrice'])

from sklearn.linear_model import LinearRegression

lr5=LinearRegression()
lr5.fit(X_h3,y_h)

coef_df=pd.DataFrame({'Features':X_h3.columns,'Effect Size':lr5.coef_}).set_index('Features').sort_values(by='Effect Size',ascending=False)
print (coef_df)


#print('Model 3 predicted sale price: $'+str(lr5.predict([[11250]])[0]))

actuals=house_df['SalePrice'] # this is the same as our target series
preds=lr5.predict(X_h3)

compare_df=pd.DataFrame({'Predicted Sale Price':preds,'Actual Sale Price':actuals}).set_index(house_df.index)
print(compare_df.head(100))

compare_df.plot(x='Predicted Sale Price',y='Actual Sale Price',kind='scatter')

from sklearn.metrics import mean_squared_error
import numpy as np

rmse=np.sqrt(mean_squared_error(preds,actuals))
print(rmse)

from sklearn.metrics import mean_absolute_error

mae=mean_absolute_error(preds,actuals)
print(mae)

lr5.score(X_h3,y_h)

from sklearn.model_selection import train_test_split
X_trainh, X_testh, y_trainh, y_testh = train_test_split(X_h3,y_h, train_size=0.8,random_state=888)

lr6=LinearRegression()
lr6.fit(X_trainh,y_trainh)

print('Train score: '+str(lr6.score(X_trainh,y_trainh)))
print('Test score: '+str(lr6.score(X_testh,y_testh)))