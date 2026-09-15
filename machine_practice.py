import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from xml.parsers.expat import model

df=pd.read_csv("g:\check\project\house_price.csv")

plt.style.use('dark_background')
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
'''
fig = px.scatter(x="area_sqft",y="price_lakh",data_frame=df,
                 title="Area vs Price")
fig.show()'''



target=df["price_lakh"]
inputs=df[["area_sqft","bedrooms","bathrooms","age_years","distance_to_city_km"]]






model=LinearRegression()
model.fit(inputs,target)
predictions =model.predict(inputs)
rmse = root_mean_squared_error(target, predictions, multioutput='raw_values')
print(rmse)

predicted_price = model.predict([[2000, 3, 2, 5, 10]])
print("Predicted price for a house with 2000 sqft, 3 bedrooms, 2 bathrooms, 5 years old, and 10 km from the city:", predicted_price[0])



plt.scatter(target, predictions)
#plt.plot(df['area_sqft'],df['price_lakh'], color='red', linewidth=2)
plt.xlabel("Actual Price (lakh)")
plt.ylabel("Predicted Price (lakh)")
plt.title("Actual vs Predicted House Prices")

plt.show()
area=float(input("Enter the area in sqft: "))
bedrooms=int(input("Enter the number of bedrooms: "))
bathrooms=int(input("Enter the number of bathrooms: "))
age=int(input("Enter the age of the house in years: "))
distance=float(input("Enter the distance to the city in km: "))

predicted_price = model.predict([[area, bedrooms, bathrooms, age, distance]])
print("Predicted price for the house:", predicted_price[0])