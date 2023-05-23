# %%
# This exercise is for Advanced Python Programming, a course presented in Maktabkhoone Course by Jadi 

# %%
pip install requests 

# %%
pip install catboost

# %% [markdown]
# ### import Libraries 

# %%
#import essential libraries 
import requests
from bs4 import BeautifulSoup
import re 
import pandas as pd 

import mysql.connector
from mysql.connector import Error

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pylab
import warnings

# %% [markdown]
# ## Fetch the Data from Truecar

# %%
def get_car_data(car_name, url, df): 
    r = requests.get(url)
    print(r)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    car_model_year = soup.find_all('div', {"data-test" : "vehicleCardYearMakeModel"})
    car_price = soup.find_all('div', {"data-test" : "vehicleCardPricingBlockPrice"})
    car_mileage = soup.find_all('div', {"data-test" : "vehicleMileage"})
    
    car_year1 = list()
    car_model1 = list()
    #car_model_year1 = list()
    car_price1 = list()
    car_mileage1 = list()

    for i in range(3, 30): 
        car_year1.append(car_model_year[i].text[0:4])
        car_model1.append(car_model_year[i].text[4:])
        #car_model_year1.append(car_model_year[i].text)
        a = car_price[i].text[1:]
        car_price1.append(int(a.replace(',', '')))
        a = car_mileage[i].text.strip()[:-5]
        car_mileage1.append(int(a.replace(',', '')))
    
    df1 = pd.DataFrame(list(zip(car_year1, car_model1, car_price1, car_mileage1)), columns =['Year','Model', 'Price', 'Mileage'])    
    df2 = df.append(df1, ignore_index=True)
    df2.head()
    #df.to_csv(r'C:\Users\Hassan\Desktop\Django_Course\used_cars.csv')
    
    #for i in range(3,30):
    #print(i-2 , "---", car_model_year[i].text, "----", car_price[i].text, "----", car_mileage[i].text)
        
    return df2 

# %%
car_name = 'bmw'
url = "https://www.truecar.com/used-cars-for-sale/listings/" + car_name    
df = pd.DataFrame()

df = get_car_data(car_name, url, df) 


# %%
df

# %%
df.dtypes

# %%
#https://www.truecar.com/used-cars-for-sale/listings/bmw/?page=2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

for i in range(2,5): 
    url = "https://www.truecar.com/used-cars-for-sale/listings/" + car_name + "/?page=" + str(i) 
    print(url)
    df = get_car_data(car_name, url, df) 

    

# %%
df.tail

# %% [markdown]
# ## Save the Data in DB 

# %% [markdown]
# ### Check Availability of MySQL - Connection Check 

# %%
try:
    connection = mysql.connector.connect(host='localhost', database='car_data', user='admin', password='')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")


# %% [markdown]
# ### Create a New Database for BMW Data 

# %%
try:
    connection = mysql.connector.connect(host='localhost', database='car_data', user='admin',password='')

    mySql_Create_Table_Query = """CREATE TABLE BMW ( 
                             Id int(11) NOT NULL,
                             Year varchar(250) NOT NULL,
                             Model varchar(250) NOT NULL,
                             Price float NOT NULL,
                             Milage float NOT NULL,
                             PRIMARY KEY (Id)) """

    cursor = connection.cursor() 
    result = cursor.execute(mySql_Create_Table_Query)
    print("BMW Table created successfully ")
    
    #cursor.execute('INSERT INTO bmw VALUES (\'Amir\',\'Far\',\'Niki\') ')
    connection.commit()
    cursor.close()

except mysql.connector.Error as error:
    print("Failed to create table in MySQL: {}".format(error))
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")


# %% [markdown]
# ### Insert Data into the BMW DB 

# %%
connection = mysql.connector.connect(host='localhost', database='car_data', user='admin', password='')
print("Connected to DB")

cursor = connection.cursor() 

for i in range(0,107): 
    sql = "INSERT INTO bmw (id, year, model, price, milage) VALUES (%s, %s, %s, %s, %s)"
    val = (str(i), df["Year"][i], df["Model"][i], int(df["Price"][i]), int(df["Mileage"][i]) )          
    cursor.execute(sql, val)

connection.commit()

cursor.close()
connection.close()
print("Data entered successfully.")

# %% [markdown]
# ## Basic Understanding of Data

# %%


# %%
df.shape

# %%
df.info()

# %%
df.describe()


# %%
df.isnull().sum().to_frame().rename(columns={0:"Total No. of Missing Values"})

# %%
print("Duplicate Values =",df.duplicated().sum())

# %%
df.select_dtypes(include="object").head()

# %%
print(df["Model"].unique()) 
len(df["Model"].unique()) 

# %%
print(df["Year"].unique()) 
len(df["Year"].unique()) 

# %% [markdown]
# ## Data Cleaning 

# %% [markdown]
# باید مدل‌ها را درست کنم. 

# %%


# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.distplot(df["Price"],color="red",kde=True)
plt.title("Car Price Distribution",fontweight="black",pad=20,fontsize=20)

plt.subplot(1,2,2)
sns.boxplot(y=df["Price"],palette="Set2")
plt.title("Car Price Spread",fontweight="black",pad=20,fontsize=20)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(14,6))
counts = df["Model"].value_counts()
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel("Car Model")
plt.ylabel("Total No. of cars sold")
plt.title("Total Cars produced by Companies", pad=20, fontweight="black", fontsize=20)
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ### Visualizing Car Company w.r.t Price

# %%
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
sns.boxplot(x="Model",y="Price",data=df)
plt.xticks(rotation=90)
plt.title("Model vs Price", pad=10, fontweight="black", fontsize=20)

plt.subplot(1,2,2)
x = pd.DataFrame(df.groupby("Model")["Price"].mean().sort_values(ascending=False))
sns.barplot(x=x.index,y="Price",data=x) 
plt.xticks(rotation=90)
plt.title("Model vs Average Price", pad=10, fontweight="black", fontsize=20)
plt.tight_layout()
plt.show()

# %%
Visualizing "WheelBase" & "Curbweight" Features

# %%
def scatter_plot(cols):
    x=1
    plt.figure(figsize=(15,6))
    for col in cols:
        plt.subplot(1,2,x)
        sns.scatterplot(x=col,y="Price",data=df,color="blue")
        plt.title(f"{col} vs Price",fontweight="black",fontsize=20,pad=10)
        plt.tight_layout()
        x+=1

# %%
scatter_plot(["Mileage","Price"])

# %% [markdown]
# ## Data Preprocessiong

# %%
#1. Creating new DataFrame with all the useful Features.

# %%
new_df = df[['Price','Mileage']]

# %%
#2.Creating Dummies Variables for all the Categorical Features.

# %%
# 3. Feature Scaling of Numerical Data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

num_cols = ['Price','Mileage']

new_df[num_cols] = scaler.fit_transform(new_df[num_cols])

# %%
new_df.head()

# %%


# %% [markdown]
# ### Selecting Features & Labels for Model Training & Testing

# %%
x = new_df.drop(columns=["Price"])
y = new_df["Price"]

# %%
warnings.filterwarnings("ignore")
%matplotlib inline
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.columns",None)
pd.set_option("display.max.rows",None)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# %% [markdown]
# ### Splitting Data for Model Traning & Testing.

# %%
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# %%
print("x_train - >  ",x_train.shape)
print("x_test - >  ",x_test.shape)
print("y_train - >  ",y_train.shape)
print("y_test - >  ",y_test.shape)

# %% [markdown]
# ## Model Building

# %%
training_score = []
testing_score = []

# %%
from sklearn.metrics import r2_score
def model_prediction(model):
    model.fit(x_train,y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    a = r2_score(y_train,x_train_pred)*100
    b = r2_score(y_test,x_test_pred)*100
    training_score.append(a)
    testing_score.append(b)
    
    print(f"r2_Score of {model} model on Training Data is:",a)
    print(f"r2_Score of {model} model on Testing Data is:",b)

# %% [markdown]
# ### Linear-Regression Model

# %%
model_prediction(LinearRegression())


# %% [markdown]
# ### Decision-Tree-Regressor Model

# %%
model_prediction(DecisionTreeRegressor())


# %% [markdown]
# ### Random-Forest-Regressor Model

# %%
model_prediction(RandomForestRegressor())

# %%
#Ada-Boost-Regressor Model
model_prediction(AdaBoostRegressor())


# %%
#Gradient-Boosting-Regressor Model
model_prediction(GradientBoostingRegressor())

# %%
#LGMB Regressor Model
from lightgbm import LGBMRegressor

model_prediction(LGBMRegressor())

# %%
#XGBRegressor Model
model_prediction(XGBRegressor())

# %%
#Cat-Boost-Regressor Model
from catboost import CatBoostRegressor
model_prediction(CatBoostRegressor(verbose=False))

# %% [markdown]
# ## All Model Performance Comparison

# %%
models = ["Linear Regression","Decision Tree",
          "Random Forest","Ada Boost","Gradient Boost","LGBM","XGBoost","CatBoost"]

# %%
df_models = pd.DataFrame({"Algorithms":models,
                   "Training Score":training_score,
                   "Testing Score":testing_score})

# %%
df_models

# %%
df.plot(x="Algorithms",y=["Training Score","Testing Score"], figsize=(16,6),kind="bar",
        title="Performance Visualization of Different Models",colormap="Set1")
plt.show()

# %%


# %%


# %%



