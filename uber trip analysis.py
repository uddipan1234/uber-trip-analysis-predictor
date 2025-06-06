import pandas as pd
df=pd.read_csv(r"C:\Users\welcome\Downloads\Uber-Jan-Feb-FOIL.csv")
print(df.head(10))

df.drop(columns=['dispatching_base_number'],inplace=True)
print(df.columns)
#CONVERTING DATE TIME TO DATE TIME FORMAT
df['date']=pd.to_datetime(df['date'],errors='coerce')
 #feature enginering
df['hour']=df['date'].dt.hour
df['weekday']=df['date'].dt.weekday
df['Month']=df['date'].dt.month
#SPILITING DATA
x=df[['hour','weekday','Month','active_vehicles']]
y=df['trips']
#train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#callingg model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
#prediction
y_pred=model.predict(x_test)
#evaluation
from sklearn.metrics import mean_absolute_error ,mean_squared_error,r2_score
print('Mean absolute error:',mean_absolute_error(y_test,y_pred))
print('Mean squared error:',mean_absolute_error(y_test,y_pred))
print('r2 score:',r2_score(y_test,y_pred))


