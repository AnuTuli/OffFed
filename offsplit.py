import pandas as pd
import numpy as np
 
# create a DataFrame
x=0
df=pd.read_csv('off.csv')
df1=df.iloc[:134456]
df2=df.iloc[134456:]
df11=df1.iloc[:33614]
df21=df2.iloc[:17592]
df_c=pd.concat([df11, df21])
df_c.to_csv('off1.csv')
df11=df1.iloc[33614:67228]
df21=df2.iloc[17592:35184]
df_c=pd.concat([df11, df21])
df_c.to_csv('off2.csv')
df11=df1.iloc[67228:100842]
df21=df2.iloc[35184:52776]
df_c=pd.concat([df11, df21])
df_c.to_csv('off3.csv')
df11=df1.iloc[100842:]
df21=df2.iloc[52776:]
df_c=pd.concat([df11, df21])
df_c.to_csv('off4.csv')

