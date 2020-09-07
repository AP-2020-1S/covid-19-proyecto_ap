import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Forma recomendada por la fuente de los datos 
# https://dev.socrata.com/foundry/www.datos.gov.co/gt2j-8ykr

from sodapy import Socrata

client = Socrata("www.datos.gov.co",None)
results = client.get_all("gt2j-8ykr")
df = pd.DataFrame.from_records(results)

# Cambio del tipo de datos
df['fecha_de_notificaci_n'] = pd.to_datetime(df['fecha_de_notificaci_n'])
df['fecha_diagnostico'] = pd.to_datetime(df['fecha_diagnostico'])
df['fecha_recuperado'] = pd.to_datetime(df['fecha_recuperado'])
df['fecha_reporte_web'] = pd.to_datetime(df['fecha_reporte_web'])
df['fecha_de_muerte'] = pd.to_datetime(df['fecha_de_muerte'])
df['edad'] = pd.to_numeric(df['edad'])

# Unificar valores columna "sexo"
df['sexo'] = df['sexo'].str.upper()
df['sexo'].value_counts()

# Top 5 ciudades con más casos
df_c = df['ciudad_de_ubicaci_n'].value_counts().head(5)

# Segmentar DataFrame con las 5 ciudades principales
df = df[df['ciudad_de_ubicaci_n'].isin(['Bogotá D.C.','Medellín','Barranquilla', 'Cali', 'Cartagena de Indias'])]

# Dataframe general

# Dataframe Contagios
datos_contagios = df.groupby(['fecha_reporte_web', 'ciudad_de_ubicaci_n']).size().reset_index()
datos_contagios.columns = ['Fecha', 'Ciudad','Contagios']

# Dataframe recuperados
datos_recuperados = df.groupby(['fecha_recuperado', 'ciudad_de_ubicaci_n']).size().reset_index()
datos_recuperados.columns = ['Fecha', 'Ciudad','Recuperados']

# Dataframes muertes
datos_muertes = df.groupby(['fecha_de_muerte', 'ciudad_de_ubicaci_n']).size().reset_index()
datos_muertes.columns = ['Fecha', 'Ciudad','Muertes']
datos_muertes

# Consolidado
data_final = datos_contagios.merge(datos_recuperados, on=['Fecha', 'Ciudad'], how='left')
data_final = data_final.merge(datos_muertes,on=['Fecha', 'Ciudad'], how='left').fillna(0)

# Lista de ciudades
ciudades = ['Bogotá D.C.','Medellín','Barranquilla', 'Cali', 'Cartagena de Indias']

# Dataframes contagios acumulado por ciudad
for i in ciudades:
  globals()['cum_'+str(i[0:3])] = data_final[data_final['Ciudad']==i].drop(['Ciudad', 'Recuperados', 'Muertes'], axis=1).reset_index()
  globals()['cum_'+str(i[0:3])]['Contagios'] = globals()['cum_'+str(i[0:3])]['Contagios'].cumsum()

# DataFrames contagios diarios por ciudad por ciudad
for i in ciudades:
  globals()['c_'+str(i[0:3])] = data_final[data_final['Ciudad']==i].drop(['Ciudad', 'Recuperados', 'Muertes'], axis=1)

 # DataFrames recuperados acumulado por ciudad
for i in ciudades:
  globals()['rcum_'+str(i[0:3])] = data_final[data_final['Ciudad']==i].drop(['Ciudad', 'Contagios', 'Muertes'], axis=1).reset_index()
  globals()['rcum_'+str(i[0:3])]['Recuperados'] = globals()['rcum_'+str(i[0:3])]['Recuperados'].cumsum()

# DataFrames muertes acumulado por ciudad
for i in ciudades:
  globals()['mcum_'+str(i[0:3])] = data_final[data_final['Ciudad']==i].drop(['Ciudad', 'Contagios', 'Recuperados'], axis=1).reset_index()
  globals()['mcum_'+str(i[0:3])]['Muertes'] = globals()['mcum_'+str(i[0:3])]['Muertes'].cumsum()

# DataFrames activos acumulado por ciudad
for i in ciudades:
  globals()['acum_'+str(i[0:3])] = data_final[data_final['Ciudad']==i] 
  globals()['acum_'+str(i[0:3])]['Contagios'] = globals()['acum_'+str(i[0:3])]['Contagios'].cumsum()
  globals()['acum_'+str(i[0:3])]['Recuperados'] = globals()['acum_'+str(i[0:3])]['Recuperados'].cumsum()
  globals()['acum_'+str(i[0:3])]['Muertes'] = globals()['acum_'+str(i[0:3])]['Muertes'].cumsum()
  globals()['acum_'+str(i[0:3])]['Activos'] = globals()['acum_'+str(i[0:3])]['Contagios'] - globals()['acum_'+str(i[0:3])]['Recuperados'] - globals()['acum_'+str(i[0:3])]['Muertes']  
  globals()['acum_'+str(i[0:3])] = globals()['acum_'+str(i[0:3])].drop(['Ciudad', 'Contagios', 'Recuperados', 'Muertes'], axis=1)

# DataFrames union
for i in ciudades:
  globals()['union_'+str(i[0:3])] = []
  globals()['union_'+str(i[0:3])] = pd.merge(pd.merge(pd.merge(pd.merge(globals()['cum_'+str(i[0:3])],globals()['acum_'+str(i[0:3])],on='Fecha'),globals()['mcum_'+str(i[0:3])],on='Fecha'),globals()['rcum_'+str(i[0:3])],on='Fecha'),globals()['c_'+str(i[0:3])],on='Fecha')
  globals()['union_'+str(i[0:3])] = globals()['union_'+str(i[0:3])].drop(columns=['index_x','index_y','index'])
  globals()['union_'+str(i[0:3])]['Ciudad'] = i

union = np.concatenate([union_Bar,union_Bog,union_Cal,union_Car,union_Med])
union = pd.DataFrame(union,columns=['Fecha','Contagios acum','Activos','Muertes','Recuperados','Contagios por dia','Ciudad'])

#Convertir a enteros
union['Contagios acum'] = pd.to_numeric(union['Contagios acum'], downcast='signed')
union['Activos'] = pd.to_numeric(union['Activos'], downcast='signed')
union['Muertes'] = pd.to_numeric(union['Muertes'], downcast='signed')
union['Recuperados'] = pd.to_numeric(union['Recuperados'], downcast='signed')
union['Contagios por dia'] = pd.to_numeric(union['Contagios por dia'], downcast='signed')

union.to_csv('data.csv')