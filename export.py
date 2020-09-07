# -*- coding: utf-8 -*-


# Importar liberias necesarias
import pandas as pd
import numpy as np
from datetime import datetime as dt
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math

from sodapy import Socrata

# Forma recomendada por la fuente de los datos 
# https://dev.socrata.com/foundry/www.datos.gov.co/gt2j-8ykr

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

## ARIMA

# Diccionario con valores p,d,q definidos
param = {'Bog': (1, 2, 2), 'Bar': (0, 2, 2), 'Cal': (2, 2, 0), 'Car': (2, 2, 1),'Med': (0, 2, 1)}

# Ajuste y pronóstico de los modelos por ciudad
# Contagios, Recuperados, Muertes

# Inicializar variables
v_poblacion = ['cum_','rcum_','mcum_']   # Nombres de dataframes recuperados, muertes, contagios
P = 1     # Número de registros utilizados para el test (Ya que se está en un punto crítico los últimos datos son relevante para el ajuste)

# Crear una lista de fechas para las cuales se van a generar los pronósticos
date = pd.to_datetime(cum_Med.loc[len(cum_Med)-1,['Fecha']])
Fecha = pd.date_range(start=date[0],periods=91,freq='D')[1:]

for j in ciudades:
  c = j[:3]   # Tomar iniciales de las ciudades
  # Crear dataframe para almacenar las predicciones, agregar Fechas
  globals()['pred_90_' + str(c)] = pd.DataFrame(Fecha, columns=['Fecha'])

  for i in v_poblacion:
    name_df = globals()[str(i)+ str(c)]      # Asignar a 'name_df' el dataframe i para la ciudad c
    prueba = np.array(name_df.iloc[:,2])     # Seleccionar columna de número de casos
    train, test = prueba[:-P], prueba[-P:]   # Crear Train y Test

    # Ajuste del modelo
    model_arima = ARIMA(train, order=(param[c]))
    model_fit_arima= model_arima.fit(disp=0)

    # Predicción del modelo
    pred_aj = np.round(model_fit_arima.predict(typ='levels'))                            # Predicción del ajuste (datos reales)
    pred = np.round(model_fit_arima.forecast(90+P)[0])                                   # Predicción 91 días adelante
    name_df['Predicción'] = [None]*param[c][1] + pred_aj.tolist() + pred[:P].tolist()    # agregar al dataframe original la columna predicción

    # Agregar a dataframe de predicciones cada columna (variable)
    globals()['pred_90_' + str(c)][i] = np.round(model_fit_arima.forecast(90+P)[0][P:])

  # Calcular y agregar al dataframe de predicciones los casos Activos y los casos por día
  globals()['pred_90_' + str(c)]['Activos'] = globals()['pred_90_' + str(c)]['cum_'] - globals()['pred_90_' + str(c)]['rcum_'] - globals()['pred_90_' + str(c)]['mcum_']
  globals()['pred_90_' + str(c)]['Contagios por dia'] = globals()['pred_90_' + str(c)]['cum_'] - globals()['pred_90_' + str(c)]['cum_'].shift(1)
  globals()['pred_90_' + str(c)]['Contagios por dia'].fillna(globals()['pred_90_' + str(c)]['cum_'][0] - globals()['cum_'+str(c)]['Contagios'][len(name_df)-1], inplace=True)
  globals()['pred_90_' + str(c)]['Ciudad'] = j
  globals()['pred_90_' + str(c)].columns = ['Fecha', 'Contagios acum', 'Recuperados', 'Muertes', 'Activos', 'Contagios por dia', 'Ciudad']
  globals()['pred_90_' + str(c)] = globals()['pred_90_' + str(c)].reindex(columns = ['Fecha', 'Contagios acum', 'Activos', 'Muertes', 'Recuperados', 'Contagios por dia', 'Ciudad'])

## SIR

#Extracción de las tasas a partir de los datos
for city in ciudades:
  tasa_cont = []
  tasa_recup = []
  tasa_muertos = []

  df_base = union[union['Ciudad'] == city].reset_index()
  for i in range(len(df_base)-1):
    tasa_cont.append(int(df_base.loc[i+1,['Contagios por dia']])/int(df_base.loc[i,['Activos']]))
    tasa_recup.append(int(df_base.loc[i+1,['Recuperados']])/int(df_base.loc[i,['Contagios acum']]))
    tasa_muertos.append(int(df_base.loc[i+1,['Muertes']])/int(df_base.loc[i,['Contagios acum']]))

  #Se ingresan valores al DF
  globals()['df_tasas_'+str(city[0:3])] = pd.DataFrame(data = tasa_cont,columns=['Tasa Contagios'])
  globals()['df_tasas_'+str(city[0:3])]['Tasa Recup'] = tasa_recup
  globals()['df_tasas_'+str(city[0:3])]['Tasa muertes'] = tasa_muertos

# Grados p,d,q utilizados basados en análisis de PACF y ACF

param = { 'Bar': (5,1,1) , 'Bog': (4,1,1), 'Cal': (2,1,1), 'Car': (5,1,1),'Med': (5,1,0)}

# Ajuste de los modelos
# Tasas contagios, recuperados, muertes
v_tasas = ['Tasa Contagios','Tasa Recup','Tasa muertes']

dias_eliminados = 14
P = 5  # Número de registros utilizados para test

# Crear una lista de fechas para las cuales se van a generar los pronósticos
date = pd.to_datetime(cum_Med.loc[len(cum_Med)-1,['Fecha']])
Fecha = pd.date_range(start=date[0],periods=91,freq='D')[1:]

for c in ciudades:
  c = c[:3]   # Tomar iniciales de las ciudades
  # Crear dataframe para almacenar las predicciones, agregar Fechas
  globals()['pred_90_tasas' + str(c)] = pd.DataFrame(Fecha, columns=['Fecha'])

  for i in v_tasas:
    name_df = globals()['df_tasas_'+ str(c)]          # Asignar a 'name_df' el dataframe i para la ciudad c
    prueba = np.array(name_df[i][dias_eliminados:])   # Seleccionar columna de cada tasa
    train, test = prueba[:-P], prueba[-P:]            # Crear Train y Test

    # Ajuste del modelo
    globals()['model_' + str(i) + str(c)] = ARIMA(train, order=param[c])
    globals()['model_fit_' + str(i) + str(c)] = globals()['model_' + str(i) + str(c)].fit(disp=0)

    # Predicción del modelo
    pred_aj = globals()['model_fit_' + str(i) + str(c)].predict(typ='levels')
    pred = globals()['model_fit_' + str(i) + str(c)].forecast(90+P)[0]
    name_df['Pred '+str(i)] = [None]*(param[c][1]+dias_eliminados) + pred_aj.tolist() + pred[:P].tolist()

    # Agregar a dataframe de predicciones cada columna (variable)
    globals()['pred_90_tasas'+ str(c)][i] = globals()['model_fit_' + str(i) + str(c)].forecast(90+P)[0][P:]

susceptibles = {'Bogotá D.C.':7743955,'Medellín':2533424,'Barranquilla':1274250,'Cali':2252616,'Cartagena de Indias':1028736}

# Parametros optimos
interaccion_popt_Bog = 0.9291864539736386
interaccion_popt_Med = 0.9160650237678833
interaccion_popt_Bar = 0.8653192483914145
interaccion_popt_Car = 0.801527242069967
interaccion_popt_Cal = 0.9139842159543023

#Se crea vector de fechas siguientes a la del último corte
fecha_pred = pd.to_datetime(union[union['Ciudad']=='Barranquilla'].reset_index().loc[len(union[union['Ciudad']=='Barranquilla'])-1,['Fecha']])
fecha_pred = pd.date_range(start=date[0],periods=91,freq='D')[1:]

def SIR_Pron(no_per_pron,city, interaccion):
  df_base = union[union['Ciudad'] == city].reset_index()
  df_tasas=globals()['pred_90_tasas'+ str(city[:3])]
  n = len(df_base)#En n+1 arranca a correr
  duracion_media = 14
  susceptibles = {'Bogotá D.C.':7743955,'Medellín':2533424,'Barranquilla':1274250,'Cali':2252616,'Cartagena de Indias':1028736}
  total_pob = susceptibles[city] 
  
  tasas = pd.DataFrame() #Tiene que tener una longitud = al número de periodos a pronosticar
  casos_confirm = int(df_base.loc[n-1,['Contagios acum']])
  recuper = int(df_base.loc[n-1,['Recuperados']])
  muertos = int(df_base.loc[n-1,['Muertes']])
  suscep = total_pob - int(df_base.loc[n-1,['Contagios acum']]) #susceptibles
  activos = casos_confirm - recuper - muertos
  #Deben ser en blanco para pronosticar
  suscept_list = [] 
  total_casos_confirm = []
  total_recuper = []
  total_muertos = []
  contagios_list =[]
  recuperados_list = []
  muertos_list = []
  activos_list = []


  for i in range(0,no_per_pron): #desde el dato n+1
    
    #Actualiza variables
    tasa_contagio = float(df_tasas.loc[i,['Tasa Contagios']])
    tasa_recu = float(df_tasas.loc[i,['Tasa Recup']])
    tasa_muertes = float(df_tasas.loc[i,['Tasa muertes']])
    relacion = suscep/total_pob
    contagios_dia = math.ceil(activos*tasa_contagio*interaccion*relacion)
    recuperados_dia = math.ceil(activos*tasa_recu/duracion_media)
    muertos_dia = math.ceil(activos*tasa_muertes/duracion_media)
    if contagios_dia < 0:
      contagios_dia = 0
    if recuperados_dia < 0:
      recuperados_dia = 0
    if muertos_dia < 0:
      muertos_dia =0
    casos_confirm = casos_confirm + contagios_dia
    recuper = recuper + recuperados_dia
    muertos = muertos + muertos_dia
    activos = casos_confirm - recuper - muertos
    suscep = total_pob - casos_confirm
    
    #Actualiza listas díarias
    contagios_list.append(contagios_dia)
    recuperados_list.append(recuperados_dia)
    muertos_list.append(muertos_dia)
    
    #Actualiza listas
    suscept_list.append(suscep)
    total_recuper.append(recuper)
    total_muertos.append(muertos)
    total_casos_confirm.append(casos_confirm)
    activos_list.append(activos)
   
  df = pd.DataFrame(data=fecha_pred,columns=['Fecha'])
  df['Susceptibles'] = suscept_list
  df['Contagios acum'] = total_casos_confirm
  df['Recuperados'] = total_recuper
  df['Muertes'] = total_muertos
  df['Contagios por dia'] = contagios_list
  df['Recuperados_dia'] = recuperados_list
  df['Muertos_dia'] = muertos_list
  df['Activos'] = activos_list
  
  return df

# Se genera DF de pronóstico SIR para cada ciudad 
for city in ciudades:
  globals()['df_pronSIR_'+str(city[0:3])] = SIR_Pron(90, city, globals()['interaccion_popt_'+str(city[:3])])
  globals()['df_pronSIR_'+str(city[0:3])] = globals()['df_pronSIR_'+str(city[0:3])][['Fecha','Contagios acum','Activos','Muertes','Recuperados','Contagios por dia']]
  globals()['df_pronSIR_'+str(city[0:3])]['Ciudad'] = city
  #Se crea DF combinado para pronósticos de cada ciudad
  globals()['pron_merge_' + str(city[0:3])] = pd.merge(globals()['pred_90_' + str(city[0:3])],globals()['df_pronSIR_'+str(city[0:3])],on=['Fecha','Ciudad'],suffixes=['_arima','_sir'])

# Crea un df unicamente con las columnas del poderado
r = 14 # parámetro de días para transferir los datos del modelo ARIMA a SIR
pesos = np.array( np.linspace(0,100,r).tolist() + [100 for i in range(len(pron_merge_Med)-r)])/100
v_columns = ['Contagios acum', 'Activos', 'Muertes', 'Recuperados', 'Contagios por dia']
pron_final = pd.DataFrame()

for city in ciudades:
  pron_df = globals()['pron_merge_' + str(city[0:3])]
  v_col = ['Fecha']
  for i in v_columns:
    pron_df[str(i)] = np.round(pron_df[str(i)+'_arima']*(1-pesos) + pron_df[str(i)+'_sir']*pesos)
    v_col.append(str(i))
  pron_df['Ciudad'] = city
  v_col.append('Ciudad')
  pron_final = pd.concat([pron_final, pron_df])

# Unir dataframe reales + dataframe pronosticos
df_final = pd.concat([union,pron_final[v_col]]).reset_index().drop('index', axis=1)

#Convertir a enteros
df_final['Contagios acum'] = pd.to_numeric(df_final['Contagios acum'], downcast='signed')
df_final['Activos'] = pd.to_numeric(df_final['Activos'], downcast='signed')
df_final['Muertes'] = pd.to_numeric(df_final['Muertes'], downcast='signed')
df_final['Recuperados'] = pd.to_numeric(df_final['Recuperados'], downcast='signed')
df_final['Contagios por dia'] = pd.to_numeric(df_final['Contagios por dia'], downcast='signed')

df_final.to_csv('data.csv')

