#%% IMPORTS

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt

#%% CARGA DE DATOS

df = pd.read_csv('path_to_file.csv')

#%% PRE-PROCESAMIENTO DE DATOS

df.rename(columns={'income_kUSD':'income'}, inplace=True)

#%% ESTANDARIZACIÓN DE DATOS

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

#%% ENTRENAMIENTO

kmeans = KMeans(n_clusters=4, n_init='auto', random_state=42)
kmeans.fit(scaled_df)

#%% INCLUIR LOS CLUSTERS GENERADOS AL DATAFRAME ORIGINAL

df['cluster'] = kmeans.labels_

#%% CALCULAR LA MEDIA DE CADA VARIABLE POR CLUSTER

cluster_means = df.groupby('cluster').mean() 

#%% VISUALIZACIÒN DE RESULTADOS PARA ANÁLISIS COMERCIAL

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

sns.scatterplot(
    x=df['income'], 
    y=df['spending_score'],
    hue=df['loyalty_score'],
    palette='dark:#5A9_r',
    ax=axes[0])

axes[0].set_title('Análisis de fidelidad de clientes: Ingresos vs. Puntaje de gastos')

sns.scatterplot(
    x=df['income'], 
    y=df['spending_score'],
    hue=df['cluster'],
    palette='viridis',
    ax=axes[1])

axes[1].set_title('Relación de grupos: Ingresos vs. Puntaje de gastos')

plt.show()

cluster_means.plot(kind='bar', figsize=(10, 6), width=0.8)
plt.title('Media de las variables por cada Cluster')
plt.xlabel('Cluster')
plt.ylabel('Valor Promedio')
plt.show()
