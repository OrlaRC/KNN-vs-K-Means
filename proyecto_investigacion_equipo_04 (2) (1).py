# =============================================================================
# UNIVERSIDAD TECNOLÓGICA DE QUERÉTARO
# PRAC-U3-04-INVESTIGACION: Contraste de Enfoques Supervisados y No Supervisados
# Equipo / Integrantes: Israel Gómez, Sandra Zoe Cabrera, Orlando Rubio, Marco Antonio Gomez
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# --- BLOQUE 1: CARGA Y RECONOCIMIENTO DEL DATASET
# =============================================================================
# URL de origen: https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset
# Descripcion: Dataset de juegos en Steam con resenas, precios, tiempos de juego
# y estimaciones de duenos (owners). Variable categorica objetivo: 'owners'.

script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(script_dir, 'archive', 'steam.csv')
df = pd.read_csv(ruta_csv)
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# =============================================================================
# --- BLOQUE 2: PREPROCESAMIENTO E INVESTIGACION DE ESCALADO
# =============================================================================
print("\n--- INICIANDO PREPROCESAMIENTO ---")

df_clean = df.dropna().copy()

features_numericas = ['price', 'positive_ratings', 'negative_ratings', 'average_playtime']
variable_objetivo = 'owners'

X = df_clean[features_numericas]
y = df_clean[variable_objetivo]

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=features_numericas)

print("Preprocesamiento completado. Datos escalados con StandardScaler.")
print(f"Caracteristicas: {features_numericas}")
print(f"Clases en owners: {y.nunique()}")
print(f"Muestras totales: {len(df_clean)}")

# =============================================================================
# --- BLOQUE 3: ENFOQUE SUPERVISADO (KNN CLASSIFIER)
# =============================================================================
print("\n--- BLOQUE 3: KNN (SUPERVISADO) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("Reporte de clasificacion (precision, recall, f1-score):")
print(classification_report(y_test, y_pred_knn, zero_division=0))
print("Matriz de confusion:")
print(confusion_matrix(y_test, y_pred_knn))
accuracy = knn.score(X_test, y_test)
print(f"Exactitud (accuracy): {accuracy:.4f}")

# =============================================================================
# --- BLOQUE 4: ENFOQUE NO SUPERVISADO (K-MEANS CIEGO)
# =============================================================================
print("\n--- BLOQUE 4: K-MEANS (NO SUPERVISADO) ---")

X_kmeans = X_scaled.copy()

kmeans = KMeans(n_clusters=13, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_kmeans)

print(f"Agrupamiento K-Means completado con K={kmeans.n_clusters} clusters.")

# =============================================================================
# --- BLOQUE 5: MATRIZ DE COINCIDENCIA CRUZADA
# =============================================================================
print("\n--- BLOQUE 5: MATRIZ CRUZADA (pd.crosstab) ---")

tabla_cruzada = pd.crosstab(
    y,
    clusters,
    rownames=['Owners (Real)'],
    colnames=['Cluster K-Means']
)
print(tabla_cruzada.to_string())

"""
=============================================================================
SECCION DE INVESTIGACION TEORICA Y CONCLUSIONES (DOCSTRING OBLIGATORIO)
=============================================================================

PREGUNTA 1:
Analizando detenidamente los resultados de su matriz cruzada impresa por
pd.crosstab: El algoritmo no supervisado (K-Means) logro reconstruir y
redescubrir de forma exacta los patrones o categorias reales del dataset sin
conocer las etiquetas? Justifiquen los aciertos o desviaciones observadas.

RESPUESTA 1:
No, K-Means NO logro reconstruir las categorias reales de owners. La matriz
cruzada muestra que la mayoria de los juegos, independientemente de su rango
de duenos, se concentraron en los clusters 0 y 5 (ej: 11679 de los 18596
juegos con 0-20000 owners fueron al cluster 0; 5354 al cluster 5). Esto revela
que las caracteristicas numericas seleccionadas (precio, reseñas positivas,
reseñas negativas, tiempo de juego promedio) no se correlacionan fuertemente
con la popularidad medida en dueños. Juegos con pocos dueños pero buenas
reseñas y precio similar se agrupan naturalmente con juegos populares de perfil
analogo. Las desviaciones se deben a que K-Means minimiza distancias
euclidianas en el espacio de caracteristicas, sin conocer la variable
'objetivo', mientras que las categorias de owners estan definidas por rangos
de popularidad que no forman clusters naturales separables en ese espacio.

PREGUNTA 2:
Describan detalladamente dos escenarios de aplicacion reales en el entorno
profesional o regional: uno donde sea estrictamente mandatorio aplicar KNN y
otro donde la unica solucion viable sea K-Means.

RESPUESTA 2:
a) Escenario donde es mandatorio aplicar KNN:
   Diagnostico medico asistido por computadora en hospitales regionales de
   Queretaro. Cuando un medico necesita clasificar un nuevo caso de diabetes
   basado en sintomas, edad, peso y niveles de glucosa, KNN es la unica opcion
   viable porque se requiere comparar directamente con pacientes historicos
   etiquetados (clases: diabetico/no diabetico). No existe un modelo
   preentrenado ni una distribucion conocida de los datos. KNN simplemente
   calcula la similitud con los K casos mas cercanos del historial medico y
   asigna la clase majoritaria. Es mandatorio cuando se necesita un metodo
   no parametrico, interpretable y que no asuma una forma funcional de los
   datos.

b) Escenario donde la unica solucion viable es K-Means:
   Segmentacion de mercados para una nueva empresa de servicios digitales
   en Latinoamerica que no tiene datos historicos etiquetados de clientes.
   No existe una variable objetivo predefinida (no sabemos cuantos ni cuales
   segmentos existen). K-Means permite descubrir grupos naturales de
   consumidores basados en patrones de consumo, ingresos y ubicacion
   geografica sin necesidad de etiquetas previas. Una vez identificados los
   segmentos, la empresa puede disenar estrategias de marketing dirigido para
   cada cluster. Es la unica opcion viable cuando no hay etiquetas disponibles
   y se busca descubrimiento de patrones ocultos en los datos.
=============================================================================
"""
