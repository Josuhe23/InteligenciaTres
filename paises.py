import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Cargar el archivo CSV
data = pd.read_csv('student-mat.csv')

# Paso 1: Agregar la columna 'country' con valores numéricos
data['country'] = data['school'].apply(lambda x: 1 if x == 'GP' else 2)  # GP=1, MS=2

# Paso 2: Seleccionar las columnas necesarias para la clasificación
X = data[['G1', 'G2', 'G3']]  # Variables independientes
y = data['country']           # Variable dependiente (categórica)

# Paso 3: Normalizar las características
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Paso 4: Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Paso 5: Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Paso 6: Hacer predicciones
y_pred = model.predict(X_test)

# Paso 7: Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy}\n")

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# Reporte de Clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['GP (1)', 'MS (2)']))

# Paso 8: Representación gráfica

# Gráfico 1: Distribución de calificaciones
sns.pairplot(data[['G1', 'G2', 'G3']])
plt.suptitle('Distribución de calificaciones (G1, G2, G3)', y=1.02)
plt.savefig('grafico_distribucion_calificaciones.png')
plt.close()

# Gráfico 2: Relación entre calificaciones y país
sns.scatterplot(data=data, x='G1', y='G3', hue='country', palette='Set2')
plt.title('Relación entre G1 y G3 por país')
plt.xlabel('Nota G1')
plt.ylabel('Nota G3')
plt.savefig('grafico_relacion_calificaciones_pais.png')
plt.close()

# Gráfico 3: Matriz de Confusión
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['GP (1)', 'MS (2)'], yticklabels=['GP (1)', 'MS (2)'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.savefig('grafico_matriz_confusion.png')
plt.close()

# Imprimir confirmación de gráficos guardados
print("Los gráficos han sido guardados como archivos de imagen:")
print("1. grafico_distribucion_calificaciones.png")
print("2. grafico_relacion_calificaciones_pais.png")
print("3. grafico_matriz_confusion.png")
