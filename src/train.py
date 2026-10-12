#==================================================================================
#LIBRERIAS
#==================================================================================


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os


#==================================================================================
#CONFIGURACION DE MLFLOW
#==================================================================================

#Carga variables de entorno definidas en el archivo .env
load_dotenv()
#Configurar la dirección del servidor de seguimiento de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Crear o utilizar el experimento donde se registrarán las corridas
mlflow.set_experiment("Titanic-RandomForest")


#==================================================================================
#CARGA DE DATOS
#==================================================================================

#Titanic dataset
df = pd.read_csv("data/train.csv")


#==================================================================================
#PREPARACION DE DATOS
#==================================================================================

# Creacion de grupos com relacion a 'Age', 'Sex', 'Pclass' para relacionar a personas similares y poder rellenar los valores faltantes en 'Age''
df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

# Transformacion de 'Cabin' en variable binaria para saber si el pasajero tiene cabina (1) o si no tiene cabina (0), "tener cabina mayor status"
df['Cabin_known'] = df['Cabin'].notnull().astype(int)

# Se relleno los valores faltantes en la columna 'Embarked' por S (Southampton) ya que este puerto era el mas popular.
df['Embarked'] = df['Embarked'].fillna('S')


#==================================================================================
#PREPARACION DE VARIABLES
#==================================================================================

#Copia del data frame limpio.
df_model = df.copy()

#One-hot encoding para variables categoricas.
df_model = pd.get_dummies(df_model, columns=['Sex', 'Embarked'], drop_first=True)

# Variables predictoras
X = df_model[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin_known',
              'Sex_male', 'Embarked_Q', 'Embarked_S']]

#Variable objetivo
y = df_model['Survived']


#==================================================================================
#DIVISION DE DATOS
#==================================================================================

#Dividir en train/test
mlflow.sklearn.autolog()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#==================================================================================
#ENTRENAMIENTO DEL MODELO
#==================================================================================

# Iniciar una nueva corrida en MLflow
mlflow.start_run(run_name="RandomForest")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Entrenar el modelo utilizando el conjunto de entrenamiento
model.fit(X_train, y_train)


#==================================================================================
#PREDICCION
#==================================================================================

# Generar predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)


#==================================================================================
#EVALUACION DEL MODELO
#==================================================================================

# Calcular métricas de desempeño
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
mlflow.end_run()


#==================================================================================
#MATRIZ DE CONFUSION Y REPORTE DE CLASIFICACION
#==================================================================================

# Generar matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Mostrar métricas detalladas por clase
print(classification_report(y_test, y_pred))

print("Entrenamiento completado correctamente.")