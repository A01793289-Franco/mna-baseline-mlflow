mna-baseline-mlflow
Experimento de MLOps utilizando MLflow Tracking y Random Forest sobre el dataset Titanic.

Crear ambiente virtual
py -m venv venv
Activar ambiente virtual
.\venv\Scripts\Activate.ps1
Instalar dependencias
pip install -r requirements.txt
Configuración de MLflow
Crear un archivo .env en la raíz del proyecto:

MLFLOW_TRACKING_URI=http://127.0.0.1:5000
Antes de ejecutar el entrenamiento es necesario iniciar el servidor de MLflow:

mlflow ui
La interfaz estará disponible en:

http://127.0.0.1:5000

Ejecutar entrenamiento
python src/train.py
Resultados
Los experimentos, métricas, parámetros y artefactos pueden visualizarse desde la interfaz de MLflow.

Estructura del proyecto
mna-baseline-mlflow/
│
├── data/
├── src/
│   └── train.py
├── notebooks/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
Tecnologías utilizadas
Python
Scikit-learn
Pandas
Matplotlib
Seaborn
MLflow
Git

Dependencias
Instalar las dependencias del proyecto:

pip install -r requirements.txt