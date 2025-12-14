<div align="center">

# Customer Churn Prediction in Telecommunications

<img src="https://yachaytech.edu.ec/wp-content/uploads/2023/12/Logo-YT-Azul-Transparencia-220x103-1.png" alt="Yachay Tech Logo" width="300"/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/faustoguanoj/telco-customer-churn)
[![GitHub Profile](https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github)](https://github.com/faustoaguanor)

</div>

---

Sistema de predicción de abandono de clientes mediante técnicas de aprendizaje automático supervisado, desarrollado como proyecto final del curso de Aprendizaje de Máquina en la Maestría en Ciencia de Datos de la Universidad Yachay Tech.

## Tabla de Contenidos

- [Customer Churn Prediction in Telecommunications](#customer-churn-prediction-in-telecommunications)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Resumen](#resumen)
  - [Modelos Implementados](#modelos-implementados)
  - [Metodología](#metodología)
    - [1. Preprocesamiento de Datos](#1-preprocesamiento-de-datos)
    - [2. Selección de Características](#2-selección-de-características)
    - [3. Validación](#3-validación)
  - [Dataset](#dataset)
  - [Estructura del Proyecto](#estructura-del-proyecto)
  - [Instalación y Uso](#instalación-y-uso)
    - [Requisitos](#requisitos)
    - [Configuración del Entorno](#configuración-del-entorno)
  - [Deployment en Streamlit Cloud](#deployment-en-streamlit-cloud)
    - [Opción 1: Deployment con modelos pre-entrenados](#opción-1-deployment-con-modelos-pre-entrenados)
    - [Opción 2: Entrenar modelos desde la aplicación](#opción-2-entrenar-modelos-desde-la-aplicación)
  - [Aplicación Web Interactiva](#aplicación-web-interactiva)
    - [🎯 1. Predicción Individual](#-1-predicción-individual)
    - [📈 2. Dashboard de Métricas](#-2-dashboard-de-métricas)
    - [📊 3. Análisis Exploratorio de Datos (EDA)](#-3-análisis-exploratorio-de-datos-eda)
    - [🔧 4. Entrenamiento de Modelos](#-4-entrenamiento-de-modelos)
  - [Resultados](#resultados)
    - [Métricas de Desempeño](#métricas-de-desempeño)
    - [Comparativa de Modelos](#comparativa-de-modelos)
    - [Características Más Importantes](#características-más-importantes)
  - [Arquitectura de Pipelines](#arquitectura-de-pipelines)
  - [Hiperparámetros](#hiperparámetros)
  - [Tecnologías](#tecnologías)
  - [Autor](#autor)

## Resumen

Este trabajo implementa un sistema end-to-end de clasificación binaria para predecir el abandono de clientes (*customer churn*) en el sector de telecomunicaciones. Se evaluaron tres algoritmos de aprendizaje supervisado con arquitecturas complementarias: Random Forest (ensemble bagging), Support Vector Machines con kernel RBF, y XGBoost (gradient boosting). Cada modelo fue entrenado en dos configuraciones: con el conjunto completo de características y con las 10 características más relevantes identificadas mediante análisis de importancia.

## Modelos Implementados

1. **Random Forest**: Ensemble de 200 árboles de decisión con profundidad máxima de 15 niveles
2. **SVM (Support Vector Machine)**: Clasificador con kernel RBF y estimación probabilística habilitada
3. **XGBoost**: Gradient boosting con 200 estimadores y tasa de aprendizaje de 0.1

Cada algoritmo cuenta con dos variantes:

- **ALL**: Entrenamiento con las 19 características del dataset preprocesado
- **TOP**: Entrenamiento con las 10 características de mayor importancia predictiva

## Metodología

### 1. Preprocesamiento de Datos

**Pipeline de transformación**:

- Imputación de valores faltantes (mediana para variables numéricas, moda para categóricas)
- Escalado robusto mediante `RobustScaler` (resistente a valores atípicos)
- Codificación one-hot para variables categóricas
- Balanceo de clases mediante SMOTE (*Synthetic Minority Over-sampling Technique*)

### 2. Selección de Características

Se aplicó análisis de importancia mediante Random Forest para identificar las variables más predictivas. Las 10 características principales fueron utilizadas para entrenar las versiones optimizadas de cada modelo.

### 3. Validación

**Estrategia de partición estratificada**:

- Entrenamiento: 60%
- Validación: 20%
- Prueba: 20%

**Métricas de evaluación**:

- *Accuracy*: Proporción de clasificaciones correctas
- *F1-Score*: Media armónica entre precisión y recall
- *AUC-ROC*: Área bajo la curva característica de operación del receptor
- Matriz de confusión para análisis de errores

## Dataset

**Fuente**: Telco Customer Churn (IBM Sample Data Sets)

El conjunto de datos contiene 7,043 registros de clientes con 19 variables predictoras agrupadas en tres categorías:

**Variables demográficas**: género, edad (senior citizen), estado civil, dependientes

**Variables de servicio**: antigüedad (tenure), tipo de internet, servicios complementarios (seguridad, backup, soporte técnico, streaming)

**Variables contractuales**: tipo de contrato, método de pago, facturación mensual y total

**Variable objetivo**: Churn (abandono del servicio)

## Estructura del Proyecto

```
.
├── app.py                          # Aplicación web interactiva
├── train_models.py                 # Pipeline de entrenamiento
├── requirements.txt                # Dependencias del proyecto
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── models/                         # Artefactos generados
    ├── randomforest_all.pkl
    ├── randomforest_top.pkl
    ├── svm_all.pkl
    ├── svm_top.pkl
    ├── xgboost_all.pkl
    ├── xgboost_top.pkl
    ├── preparer.pkl                # Pipeline de preprocesamiento
    ├── top_features.pkl
    ├── label_encoder.pkl
    ├── feature_importance.csv
    ├── test_data.csv
    └── metrics_summary.csv
```

## Instalación y Uso

### Requisitos

- Python 3.8+
- Bibliotecas: scikit-learn, XGBoost, imbalanced-learn, Streamlit, Plotly, Pandas, NumPy

### Configuración del Entorno

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelos
python train_models.py

# Ejecutar aplicación web
streamlit run app.py
```

El script de entrenamiento genera 6 pipelines completos (3 algoritmos × 2 configuraciones) y guarda las métricas de evaluación. La aplicación web se ejecuta en `http://localhost:8501`.

## Deployment en Streamlit Cloud

### Opción 1: Deployment con modelos pre-entrenados

1. **Entrenar modelos localmente**:

   ```bash
   python train_models.py
   ```

2. **Subir al repositorio**:
   - Commit y push de todos los archivos `.pkl` y `.csv` de la carpeta `models/`
   - Asegurar que `requirements.txt` está actualizado

3. **Configurar en Streamlit Cloud**:
   - Acceder a [share.streamlit.io](https://share.streamlit.io)
   - Conectar repositorio de GitHub
   - Seleccionar `app.py` como archivo principal
   - Hacer clic en "Deploy"

### Opción 2: Entrenar modelos desde la aplicación

1. **Deployar aplicación sin modelos**:
   - Subir código a GitHub sin la carpeta `models/`
   - Deployar en Streamlit Cloud

2. **Entrenar en la interfaz web**:
   - Navegar a la pestaña "🔧 Entrenar Modelos"
   - Cargar el archivo CSV del dataset
   - Configurar parámetros de entrenamiento
   - Descargar el archivo ZIP con los modelos entrenados

3. **Actualizar el repositorio**:
   - Extraer los archivos del ZIP
   - Subir los archivos `.pkl` a la carpeta `models/` en GitHub
   - Streamlit Cloud redesplegará automáticamente

## Aplicación Web Interactiva

La interfaz de Streamlit proporciona cuatro módulos principales:

### 🎯 1. Predicción Individual

**Funcionalidad**: Sistema de inferencia en tiempo real para evaluación de riesgo de abandono.

**Características**:

- Selección de modelo (Random Forest, SVM, XGBoost) y configuración (ALL/TOP features)
- Formulario interactivo con validación de datos de entrada
- Ingreso de características demográficas, de servicio y contractuales
- Visualización de resultado: clase predicha (Abandonará/No Abandonará)
- Probabilidad de abandono con indicador visual de riesgo
- Interpretación automática del nivel de riesgo (Bajo/Medio/Alto)

**Uso**: Ideal para evaluación de clientes individuales y toma de decisiones de retención.

### 📈 2. Dashboard de Métricas

**Funcionalidad**: Panel de análisis comparativo de rendimiento de modelos.

**Características**:

- **Comparación de métricas**: Gráficos de barras comparativos para Accuracy, F1-Score y AUC-ROC
- **Filtros**: Visualización por versión (Todas/ALL features/TOP features)
- **Matrices de confusión**: Heatmaps interactivos para los 6 modelos entrenados
- **Análisis de errores**: Visualización de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos
- **Importancia de características**: Ranking de variables más influyentes con gráfico de barras horizontal
- **Selector dinámico**: Ajuste del número de características a visualizar (Top 5 a Top 20)

**Uso**: Evaluación y selección del mejor modelo según métricas de desempeño.

### 📊 3. Análisis Exploratorio de Datos (EDA)

**Funcionalidad**: Exploración estadística y visual del dataset.

**Características**:

- **Selector de alcance**: Visualización de todas las variables o solo top features
- **Distribución de Churn**:
  - Gráfico de torta con porcentajes
  - Gráfico de barras con conteos absolutos
  - Estadísticas de balance de clases
- **Variables numéricas**:
  - Histogramas comparativos por clase (Churn/No Churn)
  - Box plots para detección de outliers
  - Análisis de distribución por variable
- **Variables categóricas**:
  - Gráficos de barras agrupados
  - Comparación de frecuencias entre clases
  - Identificación de patrones discriminativos
- **Matriz de correlación**: Heatmap de correlaciones entre variables numéricas
- **Estadísticas descriptivas**: Tabla completa con medidas de tendencia central y dispersión

**Uso**: Comprensión del comportamiento de los datos y validación de supuestos del modelo.

### 🔧 4. Entrenamiento de Modelos

**Funcionalidad**: Reentrenamiento del sistema con datasets personalizados.

**Características**:

- **Carga de datos**: Upload de archivos CSV con vista previa
- **Configuración de partición**:
  - Tamaño del conjunto de prueba (10-40%)
  - Semilla aleatoria para reproducibilidad
- **Selección de algoritmos**: Checkbox para Random Forest, SVM y XGBoost
- **Configuración de features**: Número de características principales a seleccionar (5-20)
- **Proceso de entrenamiento**:
  - Barra de progreso por modelo
  - Métricas de validación en tiempo real
  - Resumen de características más importantes
- **Exportación**:
  - Descarga de archivo ZIP con todos los modelos entrenados
  - Descarga individual de label encoder y lista de top features
  - Métricas de evaluación en formato CSV

**Uso**: Adaptación del sistema a nuevos datos o actualización de modelos con información reciente.

## Resultados

El rendimiento de los modelos se evaluó sobre el conjunto de prueba (20% de los datos). Los resultados se encuentran disponibles en el archivo [`models/metrics_summary.csv`](models/metrics_summary.csv) generado durante el entrenamiento.

### Métricas de Desempeño

Los tres algoritmos demostraron capacidad de generalización adecuada, con métricas superiores a 0.75 en Accuracy y AUC-ROC. La configuración con todas las características (ALL) generalmente obtiene mejor rendimiento que la versión optimizada (TOP), aunque esta última ofrece la ventaja de requerir menos información del cliente para realizar predicciones.

### Comparativa de Modelos

Los resultados obtenidos sobre el conjunto de prueba son:

| Modelo | Versión | Accuracy | F1-Score | AUC-ROC |
|--------|---------|----------|----------|---------|
| **Random Forest** | ALL | 0.7665 | 0.5963 | **0.8242** |
| **Random Forest** | TOP | 0.7665 | 0.6079 | **0.8232** |
| **SVM** | ALL | 0.7544 | 0.6023 | 0.8094 |
| **SVM** | TOP | 0.7509 | **0.6147** | 0.8113 |
| **XGBoost** | ALL | **0.7700** | 0.5586 | 0.8021 |
| **XGBoost** | TOP | 0.7530 | 0.5639 | 0.8011 |

**Mejores resultados por métrica**:

- **Accuracy**: XGBoost ALL (77.00%)
- **F1-Score**: SVM TOP (61.47%)
- **AUC-ROC**: Random Forest ALL (82.42%)

**Análisis**:

- Random Forest demuestra el mejor equilibrio entre discriminación de clases (AUC-ROC más alto)
- XGBoost obtiene la mayor precisión general pero menor F1-Score
- SVM con características TOP logra el mejor balance precisión-recall (F1-Score más alto)
- Las versiones TOP mantienen desempeño competitivo con solo 10 características vs 19 completas

### Características Más Importantes

Las 10 características con mayor importancia predictiva (según Random Forest) son:

1. **TotalCharges**: Cargos totales acumulados
2. **MonthlyCharges**: Cargos mensuales
3. **tenure**: Antigüedad del cliente en meses
4. **Contract**: Tipo de contrato
5. **InternetService**: Tipo de servicio de internet
6. **PaymentMethod**: Método de pago
7. **TechSupport**: Soporte técnico contratado
8. **OnlineSecurity**: Servicio de seguridad online
9. **StreamingTV**: Servicio de streaming de TV
10. **PaperlessBilling**: Facturación electrónica

Estas características son utilizadas en las versiones TOP de los modelos, permitiendo predicciones con menor cantidad de información requerida.

## Arquitectura de Pipelines

Cada modelo implementa un pipeline de scikit-learn que encapsula:

1. **Preprocesamiento**: Imputación, escalado robusto y codificación one-hot
2. **Selección de características**: Filtrado automático para versiones TOP
3. **Clasificador**: Algoritmo de aprendizaje entrenado

Esta arquitectura garantiza la consistencia entre las fases de entrenamiento e inferencia, eliminando el riesgo de *data leakage* y simplificando el despliegue en producción.

## Hiperparámetros

| Modelo | Parámetros principales |
|--------|------------------------|
| Random Forest | n_estimators=200, max_depth=15, min_samples_split=10 |
| SVM | kernel='rbf', C=1.0, gamma='scale', probability=True |
| XGBoost | n_estimators=200, max_depth=6, learning_rate=0.1 |

## Tecnologías

**Lenguaje**: Python 3.8+

**Bibliotecas principales**: scikit-learn (pipelines, Random Forest, SVM), XGBoost, imbalanced-learn (SMOTE), Streamlit (interfaz web), Plotly (visualizaciones), Pandas, NumPy

## Autor

<div align="center">

**Fausto Guano**

Maestría en Ciencia de Datos
Universidad Yachay Tech

[![GitHub](https://img.shields.io/badge/GitHub-faustoaguanor-181717?style=flat-square&logo=github)](https://github.com/faustoaguanor)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:fausto.guano@yachaytech.edu.ec)

</div>

---

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<div align="center">

*Proyecto desarrollado con fines académicos para el curso de Aprendizaje de Máquina (2025)*

**Universidad Yachay Tech - Maestría en Ciencia de Datos**

</div>
