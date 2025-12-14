import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_models():
    models = {}
    model_names = ['randomforest', 'svm', 'xgboost']
    versions = ['all', 'top']

    for model_name in model_names:
        models[model_name] = {}
        for version in versions:
            filepath = f"models/{model_name}_{version}.pkl"
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[model_name][version] = pickle.load(f)

    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('models/top_features.pkl', 'rb') as f:
        top_features = pickle.load(f)

    return models, label_encoder, top_features

@st.cache_data
def load_data():
    test_data = pd.read_csv('models/test_data.csv')
    feature_importance = pd.read_csv('models/feature_importance.csv')
    metrics_summary = pd.read_csv('models/metrics_summary.csv')
    
    # Cargar dataset completo para EDA
    full_data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if 'customerID' in full_data.columns:
        full_data = full_data.drop('customerID', axis=1)
    full_data['TotalCharges'] = pd.to_numeric(full_data['TotalCharges'], errors='coerce')
    
    return test_data, feature_importance, metrics_summary, full_data


def main():
    st.title("📊 Telco Customer Churn Prediction")
    st.markdown("**Sistema de Predicción de Abandono de Clientes con Ensemble Learning**")
    st.markdown("---")
    
    try:
        models, label_encoder, top_features = load_models()
        test_data, feature_importance, metrics_summary, full_data = load_data()
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        st.info("Por favor, asegúrate de entrenar los modelos primero ejecutando: python train_models.py")
        return

    # Definir mapping de modelos (usado en múltiples pestañas)
    model_mapping = {
        'Random Forest': 'randomforest',
        'SVM': 'svm',
        'XGBoost': 'xgboost'
    }

    tabs = st.tabs([
        "🎯 Predicción Individual",
        "📈 Dashboard de Métricas",
        "📊 Análisis Exploratorio (EDA)",
        "🔧 Entrenar Modelos"
    ])
    
    with tabs[0]:
        st.header("Predicción Individual")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Selección de Modelo")

            # Opción para usar modelo del repositorio o subir uno personalizado
            model_source = st.radio(
                "Fuente del Modelo:",
                ["Modelos del Repositorio", "Subir Modelo Personalizado (.pkl)"],
                key="model_source"
            )

            if model_source == "Subir Modelo Personalizado (.pkl)":
                uploaded_model = st.file_uploader(
                    "Sube tu modelo entrenado (.pkl)",
                    type=['pkl'],
                    key="custom_model_uploader",
                    help="Sube un pipeline de scikit-learn guardado con pickle"
                )

                if uploaded_model is not None:
                    try:
                        # Cargar el modelo personalizado en session_state
                        if 'custom_pipeline' not in st.session_state or st.session_state.get('last_uploaded_model') != uploaded_model.name:
                            custom_pipeline = pickle.load(uploaded_model)
                            st.session_state['custom_pipeline'] = custom_pipeline
                            st.session_state['last_uploaded_model'] = uploaded_model.name

                        selected_pipeline = st.session_state['custom_pipeline']
                        st.success(f"✅ Modelo '{uploaded_model.name}' cargado exitosamente")
                        st.info("**Modelo personalizado activo**\n\nUsando el modelo que subiste para predicciones")

                    except Exception as e:
                        st.error(f"❌ Error al cargar el modelo: {str(e)}")
                        st.info("Asegúrate de que el archivo sea un pipeline de scikit-learn válido")
                        selected_pipeline = None
                else:
                    st.warning("⚠️ Por favor sube un archivo .pkl para continuar")
                    selected_pipeline = None

            else:
                # Usar modelos del repositorio
                selected_model_name = st.selectbox(
                    "Modelo:",
                    list(model_mapping.keys())
                )

                version_mapping = {
                    'Todas las variables': 'all',
                    'Top features': 'top'
                }

                selected_version_name = st.radio(
                    "Versión:",
                    list(version_mapping.keys())
                )

                model_key = model_mapping[selected_model_name]
                version_key = version_mapping[selected_version_name]

                selected_pipeline = models[model_key][version_key]

                st.info(f"**Modelo seleccionado:** {selected_model_name}\n\n**Versión:** {selected_version_name}")

                # Determinar si es versión top para ajustar el formulario
                if 'version_key' not in locals():
                    version_key = None
            
        with col2:
            st.subheader("Datos del Cliente")

            # Solo mostrar formulario si hay un modelo seleccionado
            if selected_pipeline is None:
                st.warning("⚠️ Por favor selecciona o sube un modelo primero")
            else:
                # Determinar qué variables mostrar según la versión
                # Para modelos personalizados, mostrar todas las variables por defecto
                show_top_only = (model_source == "Modelos del Repositorio" and
                               'version_key' in locals() and version_key == 'top')

                if show_top_only:
                    st.info("💡 Ingresa solo las **Top Features** más importantes")

                    # Variables numéricas top
                    tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, value=12, key="tenure_top")
                    MonthlyCharges = st.number_input("Cargos Mensuales", min_value=0.0, max_value=200.0, value=50.0, key="monthly_top")
                    TotalCharges = st.number_input("Cargos Totales", min_value=0.0, max_value=10000.0, value=600.0, key="total_top")

                    # Variables categóricas top (necesitamos las originales para one-hot encoding)
                    Contract = st.selectbox("Contrato", ['Month-to-month', 'One year', 'Two year'], key="contract_top")
                    OnlineSecurity = st.selectbox("Seguridad Online", ['No', 'Yes', 'No internet service'], key="security_top")
                    PaymentMethod = st.selectbox("Método de Pago", [
                        'Electronic check', 'Mailed check',
                        'Bank transfer (automatic)', 'Credit card (automatic)'
                    ], key="payment_top")
                    TechSupport = st.selectbox("Soporte Técnico", ['No', 'Yes', 'No internet service'], key="tech_top")
                    InternetService = st.selectbox("Servicio Internet", ['DSL', 'Fiber optic', 'No'], key="internet_top")
                    OnlineBackup = st.selectbox("Backup Online", ['No', 'Yes', 'No internet service'], key="backup_top")

                    # Valores por defecto para las demás variables (necesarias para el pipeline)
                    gender = 'Female'
                    SeniorCitizen = 0
                    Partner = 'No'
                    Dependents = 'No'
                    PhoneService = 'Yes'
                    MultipleLines = 'No'
                    DeviceProtection = 'No'
                    StreamingTV = 'No'
                    StreamingMovies = 'No'
                    PaperlessBilling = 'Yes'

                else:
                    st.info("📝 Ingresa **todas las variables** del cliente")

                    col2_1, col2_2, col2_3 = st.columns(3)

                    with col2_1:
                        gender = st.selectbox("Género", ['Female', 'Male'])
                        SeniorCitizen = st.selectbox("Ciudadano Mayor", [0, 1])
                        Partner = st.selectbox("Pareja", ['Yes', 'No'])
                        Dependents = st.selectbox("Dependientes", ['Yes', 'No'])
                        tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, value=12)
                        PhoneService = st.selectbox("Servicio Telefónico", ['Yes', 'No'])
                        MultipleLines = st.selectbox("Múltiples Líneas", ['No', 'Yes', 'No phone service'])

                    with col2_2:
                        InternetService = st.selectbox("Servicio Internet", ['DSL', 'Fiber optic', 'No'])
                        OnlineSecurity = st.selectbox("Seguridad Online", ['No', 'Yes', 'No internet service'])
                        OnlineBackup = st.selectbox("Backup Online", ['No', 'Yes', 'No internet service'])
                        DeviceProtection = st.selectbox("Protección Dispositivos", ['No', 'Yes', 'No internet service'])
                        TechSupport = st.selectbox("Soporte Técnico", ['No', 'Yes', 'No internet service'])
                        StreamingTV = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
                        StreamingMovies = st.selectbox("Streaming Películas", ['No', 'Yes', 'No internet service'])

                    with col2_3:
                        Contract = st.selectbox("Contrato", ['Month-to-month', 'One year', 'Two year'])
                        PaperlessBilling = st.selectbox("Facturación Sin Papel", ['Yes', 'No'])
                        PaymentMethod = st.selectbox("Método de Pago", [
                            'Electronic check', 'Mailed check',
                            'Bank transfer (automatic)', 'Credit card (automatic)'
                        ])
                        MonthlyCharges = st.number_input("Cargos Mensuales", min_value=0.0, max_value=200.0, value=50.0)
                        TotalCharges = st.number_input("Cargos Totales", min_value=0.0, max_value=10000.0, value=600.0)

                if st.button("🔮 Realizar Predicción", type="primary"):
                    input_data = pd.DataFrame({
                        'gender': [gender],
                        'SeniorCitizen': [SeniorCitizen],
                        'Partner': [Partner],
                        'Dependents': [Dependents],
                        'tenure': [tenure],
                        'PhoneService': [PhoneService],
                        'MultipleLines': [MultipleLines],
                        'InternetService': [InternetService],
                        'OnlineSecurity': [OnlineSecurity],
                        'OnlineBackup': [OnlineBackup],
                        'DeviceProtection': [DeviceProtection],
                        'TechSupport': [TechSupport],
                        'StreamingTV': [StreamingTV],
                        'StreamingMovies': [StreamingMovies],
                        'Contract': [Contract],
                        'PaperlessBilling': [PaperlessBilling],
                        'PaymentMethod': [PaymentMethod],
                        'MonthlyCharges': [MonthlyCharges],
                        'TotalCharges': [TotalCharges]
                    })

                    prediction = selected_pipeline.predict(input_data)[0]
                    probability = selected_pipeline.predict_proba(input_data)[0]

                    churn_label = label_encoder.inverse_transform([prediction])[0]
                    churn_prob = probability[1]

                    st.markdown("---")
                    st.subheader("Resultado de la Predicción")

                    col_res1, col_res2 = st.columns(2)

                    with col_res1:
                        if churn_label == 'Yes':
                            st.error(f"### ⚠️ Clase Predicha: ABANDONARÁ")
                        else:
                            st.success(f"### ✅ Clase Predicha: NO ABANDONARÁ")

                    with col_res2:
                        st.metric("Probabilidad de Abandono", f"{churn_prob*100:.2f}%")

                    # Visualización con gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_prob*100,
                        title={'text': "Riesgo de Abandono (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if churn_prob > 0.5 else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.header("Dashboard de Métricas")

        # Filtro de versión
        version_filter = st.radio(
            "Filtrar por Versión:",
            ['Todas', 'Todas las variables', 'Top features'],
            key="version_filter",
            horizontal=True
        )

        filtered_metrics = metrics_summary.copy()

        if version_filter != 'Todas':
            version_map = {
                'Todas las variables': 'ALL',
                'Top features': 'TOP'
            }
            filtered_metrics = filtered_metrics[filtered_metrics['Version'] == version_map[version_filter]]

        st.markdown("---")
        st.subheader("Comparación de Métricas entre Modelos")

        metrics_to_plot = ['Test_Accuracy', 'Test_F1', 'Test_AUC']
        metrics_names = ['Accuracy', 'F1 Score', 'AUC']

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=metrics_names
        )

        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metrics_names), 1):
            fig.add_trace(
                go.Bar(
                    x=filtered_metrics['Model'] + ' - ' + filtered_metrics['Version'],
                    y=filtered_metrics[metric],
                    name=name,
                    text=filtered_metrics[metric].round(4),
                    textposition='auto',
                ),
                row=1, col=idx
            )

        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Comparación de Métricas de Performance"
        )

        fig.update_yaxes(range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tabla de Métricas")
        st.dataframe(
            filtered_metrics.style.highlight_max(axis=0, subset=['Test_Accuracy', 'Test_F1', 'Test_AUC']),
            use_container_width=True
        )
        
        st.markdown("---")
        
        st.subheader("Matrices de Confusión")
        
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        
        model_cols = [col_cm1, col_cm2, col_cm3]
        model_names_list = list(model_mapping.keys())
        
        for idx, (col, model_name) in enumerate(zip(model_cols, model_names_list)):
            with col:
                st.markdown(f"**{model_name}**")
                
                version_cm = st.radio(
                    "Versión:",
                    ['ALL', 'TOP'],
                    key=f"cm_version_{idx}"
                )
                
                model_key = model_mapping[model_name]
                version_key = 'all' if version_cm == 'ALL' else 'top'
                
                pipeline = models[model_key][version_key]
                
                X_test = test_data.drop('Churn', axis=1)
                y_test = test_data['Churn']
                
                y_pred = pipeline.predict(X_test)
                
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicho", y="Real", color="Cantidad"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    text_auto=True,
                    color_continuous_scale='Blues'
                )
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Importancia de Características")
        
        top_n_features = st.slider("Número de características a mostrar:", 5, 20, 10)
        
        fig = px.bar(
            feature_importance.head(top_n_features),
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n_features} Características Más Importantes',
            labels={'importance': 'Importancia', 'feature': 'Característica'}
        )
        
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.header("Análisis Exploratorio de Datos (EDA)")

        # Selector de tipo de variables
        var_type = st.radio(
            "Tipo de Variables:",
            ['Todas las variables', 'Top features'],
            key="eda_var_type",
            horizontal=True
        )

        st.markdown("---")
        st.subheader("Distribución de la Variable Objetivo")

        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            churn_counts = full_data['Churn'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title='Distribución de Churn',
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_eda2:
            fig = px.bar(
                x=churn_counts.index,
                y=churn_counts.values,
                title='Conteo de Churn',
                labels={'x': 'Churn', 'y': 'Cantidad'},
                color=churn_counts.index,
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Distribución de Variables Numéricas")

        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

        # Filtrar variables numéricas según el tipo seleccionado
        if var_type == 'Top features':
            numeric_cols_filtered = [col for col in numeric_cols if col in top_features]
        else:
            numeric_cols_filtered = numeric_cols

        selected_num_var = st.selectbox("Seleccionar variable numérica:", numeric_cols_filtered)
        
        col_num1, col_num2 = st.columns(2)
        
        with col_num1:
            fig = px.histogram(
                full_data,
                x=selected_num_var,
                color='Churn',
                title=f'Distribución de {selected_num_var} por Churn',
                marginal='box',
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_num2:
            fig = px.box(
                full_data,
                x='Churn',
                y=selected_num_var,
                title=f'Box Plot de {selected_num_var} por Churn',
                color='Churn',
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Distribución de Variables Categóricas")

        categorical_cols = [col for col in full_data.columns if full_data[col].dtype == 'object' and col != 'Churn']

        # Filtrar variables categóricas según el tipo seleccionado
        if var_type == 'Top features':
            # Las top features pueden incluir variables one-hot encoded (ej: Contract_Month-to-month)
            # Necesitamos extraer el nombre base de la variable
            categorical_cols_filtered = []
            for col in categorical_cols:
                # Verificar si la variable base o alguna de sus versiones one-hot está en top_features
                if col in top_features:
                    categorical_cols_filtered.append(col)
                else:
                    # Verificar si alguna versión one-hot de esta variable está en top_features
                    for tf in top_features:
                        if tf.startswith(col + '_'):
                            if col not in categorical_cols_filtered:
                                categorical_cols_filtered.append(col)
                            break
        else:
            categorical_cols_filtered = categorical_cols

        selected_cat_var = st.selectbox("Seleccionar variable categórica:", categorical_cols_filtered)
        
        cat_churn = full_data.groupby([selected_cat_var, 'Churn']).size().reset_index(name='count')
        
        fig = px.bar(
            cat_churn,
            x=selected_cat_var,
            y='count',
            color='Churn',
            title=f'Distribución de {selected_cat_var} por Churn',
            barmode='group',
            color_discrete_sequence=['#00CC96', '#EF553B']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Matriz de Correlación (Variables Numéricas)")
        
        numeric_data = full_data[numeric_cols + ['SeniorCitizen']].copy()
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Matriz de Correlación',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Estadísticas Descriptivas")

        st.dataframe(full_data.describe(), use_container_width=True)

    with tabs[3]:
        st.header("🔧 Entrenar Modelos")

        # Instrucciones de uso
        with st.expander("ℹ️ Cómo usar los modelos entrenados"):
            st.markdown("""
            ### 📖 Guía de Uso

            **Para uso local:**
            1. Entrena modelos ejecutando: `python train_models.py`
            2. Los modelos se guardan automáticamente en la carpeta `models/`
            3. La aplicación los carga automáticamente al iniciar

            **Para Streamlit Cloud:**
            1. Entrena los modelos localmente o usa la sección de abajo
            2. Descarga el archivo ZIP con todos los modelos
            3. Sube los archivos `.pkl` a tu repositorio de GitHub en la carpeta `models/`
            4. Haz commit y push de los cambios
            5. Streamlit Cloud detectará los cambios y recargará la app con los nuevos modelos

            **Modelos incluidos:**
            - Random Forest (versiones ALL y TOP)
            - SVM (versiones ALL y TOP)
            - XGBoost (versiones ALL y TOP)
            """)

        st.markdown("---")
        st.subheader("🎓 Entrenar Nuevos Modelos")
        st.markdown("Sube tu dataset en formato CSV para entrenar modelos desde cero")

        uploaded_file = st.file_uploader(
            "Arrastra y suelta tu archivo CSV aquí o haz clic para seleccionar",
            type=['csv'],
            help="El archivo debe tener la misma estructura que el dataset de Telco Churn",
            key="csv_uploader"
        )

        if uploaded_file is not None:
            try:
                # Leer el archivo CSV
                df_uploaded = pd.read_csv(uploaded_file)

                st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
                st.write(f"**Dimensiones:** {df_uploaded.shape[0]} filas × {df_uploaded.shape[1]} columnas")

                # Mostrar preview
                with st.expander("👀 Vista previa del dataset"):
                    st.dataframe(df_uploaded.head(10), use_container_width=True)

                # Configuración de entrenamiento
                st.markdown("---")
                st.subheader("⚙️ Configuración de Entrenamiento")

                col_config1, col_config2 = st.columns(2)

                with col_config1:
                    test_size = st.slider(
                        "Tamaño del conjunto de prueba (%)",
                        min_value=10,
                        max_value=40,
                        value=20,
                        step=5
                    ) / 100

                    random_state = st.number_input(
                        "Semilla aleatoria",
                        min_value=0,
                        max_value=9999,
                        value=42
                    )

                with col_config2:
                    models_to_train = st.multiselect(
                        "Modelos a entrenar:",
                        ['Random Forest', 'SVM', 'XGBoost'],
                        default=['Random Forest', 'SVM', 'XGBoost']
                    )

                    n_top_features = st.slider(
                        "Número de top features:",
                        min_value=5,
                        max_value=20,
                        value=10
                    )

                # Botón de entrenamiento
                if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
                    if len(models_to_train) == 0:
                        st.error("⚠️ Por favor selecciona al menos un modelo para entrenar")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            # Importar librerías necesarias
                            from sklearn.model_selection import train_test_split
                            from sklearn.preprocessing import LabelEncoder
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.pipeline import Pipeline
                            from sklearn.compose import ColumnTransformer
                            from sklearn.preprocessing import StandardScaler, OneHotEncoder
                            import os

                            status_text.text("🔄 Preparando datos...")
                            progress_bar.progress(10)

                            # Preparar datos
                            df_train = df_uploaded.copy()
                            if 'customerID' in df_train.columns:
                                df_train = df_train.drop('customerID', axis=1)

                            df_train['TotalCharges'] = pd.to_numeric(df_train['TotalCharges'], errors='coerce')
                            df_train = df_train.dropna()

                            # Separar características y target
                            X = df_train.drop('Churn', axis=1)
                            y = df_train['Churn']

                            # Codificar target
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y)

                            # Split train/test
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                            )

                            progress_bar.progress(20)
                            status_text.text("🔄 Identificando variables numéricas y categóricas...")

                            # Identificar variables
                            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            categorical_features = X.select_dtypes(include=['object']).columns.tolist()

                            # Crear transformadores
                            numeric_transformer = StandardScaler()
                            categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

                            preprocessor = ColumnTransformer(
                                transformers=[
                                    ('num', numeric_transformer, numeric_features),
                                    ('cat', categorical_transformer, categorical_features)
                                ]
                            )

                            progress_bar.progress(30)

                            # Guardar datos de prueba
                            test_data_save = pd.DataFrame(X_test, columns=X.columns)
                            test_data_save['Churn'] = y_test
                            test_data_save.to_csv('models/test_data.csv', index=False)

                            # Entrenar modelos
                            trained_models = {}
                            total_steps = len(models_to_train) * 2  # ALL y TOP para cada modelo
                            current_step = 0

                            for model_name in models_to_train:
                                status_text.text(f"🔄 Entrenando {model_name}...")

                                if model_name == 'Random Forest':
                                    from sklearn.ensemble import RandomForestClassifier
                                    model = RandomForestClassifier(
                                        n_estimators=100,
                                        random_state=random_state,
                                        n_jobs=-1,
                                        max_depth=15
                                    )
                                elif model_name == 'SVM':
                                    from sklearn.svm import SVC
                                    model = SVC(
                                        kernel='rbf',
                                        probability=True,
                                        random_state=random_state,
                                        C=1.0
                                    )
                                elif model_name == 'XGBoost':
                                    from xgboost import XGBClassifier
                                    model = XGBClassifier(
                                        n_estimators=100,
                                        random_state=random_state,
                                        max_depth=6,
                                        learning_rate=0.1,
                                        eval_metric='logloss'
                                    )

                                # Entrenar versión ALL
                                pipeline_all = Pipeline([
                                    ('preprocessor', preprocessor),
                                    ('classifier', model)
                                ])
                                pipeline_all.fit(X_train, y_train)

                                model_key = model_name.lower().replace(' ', '')
                                with open(f'models/{model_key}_all.pkl', 'wb') as f:
                                    pickle.dump(pipeline_all, f)

                                current_step += 1
                                progress_bar.progress(30 + int((current_step / total_steps) * 60))

                                # Obtener top features (simplificado usando feature importance de RF)
                                if model_name == 'Random Forest':
                                    # Obtener nombres de features después del preprocessing
                                    preprocessor.fit(X_train)
                                    feature_names = (numeric_features +
                                                   list(preprocessor.named_transformers_['cat']
                                                       .get_feature_names_out(categorical_features)))

                                    importances = pipeline_all.named_steps['classifier'].feature_importances_
                                    feature_importance_df = pd.DataFrame({
                                        'feature': feature_names,
                                        'importance': importances
                                    }).sort_values('importance', ascending=False)

                                    top_features_list = feature_importance_df.head(n_top_features)['feature'].tolist()

                                    # Guardar
                                    feature_importance_df.to_csv('models/feature_importance.csv', index=False)
                                    with open('models/top_features.pkl', 'wb') as f:
                                        pickle.dump(top_features_list, f)

                                current_step += 1
                                progress_bar.progress(30 + int((current_step / total_steps) * 60))

                            # Guardar label encoder
                            with open('models/label_encoder.pkl', 'wb') as f:
                                pickle.dump(le, f)

                            progress_bar.progress(100)
                            status_text.text("✅ Entrenamiento completado!")

                            st.success("🎉 ¡Modelos entrenados exitosamente!")
                            st.balloons()

                            # Preparar archivos para descarga
                            st.markdown("---")
                            st.subheader("📥 Descargar Modelos Entrenados")

                            col_dl1, col_dl2, col_dl3 = st.columns(3)

                            # Crear un archivo ZIP con todos los modelos
                            import zipfile
                            import io

                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Agregar todos los archivos .pkl del directorio models
                                for filename in os.listdir('models'):
                                    if filename.endswith('.pkl') or filename.endswith('.csv'):
                                        filepath = os.path.join('models', filename)
                                        zip_file.write(filepath, filename)

                            zip_buffer.seek(0)

                            with col_dl1:
                                st.download_button(
                                    label="📦 Descargar Todos los Modelos (ZIP)",
                                    data=zip_buffer.getvalue(),
                                    file_name="modelos_entrenados.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )

                            with col_dl2:
                                # Descargar label encoder
                                with open('models/label_encoder.pkl', 'rb') as f:
                                    st.download_button(
                                        label="🏷️ Label Encoder",
                                        data=f,
                                        file_name="label_encoder.pkl",
                                        mime="application/octet-stream",
                                        use_container_width=True
                                    )

                            with col_dl3:
                                # Descargar top features
                                with open('models/top_features.pkl', 'rb') as f:
                                    st.download_button(
                                        label="⭐ Top Features",
                                        data=f,
                                        file_name="top_features.pkl",
                                        mime="application/octet-stream",
                                        use_container_width=True
                                    )

                            st.info("""
                            💡 **Próximos pasos:**
                            - **Uso local:** Los modelos ya están en la carpeta `models/` y listos para usar
                            - **Streamlit Cloud:** Descarga el ZIP y sube los archivos a tu repositorio de GitHub en `models/`
                            """)

                        except Exception as e:
                            st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                            import traceback
                            with st.expander("Ver detalles del error"):
                                st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"❌ Error al cargar el archivo: {str(e)}")


if __name__ == "__main__":
    main()
