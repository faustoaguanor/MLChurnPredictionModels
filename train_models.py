import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from transformers import DataFramePreparer, FeatureSelector


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    return (train_set, val_set, test_set)


def main():
    print("Cargando datos...")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Eliminar customerID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convertir TotalCharges a numérico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Codificar target
    le = LabelEncoder()
    df["Churn"] = le.fit_transform(df["Churn"])

    print("\nDivisión de datos...")
    train_set, val_set, test_set = train_val_test_split(df, stratify="Churn")

    X_train = train_set.drop("Churn", axis=1)
    y_train = train_set["Churn"].copy()
    X_val = val_set.drop("Churn", axis=1)
    y_val = val_set["Churn"].copy()
    X_test = test_set.drop("Churn", axis=1)
    y_test = test_set["Churn"].copy()

    # Ajustar preprocesador
    print("\nAjustando preprocesador...")
    preparer = DataFramePreparer()
    preparer.fit(X_train)

    X_train_prep = preparer.transform(X_train)
    X_val_prep = preparer.transform(X_val)
    X_test_prep = preparer.transform(X_test)

    # Aplicar SMOTE
    print("\nAplicando SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_prep, y_train)

    # Obtener top features usando Random Forest
    print("\nIdentificando top features...")
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train_sm, y_train_sm)

    feature_importance = pd.DataFrame(
        {"feature": X_train_prep.columns, "importance": rf_temp.feature_importances_}
    ).sort_values("importance", ascending=False)

    top_n = 10
    top_features = feature_importance.head(top_n)["feature"].tolist()

    print(f"\nTop {top_n} features:")
    for i, feat in enumerate(top_features, 1):
        print(f"{i}. {feat}")

    # Configuraciones de modelos
    models_config = {
        "RandomForest": {
            "all": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            ),
            "top": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            ),
        },
        "SVM": {
            "all": SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42
            ),
            "top": SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42
            ),
        },
        "XGBoost": {
            "all": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "top": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        },
    }

    results = {}

    # Entrenar modelos
    for model_name, versions in models_config.items():
        print(f"Entrenando {model_name}")

        results[model_name] = {}

        for version_name, model in versions.items():
            print(f"\nVersión: {version_name.upper()} FEATURES")

            if version_name == "top":
                X_train_version = X_train_sm[top_features]
                X_val_version = X_val_prep[top_features]
                X_test_version = X_test_prep[top_features]
            else:
                X_train_version = X_train_sm
                X_val_version = X_val_prep
                X_test_version = X_test_prep

            # Entrenar
            print("Entrenando modelo...")
            model.fit(X_train_version, y_train_sm)

            # Predecir
            y_pred_val = model.predict(X_val_version)
            y_pred_proba_val = model.predict_proba(X_val_version)[:, 1]

            y_pred_test = model.predict(X_test_version)
            y_pred_proba_test = model.predict_proba(X_test_version)[:, 1]

            # Métricas
            metrics = {
                "val": {
                    "accuracy": accuracy_score(y_val, y_pred_val),
                    "f1": f1_score(y_val, y_pred_val),
                    "auc": roc_auc_score(y_val, y_pred_proba_val),
                    "confusion_matrix": confusion_matrix(y_val, y_pred_val),
                },
                "test": {
                    "accuracy": accuracy_score(y_test, y_pred_test),
                    "f1": f1_score(y_test, y_pred_test),
                    "auc": roc_auc_score(y_test, y_pred_proba_test),
                    "confusion_matrix": confusion_matrix(y_test, y_pred_test),
                },
            }

            print(f"\nMétricas de Validación:")
            print(f"  Accuracy: {metrics['val']['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['val']['f1']:.4f}")
            print(f"  AUC: {metrics['val']['auc']:.4f}")

            print(f"\nMétricas de Test:")
            print(f"  Accuracy: {metrics['test']['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['test']['f1']:.4f}")
            print(f"  AUC: {metrics['test']['auc']:.4f}")

            # Crear pipeline completo
            if version_name == "top":
                full_pipeline = Pipeline(
                    [
                        ("preparer", preparer),
                        ("selector", FeatureSelector(top_features)),
                        ("model", model),
                    ]
                )
            else:
                full_pipeline = Pipeline([("preparer", preparer), ("model", model)])

            # Guardar pipeline
            filename = f"models/{model_name.lower()}_{version_name}.pkl"
            print(f"\nGuardando pipeline en: {filename}")
            with open(filename, "wb") as f:
                pickle.dump(full_pipeline, f)

            results[model_name][version_name] = {
                "metrics": metrics,
                "model": model,
                "pipeline": full_pipeline,
            }

    # Guardar preparer, top_features y label_encoder
    print("\n\nGuardando componentes adicionales...")
    with open("models/preparer.pkl", "wb") as f:
        pickle.dump(preparer, f)

    with open("models/top_features.pkl", "wb") as f:
        pickle.dump(top_features, f)

    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Guardar feature importance
    feature_importance.to_csv("models/feature_importance.csv", index=False)

    # Guardar datos de test
    test_set.to_csv("models/test_data.csv", index=False)

    # Guardar resumen de métricas
    metrics_summary = []
    for model_name in results:
        for version in ["all", "top"]:
            metrics_summary.append(
                {
                    "Model": model_name,
                    "Version": version.upper(),
                    "Test_Accuracy": results[model_name][version]["metrics"]["test"][
                        "accuracy"
                    ],
                    "Test_F1": results[model_name][version]["metrics"]["test"]["f1"],
                    "Test_AUC": results[model_name][version]["metrics"]["test"]["auc"],
                }
            )

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv("models/metrics_summary.csv", index=False)

    print("RESUMEN FINAL DE MÉTRICAS (TEST SET)")
    print(metrics_df.to_string(index=False))
    print("\nEntrenamiento completado exitosamente!")


if __name__ == "__main__":
    import os

    os.makedirs("models", exist_ok=True)
    main()
