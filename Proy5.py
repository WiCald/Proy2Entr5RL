"""
Proyecto 2. Entrega 5: Modelos de Regresión Logística (Incisos 1 a 5 – Sin statsmodels)

Este script realiza:
1. La generación de una variable categórica basada en 'SalePrice' y la creación de tres variables dicotómicas:
   - EsCara: indica si la vivienda es cara (1) o no (0).
   - EsMedia: indica si la vivienda es de precio medio (1) o no (0).
   - EsEconomica: indica si la vivienda es económica (1) o no (0).
2. La división reproducible del dataset en conjuntos de entrenamiento y prueba.
3. La construcción de un modelo de Regresión Logística con validación cruzada para predecir si una vivienda es cara.
4. El análisis del modelo, evaluando la multicolinealidad mediante la matriz de correlación y mostrando los coeficientes del modelo.
5. La evaluación del modelo en el conjunto de prueba mediante matriz de confusión y reporte de clasificación.
6. Explique si hay sobreajuste (overfitting) o no (recuerde usar para esto los errores del conjunto de prueba
y de entrenamiento). Muestre las curvas de aprendizaje usando los errores de los conjuntos de
entrenamiento y prueba.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import cProfile
import io
import pstats
from functools import wraps

def aic_bic(model, X, y):
    n = X.shape[0]
    probas = model.predict_proba(X)
    ll = -log_loss(y, probas, normalize=False)
    k = X.shape[1] + 1  
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll
    return aic, bic
def main():
    # ---------------------
    # Inciso 1: Crear Variables Dicotómicas a partir de SalePrice
    # ---------------------
    try:
        df = pd.read_csv("train.csv")
    except Exception as e:
        print("Error al cargar 'train.csv':", e)
        return

    print("Columnas del dataset:", df.columns.tolist())

    # Utilizamos los cuantiles 33% y 66% para definir los umbrales
    quantiles = df['SalePrice'].quantile([0.33, 0.66])
    q_low = quantiles.iloc[0]
    q_high = quantiles.iloc[1]

    def categorizar_precio(x):
        if x >= q_high:
            return 'cara'
        elif x >= q_low:
            return 'media'
        else:
            return 'económica'

    df['Categoria'] = df['SalePrice'].apply(categorizar_precio)
    df['EsCara'] = (df['Categoria'] == 'cara').astype(int)
    df['EsMedia'] = (df['Categoria'] == 'media').astype(int)
    df['EsEconomica'] = (df['Categoria'] == 'económica').astype(int)

    print("\nPrimeras filas con SalePrice, Categoria y variables dicotómicas:")
    print(df[['SalePrice', 'Categoria', 'EsCara', 'EsMedia', 'EsEconomica']].head())

    # ---------------------
    # Inciso 2: División en Conjuntos de Entrenamiento y Prueba
    # ---------------------
    # Selección de algunas variables numéricas consideradas relevantes.
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        print("Las siguientes características no se encuentran en el dataset:", missing_features)
        return

    X = df[features]
    y = df['EsCara']  # Se modela la variable dicotómica para "vivienda cara"

    # División reproducible: 70% entrenamiento, 30% prueba (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("\nTamaño del conjunto de entrenamiento:", X_train.shape)
    print("Tamaño del conjunto de prueba:", X_test.shape)

    # ---------------------
    # Inciso 3: Modelo de Regresión Logística con Validación Cruzada
    # ---------------------
    modelo = LogisticRegression(max_iter=1000, solver='liblinear')
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
    print("\nExactitud promedio en validación cruzada:", np.mean(cv_scores))
    
    # Entrena el modelo con el conjunto de entrenamiento completo
    modelo.fit(X_train, y_train)

    # ---------------------
    # Inciso 4: Análisis del Modelo
    # ---------------------
    corr_matrix = X_train.corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title("Matriz de correlación de variables predictoras")
    plt.tight_layout()
    plt.show()

    coef_df = pd.DataFrame({
        'Variable': features,
        'Coeficiente': modelo.coef_[0]
    })
    print("\nCoeficientes del modelo:")
    print(coef_df)

    # ---------------------
    # Inciso 5: Evaluación del Modelo en el Conjunto de Prueba
    # ---------------------
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    print("Exactitud (accuracy) en el conjunto de prueba:", accuracy_score(y_test, y_pred))

    # ------------------------------
    # Inciso 6: Curva de aprendizaje
    # ------------------------------
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        modelo, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Entrenamiento")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validación cruzada")
    
    plt.title("Curvas aprendizaje")
    plt.xlabel("Tamano conjunto entrenamiento")
    plt.ylabel("Exactitud")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    # ------------------------------
    # Inciso 7: Tuneo curva de aprendizaje
    # ------------------------------
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])
    #busca parametro
    param_grid = {
    'model__penalty': ['l1', 'l2'],
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("\nMejores parámetros encontrados:")
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Exactitud:", accuracy_score(y_test, y_pred))
    # ------------------------------
    # Inciso 8: Matriz de confusion y rendimiento
    # ------------------------------
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusion (Tuneo):")
    print(cm)

    print("\n**ANÁLISIS DE RENDIMIENTO**")
    # Análisis detallado de errores
    tn, fp, fn, tp = cm.ravel()
    print("\nDetalle de errores:")
    print(f"- Falsos Positivos (FP): {fp} casos - Viviendas normales clasificadas como caras")
    print(f"- Falsos Negativos (FN): {fn} casos - Viviendas caras clasificadas como normales")
    print(f"\nTasa de error FP: {fp/(fp+tn):.2%}")
    print(f"Tasa de error FN: {fn/(fn+tp):.2%}")

    # Análisis de rendimiento con cProfile
    tn, fp, fn, tp = cm.ravel()
    print("\n**DETALLE DE ERRORES**")
    print(f"Falsos Positivos (FP): {fp} casos - Viviendas normales clasificadas como caras")
    print(f"Falsos Negativos (FN): {fn} casos - Viviendas caras clasificadas como normales")
    print(f"\nTasa de error FP: {fp/(fp+tn):.2%}")
    print(f"Tasa de error FN: {fn/(fn+tp):.2%}")

    # Función para análisis de rendimiento
    print("\n**ANÁLISIS DE RENDIMIENTO**")
    print("=== Resultados del perfilado ===")
    def profile(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            pr.disable()
            
            print(f"\nTiempo de ejecución: {end_time - start_time:.4f} segundos")
            
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(10)
            print(s.getvalue())
            
            return result
        return wrapper

    # Función a analizar
    @profile
    def train_and_predict():
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l2', C=1, solver='liblinear', max_iter=1000))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    train_and_predict()

    # ------------------------------
    # Inciso 9: Mejor modelo
    # ------------------------------
    aic_bin, bic_bin = aic_bic(best_model.named_steps['model'], X_test, y_test)
    print("\nMetricas estadisticas:")
    print(f"AIC: {aic_bin:.2f} | BIC: {bic_bin:.2f}")
    print("(Valores mas bajos indican mejor modelo)")
    #matriz
    y_pred_bin = best_model.predict(X_test)
    cm_bin = confusion_matrix(y_test, y_pred_bin)
    tn, fp, fn, tp = cm_bin.ravel()
    
    print("\nAnálisis de errores:")
    print(f"FP (Sobrestimación): {fp} casos ({fp/(fp+tn):.2%})")
    print(f"FN (Subestimación): {fn} casos ({fn/(fn+tp):.2%})")

    # ------------------------------
    # Inciso 10: Regresion para variable de precios
    # ------------------------------
    X_multi = df[features]
    y_multi = LabelEncoder().fit_transform(df['Categoria'])  # 0:barata, 1:media, 2:cara
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y_multi, test_size=0.3, random_state=42)
    
    # Pipeline y parametros
    multi_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])
    param_grid = {
        'model__estimator__C': [0.1, 1, 10],
        'model__estimator__penalty': ['l2'],
        'model__estimator__solver': ['liblinear', 'saga']
    }
    print("\nTUNEADO MULTICASE")
    grid_multi = GridSearchCV(multi_pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_multi.fit(X_train_m, y_train_m)
    
    best_multi = grid_multi.best_estimator_
    print("\nMejores parametros:", grid_multi.best_params_)

    y_pred_multi = best_multi.predict(X_test_m)
    print("\nReporte clasificacion:")
    print(classification_report(y_test_m, y_pred_multi, 
                              target_names=['Económica', 'Media', 'Cara']))

    # ------------------------------
    # Inciso 10: Comparación de eficiencia de modelos
    # ------------------------------
    @profile
    def evaluate_model(model, X_train, y_train, X_test):
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start
        
        return {
            'Tiempo Entrenamiento': train_time,
            'Tiempo Predicción': pred_time,
            'Exactitud': accuracy_score(y_test if 'y_test' in locals() else y_test_m, y_pred)
        }
    
    #Ev y comparacion
    #binario
    res_bin = evaluate_model(best_model, X_train, y_train, X_test)
    #multicase
    res_multi = evaluate_model(best_multi, X_train_m, y_train_m, X_test_m)
    
    comparison = pd.DataFrame({
        'Binario': res_bin,
        'Multiclase': res_multi
    }).T
    print("\nTabla comparativa:")
    print(comparison)
    
    print("\n*** CONCLUSIONES ***")
    print("Rendimiento computacional:")
    print("  - Tiempo (mas lento):")
    print(f"   Entrenamiento: {res_multi['Tiempo Entrenamiento']/res_bin['Tiempo Entrenamiento']:.1f}x")
    print(f"   Prediccion:    {res_multi['Tiempo Predicción']/res_bin['Tiempo Predicción']:.1f}x")
    
    print("\nPrecisión:")
    print(f"   Binario: {res_bin['Exactitud']:.2%} de exactitud")
    print(f"   Multiclase: {res_multi['Exactitud']:.2%} de exactitud")


if __name__ == "__main__":
    main()