"""
Proyecto 2. Entrega 5: Modelos de Regresión Logística
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

if __name__ == "__main__":
    main()
