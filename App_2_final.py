import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plot_acf

# Título de la aplicación
st.title("Análisis de Datos y Modelado Predictivo")

# Paso 1: Elegir si el usuario quiere hacer EDA o Modelado
option = st.radio("¿Qué te gustaría hacer?", ("Análisis Exploratorio de Datos (EDA)", "Modelado Predictivo"))

# Subida de archivo CSV o Excel
uploaded_file = st.file_uploader("Sube tus datos (CSV o Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Leer el archivo dependiendo de su formato
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    st.write("Datos cargados:")
    st.write(df.head())

    # Opción 1: Análisis Exploratorio de Datos (EDA)
    if option == "Análisis Exploratorio de Datos (EDA)":
        st.write("### Análisis Exploratorio de Datos")

        st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        st.write(df.describe())
        st.write(df.isnull().sum())

        # Selección de variables para gráficos
        column = st.selectbox("Selecciona una columna para análisis gráfico", df.columns)

        # Histograma
        st.write(f"Histograma de la columna {column}:")
        st.bar_chart(df[column])

        # Coeficiente de variación
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].mean() != 0:
                coef_var = df[column].std() / df[column].mean()
                st.write(f"Coeficiente de variación de la columna {column}: {coef_var:.4f}")
            else:
                st.write("El coeficiente de variación no se puede calcular (media = 0).")
        else:
            st.write(f"La columna seleccionada '{column}' no es numérica. No se puede calcular el coeficiente de variación.")

        # Gráfico de densidad
        if st.checkbox("Mostrar gráfico de densidad"):
            fig, ax = plt.subplots()
            sns.kdeplot(df[column], ax=ax, fill=True)
            st.pyplot(fig)


        # Boxplot para distribución y outliers
        if st.checkbox("Mostrar Boxplot"):
            fig, ax = plt.subplots()
            sns.boxplot(data=df[column], ax=ax)
            st.pyplot(fig)


        # Matriz de correlación y heatmap
        if st.checkbox("Mostrar matriz de correlación"):
            corr_matrix = df.corr()
            st.write(corr_matrix)
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        # Autocorrelación (solo si es una serie temporal o numérica)
        if st.checkbox("Mostrar autocorrelación"):
            fig, ax = plt.subplots()
            plot_acf(df[column].dropna(), ax=ax)
            st.pyplot(fig)

        # Gráfico de dispersión interactivo con Plotly
        if st.checkbox("Mostrar gráfico interactivo"):
            x_axis = st.selectbox("Selecciona el eje X", df.columns)
            y_axis = st.selectbox("Selecciona el eje Y", df.columns)

            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Gráfico interactivo de {x_axis} vs {y_axis}")
            st.plotly_chart(fig)

    # Opción 2: Modelado Predictivo
    elif option == "Modelado Predictivo":
        st.write("### Modelado Predictivo")

        # Permitir al usuario identificar la columna de fechas si existe
        date_column = st.selectbox("Selecciona la columna de fechas (si aplica)", [None] + df.columns.tolist())
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            st.write(f"Columna de fecha seleccionada: {date_column}")

        # Permitir al usuario seleccionar la variable dependiente (objetivo)
        target = st.selectbox("Selecciona la variable objetivo (dependiente)", df.columns)

        # Selección de variables independientes
        features = st.multiselect("Selecciona las variables independientes (predictoras)", df.columns.drop([target]))

        if date_column and date_column in features:
            features.remove(date_column)

        df_cleaned = df.dropna(subset=features + [target])
        if len(df) != len(df_cleaned):
            st.warning(f"Se han eliminado {len(df) - len(df_cleaned)} filas que contenían valores faltantes.")

        X = df_cleaned[features]
        y = df_cleaned[target]

        # Tomar la última observación como dato de prueba
        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        X_test = X.iloc[-1:]
        y_test = y.iloc[-1]
        with st.expander("Mostrar/Ocultar Entrenamiento y Resultados"):  
            st.write("Modelos de Predicción Tradicionales")

            # Diccionario para almacenar predicciones y modelos
            predictions_dict = {}

            # Entrenamiento y predicción de los modelos tradicionales
            model_lr = LinearRegression()
            model_lr.fit(X_train, y_train)
            prediction_lr = model_lr.predict(X_test)[0]
            predictions_dict["Regresión Lineal"] = prediction_lr

            model_dt = DecisionTreeRegressor()
            model_dt.fit(X_train, y_train)
            prediction_dt = model_dt.predict(X_test)[0]
            predictions_dict["Árbol de Decisión"] = prediction_dt

            model_rf = RandomForestRegressor()
            model_rf.fit(X_train, y_train)
            prediction_rf = model_rf.predict(X_test)[0]
            predictions_dict["Bosque Aleatorio"] = prediction_rf

            model_knn = KNeighborsRegressor(n_neighbors=5)
            model_knn.fit(X_train, y_train)
            prediction_knn = model_knn.predict(X_test)[0]
            predictions_dict["K-Nearest Neighbors"] = prediction_knn

            poly = PolynomialFeatures(degree=2)
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)
            model_poly = LinearRegression()
            model_poly.fit(X_poly_train, y_train)
            prediction_poly = model_poly.predict(X_poly_test)[0]
            predictions_dict["Regresión Polinómica"] = prediction_poly

            # Mostrar resultados de los modelos tradicionales
            st.write("### Resultados de Modelos Tradicionales")
            for model_name, prediction in predictions_dict.items():
                st.write(f"**{model_name}:**")
                st.write(f"Valor Real: {y_test}")
                st.write(f"Predicción: {prediction}")

            # Modelos de series temporales (ARIMA, ETS, SARIMA)
            if date_column:
                st.write("### Modelos de Series Temporales")

                # ARIMA - entrenar sin el último valor
                model_arima = ARIMA(y_train, order=(1, 1, 2))
                model_fit_arima = model_arima.fit()
                forecast_arima = model_fit_arima.forecast(steps=1)
                prediction_arima = forecast_arima.values[0]  # Usar .values[0] para acceder al valor
                predictions_dict["ARIMA"] = prediction_arima

                # ETS - entrenar sin el último valor
                model_ets = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=12)
                model_fit_ets = model_ets.fit()
                forecast_ets = model_fit_ets.forecast(steps=1)
                prediction_ets = forecast_ets.values[0]  # Usar .values[0] para acceder al valor
                predictions_dict["ETS"] = prediction_ets

                # SARIMA - entrenar sin el último valor
                model_sarima = SARIMAX(y_train, order=(1, 1, 2), seasonal_order=(1, 1, 1, 12))
                model_fit_sarima = model_sarima.fit()
                forecast_sarima = model_fit_sarima.forecast(steps=1)
                prediction_sarima = forecast_sarima.values[0]  # Usar .values[0] para acceder al valor
                predictions_dict["SARIMA"] = prediction_sarima

                # Mostrar resultados de las series temporales
                st.write("### Resultados de Modelos de Series Temporales")
                for model_name in ["ARIMA", "ETS", "SARIMA"]:
                    st.write(f"**{model_name}:**")
                    st.write(f"Predicción: {predictions_dict[model_name]}")

            # Comparación de todos los modelos
            st.write("### Comparación de Dato Real vs Dato Estimado")
            for model_name, prediction in predictions_dict.items():
                st.write(f"**{model_name}:**")
                st.write(f"Valor Real: {y_test}")
                st.write(f"Predicción: {prediction}")

                # Calcular errores y porcentaje de diferencia para todos los modelos
                mae = mean_absolute_error([y_test], [prediction])
                mse = mean_squared_error([y_test], [prediction])
                diff_percent = abs((y_test - prediction) / y_test) * 100

                # Mostrar los errores y porcentaje de diferencia para todos los modelos
                st.write(f"Error Absoluto Medio (MAE): {mae:.4f}")
                st.write(f"Error Cuadrático Medio (MSE): {mse:.4f}")
                st.write(f"Porcentaje de Diferencia: {diff_percent:.2f}%")
                st.write("---")

        # Gráfico comparativo
        st.write("### Comparación Gráfica de las Predicciones con el Valor Real")

        # Crear un dataframe para gráficos
        df_graph = pd.DataFrame({
            'Modelo': list(predictions_dict.keys()),
            'Predicciones': list(predictions_dict.values()),
            'Valor Real': [y_test] * len(predictions_dict)
        })

        # Gráfico de barras para comparar las predicciones con el valor real
        st.bar_chart(df_graph.set_index('Modelo'))

        # Gráfico de dispersión interactivo usando Plotly
        fig = px.scatter(df_graph, x='Modelo', y='Predicciones', labels={'Predicciones':'Predicción'},
                         title="Predicciones de cada Modelo vs Valor Real")
        fig.add_scatter(x=df_graph['Modelo'], y=df_graph['Valor Real'], mode='lines+markers', name='Valor Real')
        st.plotly_chart(fig)

        # Gráfico de líneas usando Matplotlib para análisis más visual
        st.write("### Gráfico de Líneas - Comparación de Predicciones")
        plt.figure(figsize=(10, 5))
        plt.plot(df_graph['Modelo'], df_graph['Predicciones'], label='Predicciones', marker='o')
        plt.axhline(y=y_test, color='r', linestyle='--', label='Valor Real')
        plt.title('Comparación de Predicciones por Modelo')
        plt.xlabel('Modelo')
        plt.ylabel('Valor Predicho')
        plt.legend()
        st.pyplot(plt)

        # Sección de predicción futura con datos nuevos
        st.write("### Predicción de un valor no conocido")
        new_data = []
        for feature in features:
            value = st.number_input(f"Introduce un valor para {feature}")
            new_data.append(value)

        # Pregunta al usuario por la fecha si es una serie temporal
        if date_column:
            future_date = st.date_input("Selecciona una fecha futura para predecir")

        new_data = np.array(new_data).reshape(1, -1)

        if st.button("Predecir con nuevos valores"):
            for model_name, model in zip(predictions_dict.keys(), [model_lr, model_dt, model_rf, model_knn, model_poly]):
                if model_name == "Regresión Polinómica":
                    new_data_poly = poly.transform(new_data)
                    prediction = model.predict(new_data_poly)[0]
                else:
                    prediction = model.predict(new_data)[0]

                st.write(f"Predicción con {model_name}: {prediction}")

            # Predicciones para los modelos de series temporales con la nueva fecha
            if date_column:
                st.write("### Predicciones de Series Temporales con la Nueva Fecha")

                # ARIMA para la nueva fecha
                model_arima_full = ARIMA(df[target], order=(1, 1, 2))
                model_fit_arima_full = model_arima_full.fit()
                forecast_arima_future = model_fit_arima_full.forecast(steps=1)
                st.write(f"ARIMA: Predicción para {future_date}: {forecast_arima_future.iloc[0]}")

                # ETS para la nueva fecha
                model_ets_full = ExponentialSmoothing(df[target], trend="add", seasonal="add", seasonal_periods=12)
                model_fit_ets_full = model_ets_full.fit()
                forecast_ets_future = model_fit_ets_full.forecast(steps=1)
                st.write(f"ETS: Predicción para {future_date}: {forecast_ets_future.iloc[0]}")

                # SARIMA para la nueva fecha
                model_sarima_full = SARIMAX(df[target], order=(1, 1, 2), seasonal_order=(1, 1, 1, 12))
                model_fit_sarima_full = model_sarima_full.fit()
                forecast_sarima_future = model_fit_sarima_full.forecast(steps=1)
                st.write(f"SARIMA: Predicción para {future_date}: {forecast_sarima_future.iloc[0]}")

