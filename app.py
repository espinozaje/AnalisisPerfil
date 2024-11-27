from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import chardet
import base64
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

app = Flask(__name__)

# Función para cargar y preprocesar el dataset
def load_and_preprocess(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(100000)
        encoding = chardet.detect(raw_data)['encoding']

    for enc in [encoding, 'utf-8-sig', 'latin1', 'iso-8859-1']:
        try:
            data = pd.read_csv(file_path, encoding=enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        raise ValueError("No se pudo leer el archivo con ninguna codificación compatible.")

    data.replace(["", "N/A", "null", "None", " "], pd.NA, inplace=True)
    return data

# Función para calcular estadísticas de calidad
def calculate_quality_stats(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    mean_values = {}
    std_values = {}

    for col in numeric_columns:
        try:
            mean_values[col] = round(data[col].mean(), 2)
            std_values[col] = round(data[col].std(), 2)
        except Exception:
            mean_values[col] = 'N/A'
            std_values[col] = 'N/A'

    missing_values = data.isnull().sum().sum()
    missing_percentage = (missing_values / data.size) * 100
    duplicate_rows = data.duplicated().sum()
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns.tolist()

    return {
        "Valores Faltantes Totales": missing_values,
        "Porcentaje de Valores Faltantes": f"{missing_percentage:.2f}%",
        "Filas Duplicadas": duplicate_rows,
        "Columnas No Numéricas": non_numeric_columns,
        "Media de Columnas Numéricas": mean_values,
        "Desviación Estándar de Columnas Numéricas": std_values,
    }

# Función para entrenar el modelo Random Forest
def train_random_forest_model(data):
    # Seleccionar las columnas características (features)
    features = ['repositories', 'stars', 'followers', 'followings', 'num_experiences', 
                'num_educations', 'num_certifications', 'num_recommendations', 'has_summary', 'has_photo']

    data = data.dropna(subset=features + ['quality_score'])
    
    X = data[features]  # Características (features)
    y = data['quality_score']  # Objetivo (target)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Error absoluto medio: {mae:.2f}")

    # Guardar el modelo entrenado en un archivo
    joblib.dump(model, 'random_forest_model.pkl')

    return model

# Función para cargar el modelo entrenado y hacer predicciones
def predict_with_random_forest(data):
    # Cargar el modelo guardado
    model = joblib.load('random_forest_model.pkl')

    # Seleccionar las mismas columnas características
    features = ['repositories', 'stars', 'followers', 'followings', 'num_experiences', 
                'num_educations', 'num_certifications', 'num_recommendations', 'has_summary', 'has_photo']
    
    # Preprocesar los datos (asegurarse de que no haya valores nulos en las características)
    data = data.dropna(subset=features)
    
    # Predecir los scores de calidad
    predictions = model.predict(data[features])

    # Agregar las predicciones al DataFrame
    data['predicted_quality_score'] = predictions
    
    # Clasificar las predicciones
    data['predicted_category'] = data['predicted_quality_score'].apply(lambda score: "Alta calidad" if score >= 50 else "Media calidad" if score >= 30 else "Baja calidad")
    
    return data

# Función para calcular valores atípicos
def calculate_outliers(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    outlier_summary = {}

    for col in numeric_columns:
        try:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            outlier_summary[col] = len(outliers)
        except Exception:
            outlier_summary[col] = 'N/A'

    return outlier_summary

# Función para procesar columnas específicas
def process_columns(data):
    def parse_number(value):
        try:
            if isinstance(value, str):
                value = value.strip().lower()
                if value.isdigit():
                    return int(value)
                if 'k' in value:
                    return int(float(value.replace('k', '')) * 1000)
                if 'm' in value:
                    return int(float(value.replace('m', '')) * 1e6)
                return int(float(value))
            return int(value) if isinstance(value, (int, float)) else 0
        except ValueError:
            return 0

    columns_to_parse = ['repositories', 'stars', 'followers', 'followings']
    for col in columns_to_parse:
        if col in data.columns:
            data[col] = data[col].fillna(0).apply(parse_number)
        else:
            data[col] = 0

    required_columns = ['avatar', 'experience', 'education', 'certifications',
                        'recommendations_count', 'about', 'repositories',
                        'stars', 'followers', 'followings']
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0

    data.fillna(0, inplace=True)
    return data

# Función para calcular métricas personalizadas
def calculate_custom_metrics(data):
    def safe_eval(x):
        try:
            return len(ast.literal_eval(x)) if isinstance(x, str) else 0
        except (ValueError, SyntaxError):
            return 0

    data['has_photo'] = data['avatar'].notnull().astype(int)
    data['num_experiences'] = data['experience'].apply(safe_eval)
    data['num_educations'] = data['education'].apply(safe_eval)
    data['num_certifications'] = data['certifications'].apply(safe_eval)
    data['num_recommendations'] = data['recommendations_count'].astype(int)
    data['has_summary'] = data['about'].notnull().astype(int)

    def calcular_calidad(row):
        score = 10 * row['has_photo'] + 5 * row['num_experiences'] + 3 * row['num_educations'] + \
                4 * row['num_certifications'] + 5 * row['num_recommendations'] + \
                10 * row['has_summary'] + 2 * row['repositories'] + \
                row['stars'] + 2 * row['followers'] + row['followings']
        return score

    data['quality_score'] = data.apply(calcular_calidad, axis=1)

    def categorizar(score):
        return "Alta calidad" if score >= 50 else "Media calidad" if score >= 30 else "Baja calidad"

    data['category'] = data['quality_score'].apply(categorizar)
    return data

# Función para generar gráficos
def generate_graphs(data):
    plots = {}

    # Gráfico de barras para la distribución por categoría
    plt.figure(figsize=(8, 6))
    sns.countplot(x='category', data=data, palette="viridis")
    plt.title('Distribución de Calidad de Perfiles')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['Distribución por Categoría'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Gráfico de caja (boxplot) para valores atípicos
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data.select_dtypes(include=['number']), palette="coolwarm")
    plt.title('Boxplot de Variables Numéricas')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['Boxplot Variables Numéricas'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Mapa de correlaciones
    plt.figure(figsize=(12, 10))
    numeric_data = data.select_dtypes(include=['number']).dropna(axis=1, how='all')
    numeric_data = numeric_data.loc[:, numeric_data.var() > 0]
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Mapa de Correlaciones')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['Mapa de Correlaciones'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()



    return plots

# Función principal de análisis
def analyze_data(file_path):
    data = load_and_preprocess(file_path)
    quality_stats = calculate_quality_stats(data)
    outlier_summary = calculate_outliers(data)
    data = process_columns(data)
    data = calculate_custom_metrics(data)
        # Entrenar el modelo de Random Forest
    model = train_random_forest_model(data)

    # Realizar las predicciones
    data = predict_with_random_forest(data)
    
    plots = generate_graphs(data)
    return data, plots, quality_stats, outlier_summary

# Rutas de Flask
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csv_file']
        file_path = 'uploaded_file.csv'
        file.save(file_path)

        data, plots, quality_stats, outlier_summary = analyze_data(file_path)

        kpis = {
            'Total Perfiles': len(data),
            'Perfiles Alta Calidad': len(data[data['category'] == 'Alta calidad']),
            'Perfiles Media Calidad': len(data[data['category'] == 'Media calidad']),
            'Perfiles Baja Calidad': len(data[data['category'] == 'Baja calidad']),
        }

        # Crear los datos para el gráfico
        chart_data_kpi = {
            'labels': ['Alta Calidad', 'Media Calidad', 'Baja Calidad'],
            'values': [kpis['Perfiles Alta Calidad'], kpis['Perfiles Media Calidad'], kpis['Perfiles Baja Calidad']]
        }

        return render_template(
            'results.html',
            kpis=kpis,
            plots=plots,
            quality_summary=quality_stats,
            outlier_summary=outlier_summary,
            chart_data_kpi=chart_data_kpi
        )

    return render_template('index.html')

@app.route('/download_csv')
def download_csv():
    data, _, _, _ = analyze_data('uploaded_file.csv')
    output_path = 'perfiles_calificados_con_metricas.xlsx'
    data.to_excel(output_path, index=False)
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
