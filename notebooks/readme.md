# Celda 1: Instalación de dependencias con restricción de versión para NumPy
!pip install -q "numpy<2.4" pyreadstat duckdb ydata-profiling xgboost shap plotly folium scikit-learn

# Forzar el reinicio automático del runtime para aplicar los cambios de NumPy
import os
os.kill(os.getpid(), 9)

import sys

try:
    import pandas as pd
    import duckdb
    import numpy as np
    import pyreadstat
    from ydata_profiling import ProfileReport
    import plotly
    import folium
    import sklearn
    import xgboost as xgb
    import shap
    import matplotlib

    print("✅ ¡Todas las librerías se importaron correctamente sin errores de versión!")
    print(f"- Python: {sys.version.split()[0]}")
    print(f"- Pandas: {pd.__version__}")
    print(f"- DuckDB: {duckdb.__version__}")
    print(f"- NumPy: {np.__version__}")
    print(f"- Pyreadstat: {pyreadstat.__version__}")
    print(f"- Plotly: {plotly.__version__}")
    print(f"- Folium: {folium.__version__}")
    print(f"- Scikit-Learn: {sklearn.__version__}")
    print(f"- XGBoost: {xgb.__version__}")
    print(f"- SHAP: {shap.__version__}")
    print(f"- Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Error de importación detectado: {e}")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")


    # Celda 3: Carga de datos y preparación
import pyreadstat
import duckdb

# Reemplaza la ruta por la ubicación real de tus archivos en Colab
df_persona, meta_p = pyreadstat.read_sav('/content/drive/MyDrive/Data Science2026/Mod3_tarea2/EH2025_Persona.sav')
df_vivienda, meta_v = pyreadstat.read_sav('/content/drive/MyDrive/Data Science2026/Mod3_tarea2/EH2025_Vivienda_1.sav')

# Conexión a DuckDB
con = duckdb.connect(database=':memory:')
con.register('persona', df_persona)
con.register('vivienda', df_vivienda)

# Consulta SQL para unir datos y crear la variable objetivo (Asistencia_Escolar)
# s03b_10: 1, 2, 3 = Asiste (Presencial, Semipresencial, Virtual). 4 = No asiste.
query = """
    SELECT
        p.folio,
        p.area,
        p.depto,
        p.s01a_02 AS sexo,
        p.s01a_03 AS edad,
        p.s01a_09 AS etnia,
        v.s06a_01 AS tipo_vivienda,
        CASE
            WHEN p.s03b_10 IN (1, 2, 3) THEN 1
            WHEN p.s03b_10 = 4 THEN 0
            ELSE NULL
        END AS Asistencia_Escolar
    FROM persona p
    INNER JOIN vivienda v ON p.folio = v.folio
    WHERE p.s01a_03 BETWEEN 4 AND 18 -- Rango de edad escolar
"""

# Ejecutar consulta y convertir a DataFrame
df_analisis = con.execute(query).df().dropna(subset=['Asistencia_Escolar'])

# Exportar datos procesados
df_analisis.to_parquet('dataset_procesado.parquet')

# Celda 4: EDA Automatizado con YData-Profiling
profile = ProfileReport(df_analisis, title="Reporte Exploratorio EH 2025 - Deserción Escolar", explorative=True)
profile.to_file("Reporte_EDA_EH2025.html")
# Descarga este HTML y colócalo en la carpeta /outputs/ de tu GitHub

Celda 5: Modelado Predictivo (Random Forest vs XGBoost) y Explicabilidad (SHAP)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap

# Preparar variables predictoras (X) y objetivo (y)
# Rellenamos nulos simples para evitar problemas con algoritmos
features = ['area', 'depto', 'sexo', 'edad', 'etnia', 'tipo_vivienda']
X = df_analisis[features].fillna(df_analisis[features].mode().iloc[0])
y = df_analisis['Asistencia_Escolar']

# Dividir en entrenamiento y prueba (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 1. Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 2. XGBoost
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=(len(y) - sum(y))/sum(y))
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluación de Modelos
print("=== REPORTE RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Random Forest: {roc_auc_score(y_test, y_prob_rf):.4f}\n")

print("=== REPORTE XGBOOST ===")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC XGBoost: {roc_auc_score(y_test, y_prob_xgb):.4f}\n")

# Explicabilidad con SHAP usando el modelo XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

# Visualizar la importancia de características mediante SHAP
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Importancia de Variables (SHAP) - Deserción Escolar", fontsize=14)
plt.tight_layout()
plt.show()

Celda 6: Clustering por Perfil de Riesgo utilizando K-Means

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Agregamos por departamento para perfilar el riesgo geográfico/demográfico
perfil_depto = df_analisis.groupby('depto').agg(
    tasa_desercion=('Asistencia_Escolar', lambda x: 1 - x.mean()),
    edad_promedio=('edad', 'mean'),
    proporcion_rural=('area', lambda x: (x == 2).mean())
).reset_index()

# Escalar características para K-Means
scaler = StandardScaler()
features_to_scale = ['tasa_desercion', 'edad_promedio', 'proporcion_rural']
scaled_features = scaler.fit_transform(perfil_depto[features_to_scale])

# Definir clusters (ej. 3 niveles de riesgo: Bajo, Medio, Alto)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
perfil_depto['Cluster_Riesgo'] = kmeans.fit_predict(scaled_features)

# Mapear clusters a etiquetas cualitativas basadas en la tasa de deserción promedio
cluster_means = perfil_depto.groupby('Cluster_Riesgo')['tasa_desercion'].mean().sort_values()
labels_map = {cluster_means.index[0]: 'Riesgo Bajo',
              cluster_means.index[1]: 'Riesgo Medio',
              cluster_means.index[2]: 'Riesgo Alto'}
perfil_depto['Perfil_Riesgo'] = perfil_depto['Cluster_Riesgo'].map(labels_map)

display(perfil_depto.sort_values(by='tasa_desercion', ascending=False))

Celda 7: Mapa Interactivo de Tasas de Deserción por Departamento (Folium)
import folium

# Coordenadas aproximadas de los departamentos de Bolivia para representación visual
coordenadas_deptos = {
    1: [-19.043, -65.261, "Chuquisaca"],
    2: [-16.500, -68.150, "La Paz"],
    3: [-17.389, -66.157, "Cochabamba"],
    4: [-17.983, -67.150, "Oruro"],
    5: [-19.583, -65.750, "Potosí"],
    6: [-21.535, -64.730, "Tarija"],
    7: [-17.786, -63.181, "Santa Cruz"],
    8: [-14.833, -64.900, "Beni"],
    9: [-11.020, -68.770, "Pando"]
}

# Construir dataframe con coordenadas
map_df = perfil_depto.copy()
map_df['lat'] = map_df['depto'].map(lambda x: coordenadas_deptos.get(int(x), [0, 0, ""])[0])
map_df['lon'] = map_df['depto'].map(lambda x: coordenadas_deptos.get(int(x), [0, 0, ""])[1])
map_df['nombre_depto'] = map_df['depto'].map(lambda x: coordenadas_deptos.get(int(x), [0, 0, "Desconocido"])[2])

# Inicializar mapa centrado en Bolivia
m = folium.Map(location=[-16.290154, -63.588653], zoom_start=5.5)

# Añadir marcadores por departamento con escala de colores
for _, row in map_df.iterrows():
    tasa = row['tasa_desercion'] * 100
    color = 'red' if row['Perfil_Riesgo'] == 'Riesgo Alto' else ('orange' if row['Perfil_Riesgo'] == 'Riesgo Medio' else 'green')

    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=folium.Popup(f"<b>{row['nombre_depto']}</b><br>"
                           f"Tasa de Deserción: {tasa:.2f}%<br>"
                           f"Nivel de Riesgo: {row['Perfil_Riesgo']}", max_width=300),
        tooltip=f"{row['nombre_depto']} - {tasa:.1f}% Deserción",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(m)

# Mostrar el mapa

Celda 8: Recomendaciones Focalizadas basadas en el Modelo
Basándonos en los resultados de los árboles de decisión (importancia de características por SHAP) y en el análisis de clusters geográficos:

Foco por Edad: Las variables como la edad suelen marcar hitos críticos de transición escolar (primaria a secundaria). Las políticas deben priorizar incentivos de retención entre los 12 y 16 años.
Mitigación Territorial: Enviar programas de bonos de transporte o becas directamente a los departamentos agrupados en Riesgo Alto.
Monitoreo Étnico-Social: Ajustar los planes de estudio y asistencia con pertinencia cultural según las variables de etnicidad y condiciones de vivienda.

# Análisis detallado de factores de deserción escolar (tasa_desercion = 1 - tasa_asistencia)
print("=== 1. INFLUENCIA DEL ÁREA GEOGRÁFICA ===")
area_map = {1: 'Urbana', 2: 'Rural'}
df_analisis['desc_area'] = df_analisis['area'].map(area_map)
print(df_analisis.groupby('desc_area')['Asistencia_Escolar'].agg(tasa_desercion=lambda x: f"{(1-x.mean())*100:.2f}%"))
print("\n")

print("=== 2. INFLUENCIA DE LA EDAD (Rangos Críticos) ===")
df_analisis['rango_edad'] = pd.cut(df_analisis['edad'], bins=[3, 11, 14, 18], labels=['Primaria (4-11)', 'Secundaria Baja (12-14)', 'Secundaria Alta (15-18)'])
print(df_analisis.groupby('rango_edad', observed=False)['Asistencia_Escolar'].agg(tasa_desercion=lambda x: f"{(1-x.mean())*100:.2f}%"))
print("\n")

print("=== 3. INFLUENCIA DE LA ETNICIDAD (Autoidentificación) ===")
etnia_map = {1: 'Indígena/Originario', 2: 'No Indígena'}
df_analisis['desc_etnia'] = df_analisis['etnia'].map(etnia_map)
print(df_analisis.groupby('desc_etnia')['Asistencia_Escolar'].agg(tasa_desercion=lambda x: f"{(1-x.mean())*100:.2f}%"))
print("\n")

print("=== 4. INFLUENCIA POR GÉNERO ===")
sexo_map = {1: 'Hombre', 2: 'Mujer'}
df_analisis['desc_sexo'] = df_analisis['sexo'].map(sexo_map)
print(df_analisis.groupby('desc_sexo')['Asistencia_Escolar'].agg(tasa_desercion=lambda x: f"{(1-x.mean())*100:.2f}%"))



# Caso 2 Suponiendo que df_ml es tu dataset integrado de la Parte 1 (DuckDB join)
# df_ml = pd.read_parquet('outputs/dataset_procesado_caso2.parquet')

# Para este script simularemos el DataFrame con la estructura del diccionario del INE
import numpy as np
import pandas as pd

np.random.seed(42)
df_ml = pd.DataFrame({
    'folio': [f'H{i}' for i in range(1000)],
    'depto': np.random.randint(1, 10, 1000), # 1 a 9 departamentos
    'area': np.random.choice([1, 2], 1000), # 1=Urbano, 2=Rural
    'edad': np.random.randint(18, 65, 1000),
    's06a_07': np.random.choice([1, 2, 7, 9], 1000), # Agua
    's06a_09': np.random.choice([1, 2, 5], 1000), # Baño
    's06a_12': np.random.choice([1, 2], 1000), # Luz
    's04f_35': np.random.choice([1, 2], 1000, p=[0.3, 0.7]), # 1=Aporta AFP (Formal)
    'ylab': np.random.lognormal(mean=8, sigma=0.8, size=1000) # Ingreso laboral
})

# Crear Variable Objetivo: Empleo Formal
df_ml['Empleo_Formal'] = np.where(df_ml['s04f_35'] == 1, 1, 0)

# Crear Proxy de Índice de Pobreza Multidimensional (IPM - Carencias básicas)
# Se asume carencia si: no red de agua, no alcantarillado, no luz
df_ml['carencia_agua'] = np.where(df_ml['s06a_07'] >= 7, 1, 0)
df_ml['carencia_bano'] = np.where(df_ml['s06a_09'] == 5, 1, 0)
df_ml['carencia_luz'] = np.where(df_ml['s06a_12'] == 2, 1, 0)
df_ml['IPM_Puntaje'] = df_ml['carencia_agua'] + df_ml['carencia_bano'] + df_ml['carencia_luz']

# Categorizar Pobreza Multidimensional de forma segura cerrando la sintaxis
df_ml['Condicion_Pobreza'] = pd.cut(
    df_ml['IPM_Puntaje'],
    bins=[-1, 0, 1, 3],
    labels=['Sin Carencias', 'Carencia Moderada', 'Carencia Alta']
)

# Mostrar cabecera
display(df_ml.head())

# ==============================================================================
# 2. SEGMENTACIÓN POR CLUSTERING (K-MEANS)
# ==============================================================================
# Escalado de variables numéricas y proxy IPM
X_cluster = df_ml[['edad', 'ylab', 'IPM_Puntaje']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Entrenar K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_ml['Cluster_Vulnerabilidad'] = kmeans.fit_predict(X_scaled)

# Asignar nombres a los clústeres para mejor interpretación
cluster_map = {0: 'Jóvenes Precarizados', 1: 'Adultos Informales', 2: 'Alta Pobreza Estructural', 3: 'Sector Formal Estable'}
df_ml['Perfil_Hogar'] = df_ml['Cluster_Vulnerabilidad'].map(cluster_map)

# ==============================================================================
# Resumen Estadístico de los Clústeres Obtenidos
# ==============================================================================
# Agrupamos por perfil de hogar para analizar el tamaño del clúster y promedios clave
resumen_clusters = df_ml.groupby('Perfil_Hogar').agg(
    Conteo_Hogares=('folio', 'count'),
    Edad_Promedio=('edad', 'mean'),
    Ingreso_Promedio_Bs=('ylab', 'mean'),
    Carencias_IPM_Promedio=('IPM_Puntaje', 'mean'),
    Tasa_Formalidad=('Empleo_Formal', 'mean')
).reset_index()

# Dar formato de porcentaje a la tasa de formalidad y redondear
resumen_clusters['Tasa_Formalidad'] = resumen_clusters['Tasa_Formalidad'].map(lambda x: f"{x*100:.1f}%")
resumen_clusters['Ingreso_Promedio_Bs'] = resumen_clusters['Ingreso_Promedio_Bs'].round(2)
resumen_clusters['Edad_Promedio'] = resumen_clusters['Edad_Promedio'].round(1)
resumen_clusters['Carencias_IPM_Promedio'] = resumen_clusters['Carencias_IPM_Promedio'].round(2)


# ==============================================================================
# 3. VISUALIZACIÓN 1: MAPA GEOESPACIAL INTERACTIVO (FOLIUM)
# ==============================================================================
# Calcular tasa de informalidad por departamento
tasas_depto = df_ml.groupby('depto')['Empleo_Formal'].apply(lambda x: 1 - x.mean()).reset_index()
tasas_depto.columns = ['depto', 'Tasa_Informalidad']

# Coordenadas aproximadas de capitales (1:Chuquisaca, 2:La Paz, 3:Cochabamba, etc.)
coords = {1: [-19.03, -65.26], 2: [-16.50, -68.15], 3: [-17.39, -66.16], 4: [-17.96, -67.11], 
          5: [-19.58, -65.75], 6: [-21.53, -64.73], 7: [-17.78, -63.18], 8: [-14.83, -64.90], 9: [-11.03, -68.77]}

mapa_informalidad = folium.Map(location=[-16.29, -63.58], zoom_start=5, tiles='cartodb positron')

for idx, row in tasas_depto.iterrows():
    dept = row['depto']
    tasa = row['Tasa_Informalidad']
    if dept in coords:
        color = 'darkred' if tasa > 0.75 else ('orange' if tasa > 0.6 else 'green')
        folium.CircleMarker(
            location=coords[dept], radius=tasa*30, color=color, fill=True, fill_opacity=0.7,
            popup=f"Depto: {dept}<br>Informalidad: {tasa:.1%}"
        ).add_to(mapa_informalidad)

# Mostrar mapa (en Colab se renderiza automáticamente al llamar la variable)
display(mapa_informalidad)

==============================================================================
# 4. VISUALIZACIÓN 2: SUNBURST MULTIVARIADO (PLOTLY)
# ==============================================================================
# Jerarquía: Área -> Condición Pobreza -> Empleo Formal
df_sunburst = df_ml.groupby(['area', 'Condicion_Pobreza', 'Empleo_Formal']).size().reset_index(name='conteo')
df_sunburst['area_str'] = np.where(df_sunburst['area'] == 1, 'Urbano', 'Rural')
df_sunburst['formalidad_str'] = np.where(df_sunburst['Empleo_Formal'] == 1, 'Formal', 'Informal')

fig_sunburst = px.sunburst(
    df_sunburst, path=['area_str', 'Condicion_Pobreza', 'formalidad_str'], values='conteo',
    title="Estructura de la Pobreza Multidimensional y Formalidad Laboral",
    color='Condicion_Pobreza', color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_sunburst.update_traces(textinfo='label+percent parent')
fig_sunburst.show()

# ==============================================================================
# 5. VISUALIZACIÓN 3: SCATTER PLOT INTERACTIVO (PLOTLY) - CORREGIDO
# ==============================================================================
# Filtramos un poco los outliers de ingresos para mejor visualización
df_scatter = df_ml[df_ml['ylab'] < df_ml['ylab'].quantile(0.95)].copy()

# Creamos la columna 'formalidad_str' en el dataframe del scatter plot para corregir el ValueError
df_scatter['formalidad_str'] = np.where(df_scatter['Empleo_Formal'] == 1, 'Formal', 'Informal')

fig_scatter = px.scatter(
    df_scatter, x='edad', y='ylab', color='formalidad_str', size='IPM_Puntaje',
    hover_data=['depto', 'Perfil_Hogar'], opacity=0.7,
    title="Relación entre Edad, Ingreso Laboral y Pobreza (Tamaño de burbuja = Carencias IPM)",
    labels={'ylab': 'Ingreso Mensual (Bs)', 'edad': 'Edad', 'formalidad_str': 'Condición Laboral'},
    color_discrete_map={'Formal': '#2ca02c', 'Informal': '#d62728'}
)
fig_scatter.show()

# ==============================================================================
# 6. VISUALIZACIÓN 4: PERFILES DE CLÚSTERES (PLOTLY RADAR/BAR)
# ==============================================================================
# Calcular promedios por clúster
cluster_avg = df_ml.groupby('Perfil_Hogar')[['ylab', 'edad', 'IPM_Puntaje', 'Empleo_Formal']].mean().reset_index()

fig_bar = px.bar(
    cluster_avg, x='Perfil_Hogar', y=['Empleo_Formal', 'IPM_Puntaje'], barmode='group',
    title="Caracterización de Perfiles: Tasa de Formalidad y Promedio de Carencias (IPM)",
    labels={'value': 'Promedio', 'variable': 'Indicador'}
)
fig_bar.show()

# Caso 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap
# Configuración visual
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.figsize': (10, 6)})

# PASO 1: PREPARACIÓN DEL DATASET (FEATURE ENGINEERING)
# ==============================================================================
# Cargar los datos consolidados (procedentes de DuckDB de la Parte 1)
# df_ml = pd.read_parquet('outputs/dataset_procesado_caso3.parquet')

# Generamos un DataFrame simulado basado en el diccionario del INE para demostración de código
np.random.seed(42)
n_samples = 2000
df_ml = pd.DataFrame({
    'depto': np.random.randint(1, 10, n_samples),                 # Departamento 1-9
    'area': np.random.choice([1, 2], n_samples),                  # 1: Urbano, 2: Rural
    'sexo': np.random.choice([1, 2], n_samples),                  # 1: Hombre, 2: Mujer
    'edad': np.random.randint(15, 80, n_samples),                 # Edad a partir de los 15 años
    'etnia': np.random.choice([1, 2], n_samples, p=[0.4, 0.6]),   # 1: Sí se autoidentifica, 2: No
    's09b_02a': np.random.choice([1, 2, 3, 4, 5, 6, 7], n_samples, p=[0.05, 0.05, 0.02, 0.03, 0.02, 0.03, 0.80]) # Delitos
})

# Variable Objetivo Binaria: Victimización (1 = Sufrió delito, 0 = Ninguno/Cod 7)
df_ml['Target_Victimizacion'] = np.where(df_ml['s09b_02a'] < 7, 1, 0)

# Selección de características predictoras (X) y el objetivo (y)
features = ['depto', 'area', 'sexo', 'edad', 'etnia']
X = df_ml[features]
y = df_ml['Target_Victimizacion']

# División del conjunto de datos (80% Entrenamiento, 20% Prueba con estratificación)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# PASO 2: MODELADO AVANZADO (VALIDACIÓN CRUZADA ESTRATIFICADA)
# ==============================================================================
print("Iniciando entrenamiento de modelos de Machine Learning...")

# Configurar Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modelo 1: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42)
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='roc_auc')

# Modelo 2: XGBoost Classifier
# Nota: scale_pos_weight ayuda a compensar el desbalance de clases (muchas más personas seguras que víctimas)
scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                          scale_pos_weight=scale_weight, random_state=42)
cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring='roc_auc')

print(f"Random Forest CV AUC-ROC Promedio: {cv_scores_rf.mean():.4f}")
print(f"XGBoost CV AUC-ROC Promedio:       {cv_scores_xgb.mean():.4f}")

# Entrenamiento final sobre todo el set de training
xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# PASO 3: EVALUACIÓN RIGUROSA (MÉTRICAS Y MATRIZ DE CONFUSIÓN)
# ==============================================================================
# Predecir con el modelo de mejor desempeño (Usaremos XGBoost como ejemplo principal)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- REPORTE DE CLASIFICACIÓN (XGBOOST) ---")
print(classification_report(y_test, y_pred_xgb, target_names=["Seguro (0)", "Victimizado (1)"]))

auc_xgb = roc_auc_score(y_test, y_prob_xgb)
print(f"Puntaje AUC-ROC en Test: {auc_xgb:.4f}")

# Visualización de la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Victimizado", "Victimizado"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Modelo XGBoost")
plt.grid(False)
plt.show()

# PASO 4: INTERPRETABILIDAD (eXplainable AI) MEDIANTE VALORES SHAP
# ==============================================================================
print("Calculando valores SHAP para interpretabilidad del modelo...")

# Inicializar el explainer de SHAP basado en los árboles de XGBoost
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# 1. Gráfico de Resumen (Summary Plot)
# Muestra la importancia de la variable y cómo su valor (alto/bajo) afecta la predicción
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title("Importancia de Variables (SHAP) - Predicción de Victimización")
plt.tight_layout()
plt.show()

# 2. Gráfico de Barras de Impacto Global
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Impacto Global de las Variables Predictoras")
plt.tight_layout()
plt.show()


