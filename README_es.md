*Leer este documento en otros idiomas: [English](README.md)*

# AI4I 2020 Predictive Maintenance - Clasificación Binaria Avanzada de Fallas

## Descripción General del Proyecto

Este proyecto desarrolla un pipeline integral de Machine Learning para la clasificación binaria en mantenimiento predictivo utilizando el conjunto de datos AI4I 2020 Predictive Maintenance Dataset.

El objetivo principal es anticipar comportamientos anómalos de las máquinas y predecir fallas generales basándose en mediciones operativas y de sensores. En entornos industriales, el principal valor de negocio radica en alertar proactivamente a los técnicos sobre una falla inminente, en lugar de predecir el modo exacto de falla, especialmente cuando ciertos tipos específicos de fallas son extremadamente poco frecuentes.

Esta solución combina una definición del problema orientada al negocio, ingeniería de características basada en la física del proceso, benchmarking automatizado de algoritmos de Gradient Boosting, optimización de hiperparámetros con Optuna y una selección del umbral de decisión libre de data leakage para construir un sistema de clasificación robusto.

---

## Resultados (Conjunto de Prueba No Visto)

Modelo final: **XGBoost** (seleccionado automáticamente por el benchmark; la búsqueda con Optuna no superó su configuración por defecto en validación, por lo que se mantuvieron los valores por defecto), evaluado sobre un conjunto de prueba estratificado de 1.500 máquinas (51 fallas).

| Métrica | Valor |
|---------|-------|
| F1 Macro (umbral 0,50) | 0,949 |
| Recall de la clase Falla (umbral 0,50) | 0,90 |
| Precision de la clase Falla (umbral 0,50) | 0,90 |
| PR-AUC (Average Precision) | 0,937 |
| ROC-AUC | 0,988 |

Puntos de operación de negocio (umbrales seleccionados con predicciones out-of-fold, nunca con el conjunto de prueba):

| Estrategia | Umbral | Fallas detectadas | Fallas no detectadas | Falsas alarmas |
|------------|--------|-------------------|----------------------|----------------|
| A - Por defecto | 0,50 | 46 | 5 | 5 |
| B - F1 óptimo (alta precisión) | ~0,87 | 46 | 5 | 1 |
| C - Recall ≥ 90% (aversión al riesgo) | ~0,007 | 49 | 2 | 90 |

En las predicciones out-of-fold, el recall con el umbral por defecto es 0,78, por lo que el recall de 0,90 en el conjunto de prueba debe leerse como un resultado favorable y no como un nivel garantizado.

---

## Conjunto de Datos

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository)

| Propiedad         | Valor                   |
| ----------------- | ----------------------- |
| Tarea             | Clasificación Binaria   |
| Instancias        | 10.000                  |
| Variables         | 6 Variables Principales |
| Tipo              | Multivariable           |
| Valores Faltantes | No                      |
| Tasa de Fallas    | 3,39%                   |

### Variables

* Tipo de Producto (L, M, H)
* Temperatura del Aire [K]
* Temperatura del Proceso [K]
* Velocidad de Rotación [rpm]
* Torque [Nm]
* Desgaste de la Herramienta [min]

---

## Definición de la Variable Objetivo (Cambio Arquitectónico)

Originalmente, el conjunto de datos contiene indicadores específicos de fallas (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure y Random Failure). Debido al extremo desbalance de clases de estos modos específicos, predecirlos individualmente produce una confiabilidad estadística limitada.

### Enfoque Estratégico

Todas las clases minoritarias de falla fueron agrupadas en una única macroclase, utilizando la propia etiqueta `Machine failure` del dataset:

**Falla de Máquina**

* 0 = Operación Normal
* 1 = Falla

### Ventaja Estadística

Este enfoque agrupa ejemplos aislados de fallas en una clase de anomalías más grande y estadísticamente significativa, permitiendo que modelos avanzados aprendan patrones de forma más segura y efectiva.

### Hallazgos de Calidad de Datos

El EDA muestra que los indicadores de modo de falla y la etiqueta global no están perfectamente alineados (9 fallas sin indicador de modo, 18 indicadores de modo sin falla y 24 filas con varios modos). Por eso se utiliza `Machine failure` como única fuente de verdad para la variable objetivo.

### Prevención de Data Leakage

Todos los indicadores específicos de modos de falla (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) son eliminados estrictamente del conjunto de variables antes del entrenamiento.

---

## Ingeniería de Características

Se diseñaron variables específicas del dominio que capturan los mecanismos físicos de falla documentados para el dataset:

| Variable | Fórmula | Modo de falla relacionado |
|----------|---------|---------------------------|
| `temperature_diff` | Temperatura del proceso − Temperatura del aire [K] | Heat Dissipation Failure |
| `power_w` | Torque [Nm] × Velocidad de rotación [rpm] × 2π / 60 [W] | Power Failure |
| `wear_torque_strain` | Desgaste de herramienta [min] × Torque [Nm] | Overstrain Failure |

---

## Preprocesamiento de Datos

Se implementó un pipeline modular de Scikit-Learn (ajustado únicamente con los datos de entrenamiento) que incluye:

### Variables Numéricas

* Imputación por Mediana
* Escalado Estandarizado

Los valores atípicos **no** se recortan a propósito: los valores extremos de torque, potencia y esfuerzo son la propia señal de falla, por lo que recortarlos eliminaría justamente la información que el modelo necesita.

### Variables Categóricas

* Imputación por la Categoría Más Frecuente
* Codificación de Categorías Raras (mediante Transformer Personalizado)
* One-Hot Encoding

---

## Manejo del Desbalance de Clases

Incluso con una variable objetivo binaria consolidada, las fallas representan solo ~3,4% de las operaciones.

Para abordar este problema sin introducir ruido sintético, se calculan pesos balanceados para las muestras y se incorporan directamente durante el entrenamiento. La **misma** ponderación se aplica de forma consistente en el benchmark, en cada trial de Optuna, en el reentrenamiento final y en la validación cruzada utilizada para seleccionar el umbral.

---

## Benchmarking de Modelos

El pipeline evalúa automáticamente múltiples arquitecturas de Gradient Boosting:

* Gradient Boosting Classifier (Scikit-Learn)
* HistGradientBoosting Classifier (Scikit-Learn)
* XGBoost
* LightGBM
* CatBoost

Los modelos se ordenan según el **F1 Macro** sobre el conjunto de validación, y se reporta el **PR-AUC** como complemento independiente del umbral. Ambas métricas son adecuadas para datasets desbalanceados donde los falsos negativos (fallas no detectadas) tienen un alto costo para el negocio.

---

## Optimización de Hiperparámetros

El modelo con mejor desempeño en el benchmark es seleccionado dinámicamente y optimizado mediante Optuna (sampler TPE con semilla fija, 40 trials). Los hiperparámetros ajustados solo se conservan si superan a la configuración por defecto en validación; en la ejecución final el mejor trial alcanzó un F1 Macro de validación de 0,899 frente a 0,908 de la configuración por defecto, por lo que se mantuvo XGBoost con sus valores por defecto.

El modelo final se reentrena con los conjuntos de entrenamiento y validación combinados antes de ser evaluado una única vez sobre un conjunto de prueba estratificado y completamente no visto.

---

## Selección del Umbral de Decisión

El umbral de decisión se trata como una palanca de negocio. Para no filtrar información del conjunto de prueba, los umbrales se seleccionan sobre **predicciones out-of-fold** (validación cruzada estratificada de 5 folds sobre entrenamiento + validación), y el conjunto de prueba solo se utiliza para confirmar cada estrategia.

---

## Evaluación

El rendimiento y la robustez del modelo se evalúan mediante:

* F1 Score (Macro y de la clase Falla)
* PR-AUC (Average Precision)
* Análisis de Precision y Recall según el umbral
* Reporte Completo de Clasificación
* Mapas de Calor de la Matriz de Confusión

---

## Estructura del Proyecto

```
├── data/
│   └── ai4i2020.csv
├── notebooks/
│   └── AI4I 2020 Predictive Maintenance.ipynb
├── requirements.txt
├── README.md
└── README_es.md
```

## Cómo Ejecutarlo

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Luego abrir `notebooks/AI4I 2020 Predictive Maintenance.ipynb` y ejecutar todas las celdas (el dataset se carga con una ruta relativa a la carpeta `notebooks/`). Una ejecución completa tarda unos minutos.

---

## Licencia

Este es un proyecto educativo y demostrativo de Ciencia de Datos basado en un conjunto de datos industrial de acceso público con fines analíticos.

---

## Autor

**Armando Guarnera**  
Científico de Datos  
Argentina
