<div style="text-align: center;">

# Actividad 5 | Visualización de Resultados

### Validación cruzada *k*-fold y visualización del mejor modelo sobre Big Data (PySpark)

**Tecnológico de Monterrey** · Maestría en Inteligencia Artificial Aplicada
**TC4034 — Análisis de Grandes Volúmenes de Datos · Módulo 6 (Semanas 9 y 10)**

**Entregable:** Visualización_Equipo61 · **Modalidad:** En equipo · **Fecha:** 21 de junio de 2026

**Integrantes — Equipo 61**
Eduardo Ramos Hernández (A01797393) · Diana Gabriela Ramírez Moreno (A01630769)
Oscar Ramírez Anaya (A01795438) · Emmanuel Francisco Ramírez Hernández (A01796289)

**Profesor:** Dr. Iván Olmos Pineda

</div>

---

## Descripción

Este repositorio mide la **variabilidad** y la **calidad de generalización** del mejor modelo de la
Actividad 4 —un `RandomForestClassifier` que predice si un viaje deja **propina alta** (`tip_alta`)—
mediante un proceso de **validación cruzada *k*-fold estratificada** sobre la muestra representativa **M**,
y comunica los resultados con **herramientas de visualización**. Se trabaja sobre la base global
**NYC TLC Yellow Taxi 2024** con **PySpark (MLlib)** y un muestreo estratificado proporcional reutilizado
del Módulo 3.

El entregable es un único notebook **ejecutado, con outputs y gráficos**:

| Notebook | Escala | Cómputo | Salidas |
|---|---|---|---|
| **`Actividad5_VisualizacionResultados.ipynb`** | 12 meses 2024 → M ≈ 80k | PySpark **local** (`local[*]`) | **Ejecutado, con gráficos** |

> El mismo pipeline escala sin cambios a **GCP Dataproc** leyendo desde Cloud Storage. El parámetro `MESES`
> permite una corrida local más ligera (3 meses) sin cambiar la metodología.

## El notebook — descargar o visualizar

- **Visualizar en GitHub:** [`Actividad5_VisualizacionResultados.ipynb`](./Actividad5_VisualizacionResultados.ipynb)
- **Visor enriquecido (nbviewer):** https://nbviewer.org/github/oscar-ramirez-anaya/vis_resutlts/blob/main/Actividad5_VisualizacionResultados.ipynb
- **Descargar (raw):** https://raw.githubusercontent.com/oscar-ramirez-anaya/vis_resutlts/main/Actividad5_VisualizacionResultados.ipynb

## Las cinco secciones del notebook

1. **Definición del proceso de validación cruzada** — argumentación multidimensional del valor **k = 5**
   (sesgo–varianza de Kohavi, representatividad por estrato verificada y costo en *Big Data*), con tabla
   comparativa k = 3/5/10.
2. **Construcción de los *k*-folds** — reparto **estratificado y determinista** (`xxhash64`, reproducible,
   sin `rand()`) por `particion_id × tip_alta`, con verificación de disjunción/exhaustividad, balance de
   clase y composición por estrato.
3. **Experimentacion** — entrenamiento del mejor modelo por pliegue con métricas **distribuidas** (AUC-ROC,
   AUC-PR) y **binarias por clase** (F1, MCC, Balanced Accuracy), brecha de sobre-ajuste, tiempos y baseline.
4. **Resultados** — visualización: barras por pliegue, boxplot + violín, curvas ROC/PR con banda de
   variabilidad, sobre-ajuste train vs. test, matriz de confusión, mapa de calor métrica × pliegue,
   calibración, sensibilidad al umbral, importancia de variables y una **vista interactiva (Plotly)**.
5. **Discusión y conclusiones** — significancia frente al baseline, variabilidad (CV %, IC 95 %),
   generalización, limitaciones y trabajo futuro.

## Resultados

**Volumen procesado (12 meses, local).** Base global **D = 41,169,720** registros → capa **Silver
39,263,800** (−4.63 %) → muestra **M = 80,006** → población supervisada (solo tarjeta) **60,574**, con
**prevalencia `tip_alta=1` = 0.758** (la clase positiva es mayoritaria).

**Validación cruzada estratificada (k = 5).** Métricas sobre el conjunto de prueba, promediadas entre
pliegues:

| Métrica (k=5) | Media ± std | CV % |
|---|---|---|
| AUC-ROC | **0.5825 ± 0.0068** | 1.17 |
| AUC-PR | **0.8054 ± 0.0032** | 0.40 |
| F1 (`tip_alta`) | 0.8623 ± 0.0001 | ~0 |
| MCC | 0.000 ± 0.010 | — |
| Brecha train−test (AUC) | **+0.0343** | — |
| Baseline trivial | AUC-ROC 0.500 · AUC-PR 0.758 | — |

**Lectura.** Los **coeficientes de variación de un dígito** muestran que el desempeño es **muy estable** y
**reproducible** entre particiones —el objetivo central de esta actividad—. El modelo supera al baseline en
*ranking* (+0.082 AUC-ROC) y **generaliza** (brecha train−test pequeña), pero su **poder discriminante en
la decisión dura es modesto** (MCC ≈ 0): predecir la propina alta a partir de variables operativas del
viaje es un problema **intrínsecamente difícil**, como confirman la calibración, la separación de clases y
el análisis de umbral del notebook. El valor del trabajo es **metodológico**: cuantificar la variabilidad
y comunicarla con visualizaciones, no solo reportar una cifra puntual.

## Mapeo a la rúbrica (cada criterio 20 %)

1. **Definir validación cruzada** — §1: argumentación de **k** y representatividad de cada pliegue.
2. **Construcción de los k-folds** — §2: poblado estratificado, determinista y verificado.
3. **Fase de entrenamiento** — §3: mejor modelo por pliegue, métricas adecuadas y control de sobre-ajuste.
4. **Visualización de resultados** — §4: repertorio amplio de gráficas con interpretación.
5. **Discusión y conclusiones** — §5: análisis de significancia y variabilidad.

## Cómo ejecutar

```bash
bash start_jupyter.sh         # descarga los 12 meses de 2024 si faltan y abre JupyterLab
# Run -> Restart Kernel and Run All

# Corrida local más ligera (3 meses):
MESES="01 02 03" bash start_jupyter.sh
```

Entorno: PySpark local (`local[*]`, Java 17). Datos públicos NYC TLC; **no se versionan** (ver
`.gitignore`). El notebook se regenera de forma reproducible con:

```bash
python3 scripts/build_notebook.py
python3 -m jupyter nbconvert --to notebook --execute --inplace Actividad5_VisualizacionResultados.ipynb
```

## Estructura del repositorio

```
.
├── Actividad5_VisualizacionResultados.ipynb   # Notebook entregable — ejecutado con gráficos
├── README.md
├── start_jupyter.sh                           # lanzador local (PySpark + descarga de datos)
├── scripts/
│   └── build_notebook.py                      # generador reproducible del notebook (nbformat)
└── .gitignore                                 # excluye datos, secretos y artefactos
```

---

*Datos: NYC TLC Trip Record Data (dominio público). La base global y su particionamiento se reutilizan de
las actividades previas del mismo curso.*
