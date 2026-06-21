#!/bin/bash
# ══════════════════════════════════════════════════════
#  Lanzador de Jupyter con PySpark — TC4034 Actividad 5
#  Visualización de resultados (validación cruzada k-fold)
#  Equipo 61
# ══════════════════════════════════════════════════════
export SPARK_HOME=/opt/homebrew/opt/apache-spark/libexec
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.17/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

echo "╔══════════════════════════════════════════╗"
echo "║   PySpark + Jupyter — TC4034 · Módulo 6   ║"
echo "║   Actividad 5 — Equipo 61                 ║"
echo "╚══════════════════════════════════════════╝"
echo ""
python3 -c "import pyspark; print('PySpark', pyspark.__version__, 'listo')" || {
    echo "PySpark no importable. Verifica la instalacion."; exit 1; }

cd "$(dirname "$0")"

# Asegura que existan los datos. Por defecto descarga los 12 meses de 2024
# (configura MESES="01 02 03" para una corrida local más ligera).
MESES="${MESES:-01 02 03 04 05 06 07 08 09 10 11 12}"
BASE=https://d37ci6vzurychx.cloudfront.net/trip-data
mkdir -p data/yellow_2024
for m in $MESES; do
  f=data/yellow_2024/yellow_tripdata_2024-$m.parquet
  if [ ! -f "$f" ]; then
    echo "Descargando 2024-$m ..."
    curl -fsSL -o "$f" "$BASE/yellow_tripdata_2024-$m.parquet" && echo "  OK 2024-$m"
  fi
done

echo ""
echo "Abriendo JupyterLab. Abre Actividad5_VisualizacionResultados.ipynb y ejecuta todas las celdas."
python3 -m jupyter lab Actividad5_VisualizacionResultados.ipynb
