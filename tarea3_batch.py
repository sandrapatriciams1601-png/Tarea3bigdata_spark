"""
Tarea 3 - Procesamiento de Datos con Apache Spark
Curso Big Data (UNAD, 202016911)

Procesamiento en BATCH del dataset "Beneficiarios Familias en su Tierra"
(Prosperidad Social - datos.gov.co)

Problema a resolver:
Analizar la distribucion geografica, demografica y socioeconomica de los
beneficiarios del programa Familias en su Tierra, con el fin de identificar
patrones de cobertura por departamento, tipo de poblacion (desplazados vs.
vulnerables) y tipo de beneficio asignado.

Autor: Sandra Patricia Mosquera Soto
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ---------------------------------------------------------------------------
# 1. CREACION DE LA SESION SPARK
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("Tarea3_FamiliasEnSuTierra") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 70)
print("TAREA 3 - PROCESAMIENTO EN BATCH CON APACHE SPARK")
print("Dataset: Beneficiarios Familias en su Tierra (Prosperidad Social)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 2. CARGA DEL CONJUNTO DE DATOS
# ---------------------------------------------------------------------------
# El archivo fue descargado desde datos.gov.co:
# https://www.datos.gov.co/api/views/mebh-t5gy/rows.csv?accessType=DOWNLOAD
# y subido previamente a HDFS: hdfs dfs -put dataset.csv /Tarea3/

ruta = "hdfs://localhost:9000/Tarea3/dataset.csv"

df = spark.read.csv(ruta, header=True, inferSchema=True, encoding="UTF-8")

print("\n--- Esquema del DataFrame ---")
df.printSchema()

print(f"\nNumero total de registros cargados: {df.count()}")
print(f"Numero de columnas: {len(df.columns)}")

print("\n--- Primeras 5 filas ---")
df.show(5, truncate=False)

# ---------------------------------------------------------------------------
# 3. LIMPIEZA Y TRANSFORMACION DE DATOS DATA FRAME
# ---------------------------------------------------------------------------

# 3.1 Eliminar duplicados exactos
df_limpio = df.dropDuplicates()
print(f"\nRegistros despues de eliminar duplicados: {df_limpio.count()}")

# 3.2 Estandarizar valores "ND" (No Definido) como nulos, para no sesgar el EDA
columnas_texto = ["Genero", "Etnia", "Discapacidad", "TipoBeneficio",
                   "NivelEscolaridad", "EstadoCivil", "CondicionSexual"]

for col in columnas_texto:
    df_limpio = df_limpio.withColumn(
        col, F.when(F.trim(F.col(col)) == "ND", None).otherwise(F.col(col))
    )

# 3.3 Normalizar texto (quitar espacios, convertir a formato consistente)
df_limpio = df_limpio.withColumn(
    "NombreDepartamentoAtencion", F.trim(F.upper(F.col("NombreDepartamentoAtencion")))
)

# 3.4 Extraer el anio de inscripcion a partir de la fecha (para el analisis temporal)
df_limpio = df_limpio.withColumn(
    "AnioInscripcion", F.year(F.to_date(F.col("FechaInscripcionBeneficiario"), "yyyy-MM-dd"))
)

# 3.5 Filtrar registros validos (con departamento y tipo de poblacion informados)
df_limpio = df_limpio.filter(
    F.col("NombreDepartamentoAtencion").isNotNull() &
    F.col("TipoPoblacion").isNotNull()
)

print(f"\nRegistros finales despues de limpieza: {df_limpio.count()}")

# Cachear el DataFrame limpio porque se reutiliza en varias consultas
df_limpio.cache()

# ---------------------------------------------------------------------------
# 4. ANALISIS EXPLORATORIO DE DATOS (EDA) - con DataFrames
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("4.1 Distribucion de beneficiarios por DEPARTAMENTO (Top 10)")
print("=" * 70)
por_departamento = (
    df_limpio.groupBy("NombreDepartamentoAtencion")
    .agg(F.count("*").alias("total_beneficiarios"))
    .orderBy(F.desc("total_beneficiarios"))
)
por_departamento.show(10, truncate=False)

print("\n" + "=" * 70)
print("4.2 Distribucion por TIPO DE POBLACION")
print("=" * 70)
por_tipo_poblacion = (
    df_limpio.groupBy("TipoPoblacion")
    .agg(F.count("*").alias("total"))
    .orderBy(F.desc("total"))
)
por_tipo_poblacion.show(truncate=False)

print("\n" + "=" * 70)
print("4.3 Distribucion por RANGO DE EDAD")
print("=" * 70)
por_edad = (
    df_limpio.groupBy("RangoEdad")
    .agg(F.count("*").alias("total"))
    .orderBy(F.desc("total"))
)
por_edad.show(truncate=False)

print("\n" + "=" * 70)
print("4.4 Distribucion por GENERO")
print("=" * 70)
por_genero = (
    df_limpio.groupBy("Genero")
    .agg(F.count("*").alias("total"))
    .orderBy(F.desc("total"))
)
por_genero.show(truncate=False)

print("\n" + "=" * 70)
print("4.5 Distribucion por TIPO DE BENEFICIO")
print("=" * 70)
por_beneficio = (
    df_limpio.groupBy("TipoBeneficio")
    .agg(F.count("*").alias("total"))
    .orderBy(F.desc("total"))
)
por_beneficio.show(truncate=False)

print("\n" + "=" * 70)
print("4.6 Cruce: Tipo de poblacion x Genero (tabla dinamica)")
print("=" * 70)
cruce_poblacion_genero = df_limpio.groupBy("TipoPoblacion").pivot("Genero").count()
cruce_poblacion_genero.show(truncate=False)

# ---------------------------------------------------------------------------
# 5. ANALISIS EXPLORATORIO CON RDDs
# 
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("5. ANALISIS CON RDDs (map, filter, reduceByKey, collect, count)")
print("=" * 70)

# Convertimos el DataFrame limpio a RDD de filas
rdd = df_limpio.rdd

# 5.1 TRANSFORMACION map: extraer tuplas (departamento, 1)
rdd_departamentos = rdd.map(lambda fila: (fila["NombreDepartamentoAtencion"], 1))

# 5.2 TRANSFORMACION reduceByKey: sumar beneficiarios por departamento
conteo_departamentos = rdd_departamentos.reduceByKey(lambda a, b: a + b)

# 5.3 ACCION collect + ordenamiento: traer el resultado al driver
top_departamentos_rdd = sorted(conteo_departamentos.collect(), key=lambda x: -x[1])[:5]
print("\nTop 5 departamentos (calculado con RDD - reduceByKey):")
for depto, total in top_departamentos_rdd:
    print(f"  {depto}: {total}")

# 5.4 TRANSFORMACION filter: quedarnos solo con beneficiarios en condicion de discapacidad
rdd_con_discapacidad = rdd.filter(lambda fila: fila["Discapacidad"] == "SI")

# 5.5 ACCION count: contar cuantos hay
total_con_discapacidad = rdd_con_discapacidad.count()
print(f"\nBeneficiarios en condicion de discapacidad (RDD - filter + count): {total_con_discapacidad}")

# 5.6 TRANSFORMACION flatMap: separar y contar beneficios por tipo (demostrativo)
rdd_beneficios = rdd.filter(lambda f: f["TipoBeneficio"] is not None) \
                     .flatMap(lambda f: [f["TipoBeneficio"]])
conteo_beneficios = rdd_beneficios.map(lambda b: (b, 1)).reduceByKey(lambda a, b: a + b)

# 5.7 ACCION take: ver los primeros 5 resultados sin traer todo el RDD
print("\nPrimeros 5 tipos de beneficio (RDD - take):")
for beneficio, total in conteo_beneficios.take(5):
    print(f"  {beneficio}: {total}")

# 5.8 ACCION reduce: calcular el total general de beneficiarios sumando los conteos
total_general = conteo_departamentos.map(lambda x: x[1]).reduce(lambda a, b: a + b)
print(f"\nTotal general de beneficiarios (RDD - reduce): {total_general}")

# ---------------------------------------------------------------------------
# 6. VISUALIZACION DE RESULTADOS
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("\n" + "=" * 70)
print("6. Generando visualizaciones...")
print("=" * 70)

# Para graficar, convertimos los resultados ya agregados (pequenos) a Pandas
pd_departamentos = por_departamento.limit(10).toPandas()
pd_tipo_poblacion = por_tipo_poblacion.toPandas()
pd_edad = por_edad.toPandas()
pd_genero = por_genero.toPandas()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].barh(pd_departamentos["NombreDepartamentoAtencion"], pd_departamentos["total_beneficiarios"], color="#2E5C8A")
axs[0, 0].set_title("Top 10 departamentos por numero de beneficiarios")
axs[0, 0].invert_yaxis()

axs[0, 1].bar(pd_tipo_poblacion["TipoPoblacion"], pd_tipo_poblacion["total"], color="#4A9B8E")
axs[0, 1].set_title("Beneficiarios por tipo de poblacion")
axs[0, 1].tick_params(axis="x", rotation=30)

axs[1, 0].bar(pd_edad["RangoEdad"], pd_edad["total"], color="#D98B3F")
axs[1, 0].set_title("Beneficiarios por rango de edad")

axs[1, 1].pie(pd_genero["total"], labels=pd_genero["Genero"], autopct="%1.1f%%", colors=["#C85A54", "#5B8FB9", "#AAAAAA"])
axs[1, 1].set_title("Distribucion por genero")

plt.tight_layout()
plt.savefig("resultados_eda_tarea3.png", dpi=150)
print("Grafico guardado como: resultados_eda_tarea3.png")

# ---------------------------------------------------------------------------
# 7. CIERRE DE LA SESION SPARK
# ---------------------------------------------------------------------------
print("\nProceso de analisis en batch finalizado correctamente.")
spark.stop()
