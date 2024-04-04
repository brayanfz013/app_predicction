# -*- coding: utf-8 -*-
# =============================================================================
__author__ = "Brayan Felipe Zapata "
__copyright__ = "Copyright 2007, The Cogent Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Brayan Felipe Zapata"
__email__ = "bzapata@smgsoftware.com"
__status__ = "Production"
# =============================================================================
"""Codigo para ejecutar un entrenamiento completo de una base de datos """
# =============================================================================

import os
import re
import yaml
import numpy as np
import pandas as pd
import logging

from scipy import stats, signal
from pathlib import Path
from src.lib.class_load import LoadFiles
from src.data.save_models import SAVE_DIR
from src.lib.utils import select_best_model
from src.lib.factory_data import SQLDataSourceFactory, get_data, create_table, set_data
from src.lib.factory_models import ModelContext
from src.lib.factory_prepare_data import (
    DataCleaner,
    DataModel,
    MeanImputation,
    OutliersToIQRMean,
    PrepareDtypeColumns,
    base_dtypes
)
from src.models.DP_model import Modelos
from src.features.features_redis import HandleRedis
from src.features.features_postgres import HandleDBpsql
from src.models.args_data_model import Parameters
from src.data.logs import LOGS_DIR

path_folder = os.path.dirname(__file__)
folder_model = Path(path_folder).joinpath("scr/data/save_models")

handler_load = LoadFiles()
handler_redis = HandleRedis()
data_source = HandleDBpsql()
ruta_actual = os.path.dirname(__file__)

# =================================================================
#             Configuracion Logger
# =================================================================
# Configura un logger personalizado en lugar de usar el logger raíz
logfile = ruta_actual + "/src/data/config/logging.conf"
logging.config.fileConfig(os.path.join(LOGS_DIR, logfile))
logger = logging.getLogger("predict")
logger.debug("Inciando secuencia de entrenamiento")

# =================================================================
#             Cargar los parametros
# =================================================================
CONFIG_FILE = ruta_actual + "/src/data/config/config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    parameters = yaml.safe_load(file)

logger.debug("Archivo de configuraciones cargado")
parametros = Parameters(**parameters)

# =================================================================
#             Cargar datos de la fuente de datos
# =================================================================
# Interacion para hacer un cache de los datos en redis
try:
    logger.debug("verficando si existe data cache")
    data = handler_redis.get_cache_data(
        hash_name=parametros.query_template["table"],
        config=parametros.connection_data_source,
    )
    # Condicional para actualizar datos en caso de existan datos en redis
    if data is not None:
        logger.debug("Existe data en cache")

        # Secuencia de codigo para perdir nuevos datos a la base de datos
        date_col_query = parameters["query_template"]["columns"]["0"]
        LAST_DAY = str(data.iloc[-1][0])
        parameters["query_template"]["where"] = f" \"{date_col_query}\" > '{LAST_DAY}'"
        parameters["query_template"]["order"] = "".join(
            ['"' + columna + '"' for columna in [date_col_query]]
        )

        logger.debug("Realizando peticion a la fuente de datos")

        # Extraccion de la nueva data para actualizar
        new = get_data(SQLDataSourceFactory(**parameters))
        logger.debug("Actualizando cache en redis")
        data = handler_redis.set_cache_data(
            hash_name=parametros.query_template["table"],
            old_dataframe=data,
            new_dataframe=new,
            exp_time=parametros.exp_time_cache,
            config=parametros.connection_data_source,
        )
        logger.debug("Actualizacion completa de datos en redis")

    # Verificar que existieran datos en cache
    if data is None:
        logger.debug("No existe cache de datos")

        data = get_data(SQLDataSourceFactory(**parameters))
        logger.debug("Insertando datos de cache en redis")
        data = handler_redis.set_cache_data(
            hash_name=parametros.query_template["table"],
            old_dataframe=data,
            new_dataframe=None,
            exp_time=parametros.exp_time_cache,
            config=parametros.connection_data_source,
        )

except ValueError as error:
    logger.debug("[ERROR] No se puede hacer un cache de la fuente de datos")
    logger.debug(error)
    exit()
# =================================================================
#             Limpieza de datos
# =================================================================
new_types = []

for dtypo in parameters["type_data"].values():

    new_types.append(base_dtypes[dtypo])

# metodo para transformar los tipo de datos
strategy = {int: np.mean, float: np.mean, object: stats.mode}

# Estrategias para imputar los datos faltantes de NA
replace = {
    int: lambda x: int(float(x.replace(",", ""))),
    float: lambda x: float(x.replace(",", "")),
    object: lambda x: x.strip(),
}
# =================================================================

update_dtype_columns = PrepareDtypeColumns(
    replace_dtypes=new_types,
    strategy_imputation=strategy,
    preprocess_function=replace,
    **parameters,
)

# Ejecucion de fabrica para aplicar y ordenar los tipos de datos y los valores
cleaner = DataCleaner()
cleaner.strategy = update_dtype_columns
data_ = cleaner.clean(data)

# Condicion de filtrado para informacion segun los valores
filter_label: str = parameters["filter_data"]["filter_1_feature"]
filter_col: str = parameters["filter_data"]["filter_1_column"]
filter_product = data_.dataframe[filter_col] == filter_label
filter_data = data_.dataframe[filter_product].sort_values(
    by=parameters["filter_data"]["date_column"])

# Seleccion de agrupacion de tiempo
# parameters["filter_data"]["group_frequecy"] = "M"
# parameters["filter_data"]["filter_1_feature"] = filter_label

# # Datos de validacion
# validate_data = filter_data.set_index(time_series_col)["2023-12-01":].reset_index()

# # Datos de entrenamiento
# filter_data = filter_data.set_index(time_series_col)[:"2023-11-30"].reset_index()

outliners = OutliersToIQRMean(**parameters)
cleaner.strategy = outliners
outlines_data = cleaner.clean(filter_data)
# validate_outlines = cleaner.clean(validate_data)
print(outlines_data)
# Filtrado de datos para eliminar valores negativos
filter_values = outlines_data["quantity"] <= 0
outlines_data[filter_values] = 0.1

# =================================================================
#             Filtro pasabajos
# =================================================================
fs = 1 / 24 / 3600  # 1 day in Hz (sampling frequency)

nyquist = fs / 0.5  # 2 # 0.5 times the sampling frequency
cutoff = 0.5  # 0.1 fraction of nyquist frequency, here  it is 5 days
# cutoff=  4.999999999999999  days
b, a = signal.butter(5, cutoff, btype="lowpass")  # low pass filter

dUfilt = signal.filtfilt(b, a, outlines_data["quantity"])
dUfilt = np.array(dUfilt)
dUfilt = dUfilt.transpose()
outlines_data["low_past"] = dUfilt

# =================================================================
#             Preparacion de datos para el modelo
# =================================================================
data_for_model = DataModel(**parameters)
cleaner.strategy = data_for_model
data_ready, scaler_data = cleaner.clean(outlines_data)

# Creacion del dataframe para del filtro pasa bajo para los datos
low_pass_data = outlines_data["low_past"]
low_pass_data = low_pass_data.to_frame()
low_pass_data.rename(columns={"low_past": "quantity"}, inplace=True)
data_ready_lp, scaler_data_lp = cleaner.clean(low_pass_data)

# =================================================================
#            Cargar modelo
# =================================================================
# Rutas de los parametros para predicciones
save_dir = Path(SAVE_DIR).joinpath(parameters["filter_data"]["filter_1_feature"])
models_metrics = save_dir.joinpath("train_metrics").with_suffix(".json").as_posix()

MODE_USED = select_best_model(models_metrics)

scaler_name = save_dir.joinpath("scaler").with_suffix(".pkl").as_posix()
scaler_lp_name = save_dir.joinpath("scaler_lp").with_suffix(".pkl").as_posix()
last_pred = save_dir.joinpath("previus").with_suffix(".json").as_posix()
model_train = save_dir.joinpath(f"model_{MODE_USED}").with_suffix(".pt").as_posix()
parameters_model = save_dir.joinpath(f"parametros_{MODE_USED}").with_suffix(".json").as_posix()

modelo = ModelContext(model_name=MODE_USED,
                      data=data_ready,
                      split=83,
                      covarianze=data_ready_lp,
                      ** parameters
                      )

# Cargar escaler
scaler = handler_load.load_scaler(scaler_name)
scaler_lp = handler_load.load_scaler(scaler_lp_name)

# Cargar modelo para hacer las predicciones
IntModel = Modelos[MODE_USED]
trained_parameters = handler_load.json_to_dict(json_file=parameters_model)[0]
model_update_parameters = IntModel(**trained_parameters)
model_trained = model_update_parameters.load(model_train)

# Se carga el modelo , los datos de predicciones ya se cargaron previamente en ModelContext
pred_series = modelo.predict(
    model=model_trained,
    data=None,
    horizont=parameters["forecast_val"]
)

# Invertir predicciones escaler de entrenamietno
pred_scale = scaler.inverse_transform(pred_series)

# Invertir Predicciones escaler de transformacion de los datos
# pred_scale = scaler_data.inverse_transform(pred_series)

data_frame_predicciones = pred_scale.pd_dataframe()
column_field = list(data_frame_predicciones.columns)
data_frame_predicciones.reset_index(inplace=True)
data_frame_predicciones[parameters["filter_data"]
                        ["predict_column"]].clip(lower=0, inplace=True)

# ===============================================================================================
#                        PREDICCIONES
# ===============================================================================================
# TODO:
"""Esta parte tiene un ToDo importante: Tiene que ordenarse y optimizarce para se escalable
   De momento funciona de manera estatica para ciertas cosas sobre todo el tema de la escritura
   en postgres, ademas de tener codigo copiado de funciones internas ya ordenadas
"""
logger.debug("Enviando valor de las predicciones")
# Cuantificar metricas de la columan de predicciones
metric_columns_pred = data_.metrics_column(
    data_frame_predicciones[parameters["filter_data"]["predict_column"]]
)
# Seleccion de columans para generar el dataframe de salida para la base de datos
filter_temp = []
for filter_list in parameters["filter_data"]:
    if "feature" in filter_list:
        filter_temp.append(parameters["filter_data"][filter_list])

for adding_data in filter_temp:
    data_frame_predicciones[str(adding_data)] = adding_data

new_names = list(parameters["query_template_write"]["columns"].values())
rename = {x: y for x, y in zip(
    list(data_frame_predicciones.columns), new_names)}
data_frame_predicciones.rename(columns=rename, inplace=True)

# Crear tabla para guardas la informacion
create_table(SQLDataSourceFactory(**parameters))

# Ingresar los datos a la base de datos
set_data(SQLDataSourceFactory(**parameters), data_frame_predicciones)


# ===============================================================================================
#                            METRICAS
# ===============================================================================================
# filter_columns = [
#     column for column in parameters["filter_data"] if re.match(r"filter_\d+_column", column)
# ]
# filter_feature = [
#     column for column in parameters["filter_data"] if re.match(r"filter_\d+_feature", column)
# ]

# value_product = []
# for i in filter_feature:
#     value_product.append(parameters["filter_data"][i])
# fecha = parameters["query_template_write"]["columns"]["0"]
# min_date = data_frame_predicciones[fecha].min()
# max_data = data_frame_predicciones[fecha].max()
# metric_columns_pred["init_date"] = data_frame_predicciones[fecha].min()
# metric_columns_pred["end_date"] = data_frame_predicciones[fecha].max()
# metric_columns_pred["product"] = "/".join(value_product)

# type_data_out = {
#     "rango": "float",
#     "varianza": "float",
#     "desviacion_estandar": "float",
#     "coeficiente_varianza": "float",
#     "quantile_q1": "float",
#     "quantile_q3": "float",
#     "interquantile": "float",
#     "desviacion_media_absoluta": "float",
#     "init_date": "date",
#     "end_date": "date",
#     "product": "string",
# }

# fix_data_dict = {
#     "table": "public.predicciones_metricas",
#     "columns": {str(index): key for index, key in enumerate(type_data_out.keys())},
#     "order": "index",
#     "where": "posicion > 1",
# }

# parameters["query_template_write"] = fix_data_dict
# parameters["type_data_out"] = type_data_out

# create_table(SQLDataSourceFactory(**parameters))
# send_metrics = pd.DataFrame([metric_columns_pred])
# set_data(SQLDataSourceFactory(**parameters), send_metrics)

# ===============================================================================================
#                             DATOS REALES MESES
# ===============================================================================================
logger.debug("Agrupando datos reales por perido de tiempo a las predicciones")
item = filter_label
logger.debug(
    "Agrupando datos reales por perido de tiempo a las predicciones para el modelo : %s",
    item,
)

date_col = parameters["filter_data"]["date_column"]
data_col = parameters["filter_data"]["predict_column"]

outlines_data.reset_index(inplace=True)
outlines_data['low_past'] = filter_label
outlines_data.rename({"low_past": filter_col}, axis="columns", inplace=True)

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    parameters = yaml.safe_load(file)

logger.debug("Archivo de configuraciones cargado")
parametros = Parameters(**parameters)

parameters["query_template_write"]["table"] = "public.datos_orignales_agrupados"
parameters["query_template_write"]["columns"]["0"] = "fecha"
parameters["query_template_write"]["columns"]["1"] = "real"
parameters["query_template_write"]["columns"]["2"] = "code"
parameters["type_data_out"] = {"fecha": "date", "real": "float", "code": "string"}

# Crear tabla para guardas la informacion
logger.debug("Creando tabla agrupacion de datos reales semanales caso de ser necesario")
create_table(SQLDataSourceFactory(**parameters))

parameters["query_template"]["table"] = "public.datos_orignales_agrupados"
parameters["query_template"]["columns"]["0"] = "fecha"
parameters["query_template"]["columns"]["1"] = "real"
parameters["query_template"]["columns"]["2"] = "code"
parameters["type_data_out"] = {"fecha": "date", "real": "float", "code": "string"}
del parameters["query_template"]["columns"]["3"]
data_last = get_data(SQLDataSourceFactory(**parameters))


# Condicional para verifical la ultima fecha de los datos almacenados
# En caso de estar vacio rellena con el historial de los datos por meses
if data_last.empty:
    # Ingresar los datos a la base de datos
    logger.debug("agruando informacion temporal para el modelo : %s", item)
    set_data(SQLDataSourceFactory(**parameters), outlines_data)
else:
    LAST_DATE = data_last.iloc[-1, 0]
    outlines_data = outlines_data[outlines_data[date_col] > np.datetime64(LAST_DATE)]

    # Ingresar los datos a la base de datos
    logger.debug("agruando informacion temporal para el modelo : %s", item)
    set_data(SQLDataSourceFactory(**parameters), outlines_data)

# ===============================================================================================
#                            ORIGINAL  METRICAS DATA
# ===============================================================================================
logger.debug("Calculando metricas de datos reales")
type_data_out = {
    "rango": "float",
    "mean": "float",
    # "varianza": "float",
    # "desviacion_estandar": "float",
    # "coeficiente_varianza": "float",
    "quantile_q0": "float",
    "quantile_q1": "float",
    "quantile_q3": "float",
    "quantile_q4": "float",
    "interquantile": "float",
    # "desviacion_media_absoluta": "float",
    "init_date": "date",
    "end_date": "date",
    "product": "string",
}

fix_data_dict = {
    "table": "public.datos_originales_metricas",
    "columns": {str(index): key for index, key in enumerate(type_data_out.keys())},
    "order": "index",
    "where": "posicion > 1",
}

parameters["query_template_write"] = fix_data_dict
parameters["type_data_out"] = type_data_out

create_table(SQLDataSourceFactory(**parameters))

fix_data_dict = {
    "table": "public.datos_originales_metricas",
    "columns": {str(index): key for index, key in enumerate(type_data_out.keys())},
    "order": "index",
    "where": "index > 1",
}

parameters["query_template"] = fix_data_dict
parameters["type_data_out"] = type_data_out
data_original_metricas = get_data(SQLDataSourceFactory(**parameters))

if data_original_metricas.empty:

    df = data_.dataframe
    # Convertir la columna 'created_at' a tipo datetime
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Agregar columnas de mes y año
    df['month'] = df['created_at'].dt.month
    df['year'] = df['created_at'].dt.year

    # Calcular estadísticas por mes
    monthly_stats = df.groupby(['year', 'month']).agg(
        rango=('quantity', lambda x: x.max() - x.min()),
        # varianza=('quantity', 'var'),
        # desviacion_estandar=('quantity', 'std'),
        # coeficiente_varianza=('quantity', lambda x: x.std() / x.mean()),
        mean=('quantity', 'mean'),
        quantile_q0=('quantity', lambda x: x.quantile(0)),
        quantile_q1=('quantity', lambda x: x.quantile(0.25)),
        quantile_q3=('quantity', lambda x: x.quantile(0.75)),
        quantile_q4=('quantity', lambda x: x.quantile(1)),
        interquantile=('quantity', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        # desviacion_media_absoluta=('quantity', lambda x: (x - x.mean()).abs().mean())
    ).reset_index()

    # Convertir año y mes a fecha inicial y final del mes
    monthly_stats['init_date'] = pd.to_datetime(monthly_stats[['year', 'month']].assign(day=1))
    monthly_stats['end_date'] = pd.to_datetime(monthly_stats[['year', 'month']].assign(
        day=pd.DatetimeIndex(pd.to_datetime(monthly_stats['init_date'])).days_in_month))

    # Eliminar columnas de año y mes
    monthly_stats.drop(columns=['year', 'month'], inplace=True)

    # Agregar la columna 'product' al DataFrame resultante
    monthly_stats['product'] = df['product'].iloc[0]

    # Reordenar las columnas según el tipo de datos
    monthly_stats = monthly_stats[['rango',
                                   'mean',
                                   #    'varianza',
                                   #    'desviacion_estandar',
                                   #    'coeficiente_varianza',
                                   'quantile_q0',
                                   'quantile_q1',
                                   'quantile_q3',
                                   'quantile_q4',
                                   'interquantile',
                                   #    'desviacion_media_absoluta',
                                   'init_date',
                                   'end_date',
                                   'product']]

    set_data(SQLDataSourceFactory(**parameters), monthly_stats)


else:

    df = data_.dataframe[data_.dataframe[date_col] > np.datetime64(LAST_DATE)]

    # Convertir la columna 'created_at' a tipo datetime
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Agregar columnas de mes y año
    df['month'] = df['created_at'].dt.month
    df['year'] = df['created_at'].dt.year

    # Calcular estadísticas por mes
    monthly_stats = df.groupby(['year', 'month']).agg(
        rango=('quantity', lambda x: x.max() - x.min()),
        # varianza=('quantity', 'var'),
        # desviacion_estandar=('quantity', 'std'),
        # coeficiente_varianza=('quantity', lambda x: x.std() / x.mean()),
        mean=('quantity', 'mean'),
        quantile_q0=('quantity', lambda x: x.quantile(0)),
        quantile_q1=('quantity', lambda x: x.quantile(0.25)),
        quantile_q3=('quantity', lambda x: x.quantile(0.75)),
        quantile_q4=('quantity', lambda x: x.quantile(1)),
        interquantile=('quantity', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        # desviacion_media_absoluta=('quantity', lambda x: (x - x.mean()).abs().mean())
    ).reset_index()

    # Convertir año y mes a fecha inicial y final del mes
    monthly_stats['init_date'] = pd.to_datetime(monthly_stats[['year', 'month']].assign(day=1))
    monthly_stats['end_date'] = pd.to_datetime(monthly_stats[['year', 'month']].assign(
        day=pd.DatetimeIndex(pd.to_datetime(monthly_stats['init_date'])).days_in_month))

    # Eliminar columnas de año y mes
    monthly_stats.drop(columns=['year', 'month'], inplace=True)

    # Agregar la columna 'product' al DataFrame resultante
    monthly_stats['product'] = df['product'].iloc[0]

    # Reordenar las columnas según el tipo de datos
    monthly_stats = monthly_stats[['rango',
                                   'mean',
                                   #    'varianza',
                                   #    'desviacion_estandar',
                                   #    'coeficiente_varianza',
                                   'quantile_q0',
                                   'quantile_q1',
                                   'quantile_q3',
                                   'quantile_q4',
                                   'interquantile',
                                   #    'desviacion_media_absoluta',
                                   'init_date',
                                   'end_date',
                                   'product']]

    set_data(SQLDataSourceFactory(**parameters), monthly_stats)
