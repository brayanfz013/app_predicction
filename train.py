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
import numpy as np
import pandas as pd
import yaml
import logging
from scipy import stats
from scipy import signal
from pathlib import Path
from datetime import datetime
from src.features.features_redis import HandleRedis
from src.data.save_models import SAVE_DIR
from src.lib.class_load import LoadFiles
from src.lib.factory_data import SQLDataSourceFactory, get_data, create_table, set_data
from src.lib.factory_models import ModelContext
from src.lib.utils import select_best_model
from src.lib.factory_prepare_data import (
    DataCleaner,
    DataModel,
    MeanImputation,
    OutliersToIQRMean,
    PrepareDtypeColumns,
    base_dtypes
)
from src.models.args_data_model import Parameters, modelos_list_used
from src.data.logs import LOGS_DIR

handler_load = LoadFiles()
handler_redis = HandleRedis()
ruta_actual = os.path.dirname(__file__)

# =================================================================
#             Configuracion Logger
# =================================================================
# Configura un logger personalizado en lugar de usar el logger raíz
logfile = ruta_actual + "/src/data/config/logging.conf"
logging.config.fileConfig(os.path.join(LOGS_DIR, logfile))
logger = logging.getLogger("train")
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
            ['"' + columna + '"' for columna in [date_col_query]])

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
# Nuevos datos para reemplazar en las columnas
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
status_train = {
    "code": parameters['filter_data']['filter_1_feature'],
    "model_select": "",
    "data_time_gap": False,
    "data_low_amount": False,
    "data_abandoned": False,
}
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

status_train['code'] = parameters['filter_data']['filter_1_feature']

# Condicion de filtrado para informacion segun los valores
filter_label: str = parameters["filter_data"]["filter_1_feature"]
filter_col: str = parameters["filter_data"]["filter_1_column"]
filter_product = data_.dataframe[filter_col] == filter_label
filter_data = data_.dataframe[filter_product].sort_values(
    by=parameters["filter_data"]["date_column"])

# Segmento de codigo para filtrado del datos obsoletos
filte_date_col: str = parameters["filter_data"]["date_column"]
filter_data['year'] = filter_data[filte_date_col].dt.year
year = filter_data.groupby('year').size().to_frame()
consecutive_year = year.reset_index()['year'].diff().to_frame()
index_gap_year = consecutive_year[consecutive_year['year'] > 1]

# Condicional para verificar que saltos de tiempo en anos
if not index_gap_year.empty:
    index_gap_year = index_gap_year.sort_values(
        by='year', ascending=False).head(1).index.values[0]
    status_train['data_time_gap'] = True
    remove_year_before = year.reset_index()['year'].iloc[index_gap_year]
    filter_data = filter_data[filter_data['year'] >= remove_year_before]

# Vericar si no hay ingesta de datos en el ultimo ano
last_year_data = year.reset_index()['year'].values[-1]
if (datetime.now().year - last_year_data) >= 2:
    status_train['data_abandoned'] = True


outliners = OutliersToIQRMean(**parameters)
cleaner.strategy = outliners
outlines_data = cleaner.clean(filter_data)
# validate_outlines = cleaner.clean(validate_data)

# Verificar la cantidad de datos para entrenamiento
if outlines_data.shape[0] < 40:
    status_train['data_low_amount'] = True


# Filtrado de datos para eliminar valores negativos
filter_values = outlines_data["quantity"] <= 0
outlines_data[filter_values] = 0.1

# =================================================================
#             Filtro pasabajos
# =================================================================
fs = 1 / 24 / 3600  # 1 day in Hz (sampling frequency)

nyquist = fs / 0.5  # 2 # 0.5 times the sampling frequency
cutoff = 0.5  # 0.1 fraction of nyquist frequency, here  it
# print("cutoff= ", 1 / cutoff * nyquist * 24 * 3600, " days")
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
#            Preparacion de modelo
# =================================================================

metric_eval: dict = {}

for selected_model in modelos_list_used:
    try:
        MODE_USED = selected_model
        print(f"Entrenando {selected_model}")

        modelo = ModelContext(
            model_name=MODE_USED,
            data=data_ready,
            split=80,
            covarianze=data_ready_lp,
            **parameters
        )

        # Entrenar el modelo
        model_trained = modelo.train()

        # Optimizar los parametros del modelo
        if parameters["optimize"]:
            print("Optimizando parametros del modelo")
            model_trained = modelo.optimize()

        # Guargar los modelos entrenados
        modelo.save(model_trained, scaler=scaler_data, scaler_name="scaler")
        modelo.save(None, scaler=scaler_data_lp, scaler_name="scaler_lp")

        print("metodo finalizado")
    except Exception as dataerror:
        status_train['data_low_amount'] = True

# =================================================================
#            Seleccionar el mejor modelo
# =================================================================
save_dir = Path(SAVE_DIR).joinpath(
    parameters["filter_data"]["filter_1_feature"])
models_metrics = save_dir.joinpath(
    "train_metrics").with_suffix(".json").as_posix()

MODE_USED = select_best_model(models_metrics)

status_train['model_select'] = MODE_USED

# =================================================================
#            Enviar metricas a la DB
# =================================================================

parameters["query_template_write"]["table"] = "estados_entrenamientos"
parameters["query_template_write"]["columns"]["0"] = "code"
parameters["query_template_write"]["columns"]["1"] = "model_select"
parameters["query_template_write"]["columns"]["2"] = "data_time_gap"
parameters["query_template_write"]["columns"]["3"] = "data_low_amount"
parameters["query_template_write"]["columns"]["4"] = "data_abandoned"
parameters["type_data_out"] = {
    "code": "string",
    "model_select": "string",
    "data_time_gap": "bool",
    "data_low_amount": "bool",
    "data_abandoned": "bool"
}

create_table(SQLDataSourceFactory(**parameters))
set_data(SQLDataSourceFactory(**parameters),
         pd.DataFrame(status_train, index=[0]))
