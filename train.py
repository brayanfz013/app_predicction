'''Codigo para la ejecucion  y llamando para hacer predicciones en los modelos'''


# from app_prediction.src.lib.factory_data import client_code, SQLDataSourceFactory, NoSQLDataSourceFactory,PlainTextFileDataSourceFactory
import json
from scipy import stats
import numpy as np
import pandas as pd
from src.lib.factory_data import get_data, SQLDataSourceFactory
from src.lib.factory_models import ModelContext, Modelos
from src.lib.factory_prepare_data import DataCleaner,MeanImputation,OutliersToIQRMean
from src.models.args_data_model import (
    ModelRNN,
    ModelBlockRNN,
    ModelExponentialSmoothing,
    ModelTCNModel,
    ModelFFT,
    ModelTransformerModel,
    ModelNBEATSModel,
    ModelDLinearModel,
    ModelNlinearModel,
    ModelTFTModel
)

#=================================================================
#             Cargar datos de la fuente de datos 
#=================================================================
CONFIG_FILE = "/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/parameter/data_params_run.json"
FILE_FILTER_DATA = "/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/querys/filter_data.json"


with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = json.load(file)

print("Probando el estacion de datos de sql")
data = get_data(SQLDataSourceFactory(**parameters))

#=================================================================
#             Limpieza de datos
#=================================================================
# Nuevos datos para reemplazar en las columnas
new_types =[np.datetime64,int,int,'object',int,int]


#metodo para transformar los tipo de datos
strategy = {
    int:np.mean,
    float:np.mean,
    object:stats.mode
}

#Estrategias para imputar los datos faltantes de NA 
replace = {
    int:lambda x: int(float(x.replace(',',''))),
    float:lambda x: float(x.replace(',',''))
}



imputation = MeanImputation(
                            replace_dtypes=new_types,
                            strategy_imputation=strategy,
                            preprocess_function=replace,
                            **parameters
                            )


outliners = OutliersToIQRMean(
    **parameters
)

# cleaner = DataCleaner(imputation)
# data_clean = cleaner.clean(data)

# cleaner.strategy = outliners
# cleaner.clean(data_clean)

#=================================================================
#            Preparacion de modelo
#=================================================================
# parameters_train = {
#     "model_name":'Test_model',
#     "model": "LSTM",
#     "hidden_dim": 20,
#     "dropout": 0,
#     "batch_size": 16,
#     "n_epochs": 300,
#     "optimizer_kwargs":{"lr": 1e-3},
#     "log_tensorboard": True,
#     "random_state": 42,
#     "training_length": 20,
#     "input_chunk_length":14,
#     "force_reset":True,
#     "save_checkpoints":True
# }
# # print(ModelRNN.__dict__)


# print(ModelRNN(**parameters_train))

# # print(ModelContext(['RNNModel'],ModelRNN))
# print('metodo finalizado')


#=================================================================
#             Guardado de informacion
#=================================================================