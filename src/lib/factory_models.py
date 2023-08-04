'''Metodo de diseno de modelos empleando el patron de diseno basado en Estrategia '''

import os
from abc import ABC, abstractmethod
from pathlib import Path

import darts
import pandas as pd
from darts.models import (FFT, BlockRNNModel, DLinearModel,
                          ExponentialSmoothing, NBEATSModel, NLinearModel,
                          RNNModel, TCNModel, TFTModel, TransformerModel)

try:
    from src.models.args_data_model import Parameters
    from src.data.save_models import SAVE_DIR
    from src.lib.class_load import LoadFiles
    from src.models.DP_model import ModelHyperparameters, Modelos
except ImportError:
    from args_data_model import Parameters
    from class_load import LoadFiles
    from DP_model import ModelHyperparameters, Modelos
    from save_models import SAVE_DIR


class Model(ABC):
    '''Extraccion de caracteristicas base de los modelos de predicciones'''
    @abstractmethod
    def train(self):
        '''Metodo base de entrenamiento de fabrica de modelo'''

    @abstractmethod
    def predict(self, model, data, horizont):
        '''Metodo base para hacer predicciones'''

    @abstractmethod
    def optimize(self):
        '''Metodo base para calcular error de las predicciones'''

    # @abstractmethod
    # def parameters(self,parameter):
    #     '''Metodo base para cargar los parametros de los modelo'''

    @abstractmethod
    def save(self, model, scaler):
        '''metodo base para guardar tanto el modelo generado '''

    @abstractmethod
    def load(self):
        '''Metodo base para cargar los parametros del modelo y el modelo si existe'''


class ModelContext(Model):
    '''Metodo de abstraccion de modelos y sus metodos'''

    def __init__(self, model_name, data, split, **parameters):
        self.handle_loader = LoadFiles()
        self.save_path = Path(SAVE_DIR)
        self.parameters = parameters

        for filter_list in parameters['filter_data']:
            if 'feature' in filter_list:
                self.save_path = self.save_path.joinpath(
                    str(self.parameters['filter_data'][filter_list]))

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path.as_posix())

        if model_name not in list(Modelos.keys()):
            raise ValueError(f"Modelo no soportado: {model_name}")
        else:
            print(f'Modelo importado {model_name}')
        self.tunne_parameter = ModelHyperparameters(model_name, data, split)

    def train(self):
        '''Ejecutar entrenamiento'''
        model = self.tunne_parameter.build_fit_model()
        return model

    def predict(self, model: str, data: darts.TimeSeries, horizont: int):
        '''Ejecutar una prediccion'''

        # Carga la ultima fecha de prediccion
        last_train = self.handle_loader.json_to_dict(
            self.save_path.joinpath('previus').with_suffix('.json'))[0]

        # Parter los datos en funcion de la ultima fecha de prediccion
        past, _ = data.split_after(
            split_point=pd.Timestamp(last_train['last_date_pred'])
            )
        
        pred_series = model.predict(series=past, n=horizont,past_covariates = data)

        # Toma la ultima posicion para hacer predicciones
        filter_data = pred_series.pd_dataframe().reset_index(
        )[self.parameters['filter_data']['date_column']]

        last_train = {
            'last_date_pred': pd.Timestamp(filter_data.tail(1).values[0]).strftime('%Y-%m-%d')
        }
        # Guardar la ultima prediccion
        self.handle_loader.save_dict_to_json(
            last_train, self.save_path, 'previus')

        return pred_series

    def optimize(self):
        '''Metodo para generar optimizacion de datos'''
        optimizer = self.tunne_parameter.optimize()
        self.tunne_parameter.update_parameters(optimizer)
        model = self.tunne_parameter.build_fit_model()
        return model

    # def parameters(self,parameters:dict):
    #     '''Metodo para cargar los parametros de entrenamiento'''
    #     self.tunne_parameter.update_parameters()

    def save(self, model: darts.models, scaler):
        '''Guardar modelo en ruta'''

        last_train = {
            'last_date_pred': self.tunne_parameter.last_value.strftime('%Y-%m-%d')
            # La linea inferior se usa cuando se quiere hacer predicciones sobre data anterior
            # 'last_date_pred': self.tunne_parameter.split_value.strftime('%Y-%m-%d')
        }

        # Guardar la ultima prediccion
        self.handle_loader.save_dict_to_json(
            last_train, self.save_path, 'previus')

        # Guardar el escalizador del modelo
        scaler_save_folder = self.save_path.joinpath(
            'scaler').with_suffix('.pkl')
        self.handle_loader.save_scaler(scaler, scaler_save_folder)

        # Guardar los parametros de entrenamiento
        self.handle_loader.save_dict_to_json(
            file_name='parametros',
            dict_data=self.tunne_parameter.model_used_parameters,
            path_to_save=self.save_path.as_posix()
        )
        
        #Guardar el archivo de configuraciones de yaml
        file_name = self.parameters['filter_data']['filter_1_feature']
        self.handle_loader.save_yaml(
            self.parameters, 
            self.save_path.joinpath('config_'+file_name).with_suffix('.yaml').as_posix()
            )

        # Guardar el modelo
        model.save(str(self.save_path.joinpath('model').with_suffix('.pt')))

    def load_dirmodel(self):
        '''Cargar el listado de datos de archivos generados en el entrenamiento'''
        return [child.as_posix() for child in self.save_path.iterdir()]

    def load(self):
        '''Cargar los parametros del modelo'''
        # Listado de archivos en el directorio
        # for child in self.save_path.iterdir(): print(child)
        # scaler_loaded = self.handle_loader.load_scaler(scaler_save_folder)


# context = ModelContext("Modelos")
# context.train(data)
# predictions = context.predict(data)
