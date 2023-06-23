'''Metodo de diseno de modelos empleando el patron de diseno basado en Estrategia '''

from abc import ABC, abstractmethod
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    TFTModel,
    FFT,
    ExponentialSmoothing,
    DLinearModel,
    NLinearModel
    )

try:
    from src.models.DP_model import ModelHyperparameters, Modelos

except ImportError:
    from DP_model import ModelHyperparameters, Modelos

class Model(ABC):
    '''Extraccion de caracteristicas base de los modelos de predicciones'''
    @abstractmethod
    def train(self, data):
        '''Metodo base de entrenamiento de fabrica de modelo'''

    @abstractmethod
    def predict(self, data):
        '''Metodo base para hacer predicciones'''

    # @abstractmethod
    # def error(self, data):
    #     '''Metodo base para calcular error de las predicciones'''
    #     pass

    @abstractmethod
    def parameters(self, parameters):
        '''Metodo base para cargar los parametros de los modelo'''

    @abstractmethod
    def save(self):
        '''metodo base para guardar tanto el modelo generado '''

class ModelContext(Model):
    '''Metodo de abstraccion de modelos y sus metodos'''
    def __init__(self, model_name,data,split):
        # use_models = Modelos
        # parametros_modelos = Parameters_model
        # self.parameters_model = parameters

        if model_name not in list(Modelos.keys()):
            raise ValueError(f"Modelo no soportado: {model_name}")
        else:
            print(f'Modelo importado {model_name}' )
        # self._model = use_models[model_name]

        self._model = ModelHyperparameters(model_name,data,split)

    def train(self, data):
        '''Ejecutar entrenamiento'''

        self._model.fit(data)

    def parameters(self,parameters):
        '''Metodo para cargar los parametros de entrenamiento'''

    def predict(self, data):
        '''Ejecutar una prediccion'''
        return self._model.predict(data)

    def save(self):
        '''Guardar modelo en ruta'''

# context = ModelContext("Modelos")
# context.train(data)
# predictions = context.predict(data)