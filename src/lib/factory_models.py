'''Metodo de diseno de modelos empleando el patron de diseno basado en Estrategia '''

from abc import ABC, abstractmethod
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    TFTModel,
    AutoARIMA,
    Theta,
    VARIMA,
    FFT,
    ExponentialSmoothing,
    DLinearModel,
    NLinearModel
)

Modelos = {
    'RNNModel':RNNModel,
    'BlockRNNModel': BlockRNNModel,
    'NBeatsModel': NBEATSModel,
    'TCNModel': TCNModel,
    'TransformersModel': TransformerModel,
    'TFTModel': TFTModel,
    'DLinealModel': DLinearModel,
    'NLinearModel': NLinearModel,
    'ExponentialSmoothing':ExponentialSmoothing,
    'AutoARIMA':AutoARIMA,
    'Theta':Theta,
    'VARIMA':VARIMA,
    'FFT':FFT

}

class Model(ABC):
    '''Extraccion de caracteristicas base de los modelos de predicciones'''
    @abstractmethod
    def train(self, data):
        '''Metodo base de entrenamiento de fabrica de modelo'''
        
    @abstractmethod
    def predict(self, data):
        '''Metodo base para hacer predicciones'''
        pass

    # @abstractmethod
    # def error(self, data):
    #     '''Metodo base para calcular error de las predicciones'''
    #     pass

    @abstractmethod
    def parameters(self, parameters):
        '''Metodo base para cargar los parametros de los modelo'''
        pass

    @abstractmethod
    def save(self, parameters):
        '''metodo base para guardar tanto el modelo generado '''

class ModelContext(Model):
    '''Metodo de abstraccion de modelos y sus metodos'''
    def __init__(self, model_name,parameters):
        self.parameters_model = parameters
        
        if model_name not in Modelos:
            raise ValueError(f"Modelo no soportado: {model_name}")
        else:
            print(f'Modelo importado {model_name}' )
        self._model = Modelos[model_name]

    def train(self, data):
        '''Ejecutar entrenamiento'''
        self._model.fit(data)

    def parameters(self,parameters):
        '''Metodo para cargar los parametros de entrenamiento'''
        

    def predict(self, data):
        '''Ejecutar una prediccion'''
        return self._model.predict(data)

    def save(self, parameters):
        '''Guardar modelo en ruta'''

# context = ModelContext("Modelos")
# context.train(data)
# predictions = context.predict(data)

# class ModelContext:
#     def __init__(self, model):
#         self._model = model

#     @property
#     def model(self):
#         return self._model

#     @model.setter
#     def model(self, model):
#         self._model = model

#     def train(self, data):
#         self._model.train(data)

#     def predict(self, data):
#         self._model.predict(data)


# context = ModelContext(RandomForestModel())
# context.train('data')
# context.predict('data')


# context.model = AnotherModel()  # Cambia a otro modelo en tiempo de ejecución
# context.train('data')
# context.predict('data')
