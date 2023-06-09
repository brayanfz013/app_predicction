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
    @abstractmethod
    def train(self, data):
        '''Metodo base de entrenamiento de fabrica de modelo'''
        
    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def error(self, data):
        pass

    @abstractmethod
    def parameters(self, parameters):
        pass


class ModelContext:
    '''Metodo de abstraccion de modelos y sus metodos'''
    def __init__(self, model_name,parameters):
        self.parameters = parameters
        if model_name not in Modelos:
            raise ValueError(f"Modelo no soportado: {model_name}")
        else:
            print(f'Modelo importado {model_name}' )
        self._model = Modelos[model_name]()

    def train(self, data):
        self._model.fit(data)

    def predict(self, data):
        return self._model.predict(data)


# context = ModelContext(Modelos)
# context.train('data')
# predictions = context.predict('data')

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
