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
    NLinearModel)

try:
    from src.features.features_fix_data import PrepareData
    from src.lib.class_load import LoadFiles
    from src.models.args_data_model import (
        ModelBlockRNN,
        ModelExponentialSmoothing,
        ModelDLinearModel,
        ModelNlinearModel,
        ModelRNN,
        ModelNBEATSModel,
        ModelFFT,
        ModelTCNModel,
        ModelTFTModel,
        ModelTransformerModel
    )
except ImportError:
    from features_fix_data import PrepareData
    from class_load import LoadFiles
    from args_data_model import (
        ModelBlockRNN,
        ModelExponentialSmoothing,
        ModelDLinearModel,
        ModelNlinearModel,
        ModelRNN,
        ModelNBEATSModel,
        ModelFFT,
        ModelTCNModel,
        ModelTFTModel,
        ModelTransformerModel
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
    'FFT':FFT
}

Parameters_model = {
    'RNNModel':ModelRNN,
    'BlockRNNModel': ModelBlockRNN,
    'NBeatsModel': ModelNBEATSModel,
    'TCNModel': ModelTCNModel,
    'TransformersModel': ModelTransformerModel,
    'TFTModel': ModelTFTModel,
    'DLinealModel': ModelDLinearModel,
    'NLinearModel': ModelNlinearModel,
    'ExponentialSmoothing':ModelExponentialSmoothing,
    # 'AutoARIMA':AutoARIMA,
    # 'Theta':Theta,
    # 'VARIMA':VARIMA,
    'FFT':ModelFFT
}

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