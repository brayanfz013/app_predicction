from darts.models import RandomForest, LightGBMModel, LinearRegressionModel, CatBoostModel, XGBModel, BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, TransformerModel, TFTModel, DLinearModel, NLinearModel
from abc import ABC, abstractmethod

Modelos = {
    'RandomForest':RandomForest,
    'LinearRegressionModel':LightGBMModel,
    'LightGBMModel':LinearRegressionModel,
    'CatBoostModel':CatBoostModel,
    'XGBModel':XGBModel,
    'BlockRNNModel':BlockRNNModel,
    'NBeatsModel':NBEATSModel,
    'NHiTSModel':NHiTSModel,
    'TCNModel':TCNModel,
    'TransformersModel':TransformerModel,
    'TFTModel': TFTModel,
    'DLinealModel':DLinearModel,
    'NLinearModel':NLinearModel
}

class Model(ABC):
    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def error(self, data):
        pass

    @abstractmethod
    def parameters(self, parameters):
        pass


class RandomForestModel(Model):
    def train(self, data):
        print("Entrenando RandomForestModel con los datos")

    def predict(self, data):
        print("Realizando predicciones con RandomForestModel")


class ModelContext:
    def __init__(self, model_name):
        if model_name not in Modelos:
            raise ValueError(f"Modelo no soportado: {model_name}")

        self._model = Modelos[model_name]()

    def train(self, data):
        self._model.fit(data)

    def predict(self, data):
        return self._model.predict(data)
    

context = ModelContext('RandomForest')
context.train('data')
predictions = context.predict('data')

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
