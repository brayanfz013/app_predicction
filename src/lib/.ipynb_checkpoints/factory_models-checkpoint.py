"""Metodo de diseno de modelos empleando el patron de diseno basado en Estrategia """

import os
from abc import ABC, abstractmethod
from pathlib import Path

import darts
import darts.models
import pandas as pd
from darts.metrics import mape, r2_score, smape

# from darts.models import (
#     FFT,
#     BlockRNNModel,
#     DLinearModel,
#     ExponentialSmoothing,
#     NBEATSModel,
#     NLinearModel,
#     RNNModel,
#     TCNModel,
#     TFTModel,
#     TransformerModel,
# )


from src.data.save_models import SAVE_DIR
from src.lib.class_load import LoadFiles
from src.models.DP_model import ModelHyperparameters, Modelos


class Model(ABC):
    """Extraccion de caracteristicas base de los modelos de predicciones"""

    @abstractmethod
    def train(self):
        """Metodo base de entrenamiento de fabrica de modelo"""

    @abstractmethod
    def predict(self, model, data, horizont, past_cov):
        """Metodo base para hacer predicciones"""

    @abstractmethod
    def optimize(self):
        """Metodo base para calcular error de las predicciones"""

    @abstractmethod
    def save(self, model, scaler):
        """metodo base para guardar tanto el modelo generado"""

    @abstractmethod
    def load(self):
        """Metodo base para cargar los parametros del modelo y el modelo si existe"""

    @abstractmethod
    def validation(self, model, data, past_cov):
        """Metodo base para ejemplificar un metodo de validacion en los datos"""


class ModelContext(Model):
    """Metodo de abstraccion de modelos y sus metodos"""

    def __init__(
        self,
        model_name: str,
        data: darts.TimeSeries,
        split: int,
        covarianze: darts.TimeSeries = None,
        **parameters,
    ):
        self.handle_loader = LoadFiles()
        self.save_path = Path(SAVE_DIR)
        self.parameters = parameters
        self.best_model: dict = {}

        for filter_list in parameters["filter_data"]:
            if "feature" in filter_list:
                self.save_path = self.save_path.joinpath(
                    str(self.parameters["filter_data"][filter_list])
                )

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path.as_posix())

        if model_name not in list(Modelos.keys()):
            raise ValueError(f"Modelo no soportado: {model_name}")
        else:
            print(f"Modelo importado {model_name}")

        encoder_future = {"cyclic": {"future": ["weekofyear", "month"]}}
        self.tunne_parameter = ModelHyperparameters(
            model_name, data, split, covarianze)
        self.tunne_parameter.model_used_parameters["add_encoders"] = encoder_future

    def train(self):
        """Ejecutar entrenamiento"""
        model = self.tunne_parameter.build_fit_model()

        # Prediccion de los datos pasados
        pred_series = model.historical_forecasts(
            series=self.tunne_parameter.data,
            start=self.tunne_parameter.split_value,
            past_covariates=self.tunne_parameter.covarianze,
            forecast_horizon=self.tunne_parameter.model_used_parameters["output_chunk_length"],
            retrain=False,
            verbose=False,
        )

        # Definir las métricas
        metricas = {
            "MAPE": lambda pred, actual: mape(pred, actual),
            "SMAPE": lambda pred, actual: smape(pred, actual),
            "R2": lambda pred, actual: r2_score(pred, actual),
        }

        # Calcular las métricas
        resultados = {}
        for nombre, metrica in metricas.items():
            resultados[nombre] = metrica(
                pred_series, self.tunne_parameter.data)

        try:
            self.best_model = self.handle_loader.load_json(
                json_file_name="train_metrics", path_to_read=self.save_path.as_posix())
        except FileNotFoundError as no_file:
            print(f"No parameter File, {no_file}")

        self.best_model[self.tunne_parameter.model_name] = {self.tunne_parameter.model_name: resultados
                                                            }

        self.handle_loader.save_dict_to_json(
            file_name="train_metrics",
            dict_data=self.best_model,
            path_to_save=self.save_path.as_posix(),
        )
        return model

    def predict(self, model: darts.models, data: darts.TimeSeries, horizont: int, past_cov: darts.TimeSeries = None):
        """
        This Python function `predict` executes a prediction using a given model on a time series data
        with optional past covariates.

        Args:
          model (darts.models): The `model` parameter in the `predict` method should be an instance of a
        Darts model, such as `darts.models.ARIMA` or `darts.models.ExponentialSmoothing`. This model will 
        be used to make predictions on the provided data.
          data (darts.TimeSeries): The `data` parameter in the `predict` method is used to provide the
        input time series data for making predictions. If `data` is not provided (i.e., it is `None`),
        the method will split the existing data based on the last prediction date and use that as the
        input
          horizont (int): The `horizont` parameter in the `predict` method represents the number of time
        steps into the future for which you want to make predictions. It determines how far ahead in time
        the model should forecast.
          past_cov (darts.TimeSeries): The `past_cov` parameter in the `predict` method is used to
        provide past covariates data for the prediction. Covariates are additional variables that may
        influence the time series data being predicted. In this method, if `past_cov` is not provided as
        an argument, it is extracted

        Returns:
          The function `predict` returns the predicted time series generated by the model for the
        specified horizon.
        """
        # Carga la ultima fecha de prediccion
        last_train = self.handle_loader.json_to_dict(
            self.save_path.joinpath("previus").with_suffix(".json").as_posix())[0]

        if data == None:
            # Parter los datos en funcion de la ultima fecha de prediccion
            past, _ = self.tunne_parameter.data.split_after(
                split_point=pd.Timestamp(last_train["last_date_pred"]))

        else:
            past = data

        if past_cov == None:
            # Parte los datos de la covarianza filtrada en la ultima fecha
            past_cov, _ = self.tunne_parameter.covarianze.split_after(
                split_point=pd.Timestamp(last_train["last_date_pred"])
            )
        else:
            past_cov = past_cov

        pred_series = model.predict(
            series=past,
            n=horizont,
            past_covariates=past_cov,
        )

        # Toma la ultima posicion para hacer predicciones
        filter_data = pred_series.pd_dataframe().reset_index()[
            self.parameters["filter_data"]["date_column"]
        ]

        last_train = {
            "last_date_pred": pd.Timestamp(filter_data.tail(1).values[0]).strftime("%Y-%m-%d")
        }
        # Guardar la ultima prediccion
        self.handle_loader.save_dict_to_json(
            last_train, self.save_path.as_posix(), "previus")

        return pred_series

    def optimize(self):
        """Metodo para generar optimizacion de datos"""
        optimizer = self.tunne_parameter.retrain()
        self.tunne_parameter.update_parameters(study=optimizer)
        model = self.tunne_parameter.build_fit_model()
        return model

    # def parameters(self,parameters:dict):
    #     '''Metodo para cargar los parametros de entrenamiento'''
    #     self.tunne_parameter.update_parameters()

    def save(self, model: darts.models, scaler, scaler_name="scaler"):
        """Guardar modelo en ruta"""

        last_train = {
            "last_date_pred": self.tunne_parameter.last_value.strftime("%Y-%m-%d"),
            # La linea inferior se usa cuando se quiere hacer predicciones sobre data anterior
        }
        # #Eliminar parametro que no se puede serializar
        if (
            self.tunne_parameter.model_name == "TFTModel"
            and self.tunne_parameter.model_used_parameters.get("likelihood")
        ):
            print(self.tunne_parameter.model_used_parameters)
            del self.tunne_parameter.model_used_parameters["likelihood"]

        # Guardar la ultima prediccion
        self.handle_loader.save_dict_to_json(
            last_train, self.save_path.as_posix(), "previus")

        # Guardar el escalizador del modelo
        scaler_save_folder = self.save_path.joinpath(
            scaler_name).with_suffix(".pkl")
        self.handle_loader.save_scaler(scaler, scaler_save_folder)

        # Guardar los parametros de entrenamiento
        self.handle_loader.save_dict_to_json(
            file_name="parametros_" + self.tunne_parameter.model_name,
            dict_data=self.tunne_parameter.model_used_parameters,
            path_to_save=self.save_path.as_posix(),
        )

        # Guardar el archivo de configuraciones de yaml
        file_name = self.parameters["filter_data"]["filter_1_feature"]
        self.handle_loader.save_yaml(
            self.parameters,
            self.save_path.joinpath(
                "config_" + file_name).with_suffix(".yaml").as_posix(),
        )

        # Guardar el modelo
        if model is not None:
            save_model_train = "model_" + self.tunne_parameter.model_name
            model.save(str(self.save_path.joinpath(
                save_model_train).with_suffix(".pt")))

    # def load_dirmodel(self):
    #     """Cargar el listado de datos de archivos generados en el entrenamiento"""
    #     return [child.as_posix() for child in self.save_path.iterdir()]

    def load(self):
        # """Cargar los parametros del modelo"""
        # Listado de archivos en el directorio
        # for child in self.save_path.iterdir(): print(child)
        # scaler_loaded = self.handle_loader.load_scaler(scaler_save_folder)
        """Cargar el listado de datos de archivos generados en el entrenamiento"""
        return [child.as_posix() for child in self.save_path.iterdir()]

    def validation(self, model: darts.models, data: darts.timeseries, past_cov: darts.timeseries):
        """
        Metodo para valida el entrenamiento del modelo realizado predicciones del tiempo pasado
        y del tiempo futuro
        """
        if data == None:
            # Parter los datos en funcion de la ultima fecha de prediccion
            past, _ = self.tunne_parameter.data
        else:
            past = data

        if past_cov == None:
            # Parte los datos de la covarianza filtrada en la ultima fecha
            past_cov, _ = self.tunne_parameter.covarianze
        else:
            past_cov = past_cov

        # =======================================================================================
        #                       PAST VALIDATION
        # =======================================================================================
        # Prediccion de los datos pasados
        pred_series_past = model.historical_forecasts(
            series=past,  # self.tunne_parameter.data,
            start=self.tunne_parameter.split_value,
            past_covariates=past_cov,  # self.tunne_parameter.covarianze,
            # forecast_horizon=self.tunne_parameter.model_used_parameters["output_chunk_length"],
            retrain=False,
            verbose=False,
        )

        # =======================================================================================
        #                       FUTURE VALIDATION
        # =======================================================================================
        pred_series_future = model.predict(
            series=past,
            n=self.parameters['forecast_val'],
            past_covariates=past_cov,
        )

        return pred_series_past, pred_series_future

# context = ModelContext("Modelos")
# context.train(data)
# predictions = context.predict(data)
