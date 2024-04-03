import numpy as np
import optuna
import pandas as pd
import torch

# import tqdm
# from pathlib import Path
from darts.metrics import mape, r2_score, smape
from darts.models import (
    # FFT,
    BlockRNNModel,
    DLinearModel,
    # ExponentialSmoothing,
    NBEATSModel,
    NLinearModel,
    RNNModel,
    TCNModel,
    TFTModel,
    TransformerModel,
)
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.timeseries import TimeSeries

try:
    # from src.features.features_fix_data import PrepareData
    # from src.lib.class_load import LoadFiles
    # from src.data.save_models import SAVE_DIR
    from src.models.args_data_model import (
        ModelBlockRNN,
        ModelDLinearModel,
        # ModelExponentialSmoothing,
        # ModelFFT,
        ModelNBEATSModel,
        ModelNlinearModel,
        ModelRNN,
        ModelTCNModel,
        ModelTFTModel,
        ModelTransformerModel,
    )
except ImportError:
    from args_data_model import (
        ModelBlockRNN,
        ModelDLinearModel,
        # ModelExponentialSmoothing,
        # ModelFFT,
        ModelNBEATSModel,
        ModelNlinearModel,
        ModelRNN,
        ModelTCNModel,
        ModelTFTModel,
        ModelTransformerModel,
    )

    # from class_load import LoadFiles
    # from features_fix_data import PrepareData
    # from save_models import SAVE_DIR

Modelos = {
    "RNNModel": RNNModel,
    "BlockRNNModel": BlockRNNModel,  # Complejo
    "NBeatsModel": NBEATSModel,
    "TCNModel": TCNModel,  # Muchos datos detalles
    "TransformerModel": TransformerModel,
    "TFTModel": TFTModel,  # Correciones
    "DLinealModel": DLinearModel,
    "NLinearModel": NLinearModel,
    # 'ExponentialSmoothing': ExponentialSmoothing,  # Malo
    # 'FFT': FFT  # Malo
}

Parameters_model = {
    "RNNModel": ModelRNN,
    "BlockRNNModel": ModelBlockRNN,
    "NBeatsModel": ModelNBEATSModel,
    "TCNModel": ModelTCNModel,
    "TransformerModel": ModelTransformerModel,
    "TFTModel": ModelTFTModel,
    "DLinealModel": ModelDLinearModel,
    "NLinearModel": ModelNlinearModel,
    # 'ExponentialSmoothing':ModelExponentialSmoothing,
    # 'FFT':ModelFFT
}

modelos_list = [
    "BlockRNNModel",  # Complejo
    "NBeatsModel",
    "TCNModel",  # Muchos datos detalles
    "TransformerModel",
    "TFTModel",  # Correciones
    "DLinealModel",
    "NLinearModel",
]


class ModelHyperparameters:
    """
    Metodo para hiperparametrizar modelos y buscar sus parametros
    """

    def __init__(
        self, model_name: str, data: TimeSeries, split: int, covarianze: TimeSeries = None
    ) -> None:
        # Nombre del modelo
        self.model_name = model_name

        # Datos del modelo
        self.data = data
        temp_df = data.pd_dataframe().reset_index()

        # Separacion de datos para entrenamiento
        percent_split = int(temp_df.shape[0] * split / 100)
        self.split_value = temp_df.iloc[percent_split].values[0]
        self.last_value = temp_df.iloc[-1].values[0]
        self.train, self.val = self.data.split_after(
            pd.Timestamp(self.split_value.strftime("%Y%m%d"))
        )

        # Cargar los parametros del modelo seleccionado
        model_dict = Parameters_model[self.model_name]
        self.model_used_parameters = model_dict().__dict__

        # Datos de covarianza
        if covarianze is not None:

            self.covarianze = covarianze

            # Separacion de datos
            temp_covariance_df = self.covarianze.pd_dataframe().reset_index()
            percent_split = int(temp_covariance_df.shape[0] * split / 100)
            self.split_value_cov = temp_covariance_df.iloc[percent_split].values[0]
            self.last_value_cov = temp_covariance_df.iloc[-1].values[0]
            self.train_cov, self.val_cov = self.covarianze.split_after(
                pd.Timestamp(self.split_value.strftime("%Y%m%d"))
            )
        else:
            self.covarianze = self.data
            self.split_value_cov = self.split_value
            self.last_value_cov = self.last_value
            self.train_cov = self.train
            self.val_cov = self.val

    def build_fit_model(
        self,
        callbacks=None,
        enable_callback: bool = False,
        enable_encoder: bool = False,
        verbose_show: bool = True,
    ):
        """Metodo para entrenar y prepara datos de los modelos"""
        # torch.manual_seed(42)

        early_stopper = EarlyStopping(
            "val_loss", min_delta=0.001, patience=10, verbose=verbose_show
        )
        if callbacks is None:
            callbacks = [early_stopper]
        else:
            callbacks = [early_stopper] + callbacks

        # detect if a GPU is available
        if torch.cuda.is_available():
            if enable_callback:
                pl_trainer_kwargs = {
                    "accelerator": "gpu",
                    "enable_progress_bar": True,
                    "callbacks": callbacks,
                }
            else:
                pl_trainer_kwargs = {
                    "accelerator": "gpu",
                    "enable_progress_bar": True,
                }
            num_workers = 4
        else:
            pl_trainer_kwargs = {"callbacks": callbacks}
            num_workers = 4

        add_encoders_past = {
            "cyclic": {"past": ["month"]},
            "datetime_attribute": {"future": ["hour", "dayofweek"]},
            "position": {"past": ["relative"], "future": ["relative"]},
        }

        encoder_future = {
            "cyclic": {"future": ["weekofyear", "month"]},
            "datetime_attribute": {"future": ["hour", "dayofweek"]},
            "position": {"future": ["relative"]},
        }

        if self.model_name != "FFT":
            self.model_used_parameters["pl_trainer_kwargs"] = pl_trainer_kwargs
            # if enable_encoder:
            # self.model_used_parameters['add_encoders'] = add_encoders

        inst_model = Modelos[self.model_name]
        model_prepare = inst_model(**self.model_used_parameters)

        if self.model_name == "ModelRNN":
            model_prepare.fit(series=self.train)

        else:
            # train the model
            MAX_SAMPLES_PER_TS = 10  # parametro por optimizar
            model_prepare.fit(
                series=self.train,
                past_covariates=self.train_cov,
                val_series=self.val,
                val_past_covariates=self.val_cov,
                # max_samples_per_ts=MAX_SAMPLES_PER_TS,
                num_loader_workers=num_workers,
                verbose=verbose_show,
            )

        # save_parameters
        # model_prepare = model_prepare.load_from_checkpoint(self.model_used_parameters['model_name'])

        # save_dir = Path(SAVE_DIR).joinpath('model').with_suffix('.pk')
        # model_prepare.save(save_dir)

        return model_prepare

    def optimize(self, trial):
        """
        The `optimize` method is used to perform parameter optimization for a PyTorch Lightning model
        using Optuna, with parameters being suggested within specified ranges and a model being built and
        evaluated based on the R2 score on a validation set.

        Args:
          trial: The `optimize` method is used to perform parameter optimization for a model. The method
        takes a `trial` object as an argument, which is typically used in hyperparameter optimization
        libraries like Optuna or Ray Tune. The `trial` object provides methods for suggesting
        hyperparameters to be evaluated during the optimization

        Returns:
          The `optimize` method is returning the R2 scores calculated based on the predictions made by
        the model on the validation set. If the R2 score is not NaN, it returns the R2 score; otherwise,
        it returns infinity.
        """
        """Metodo encargado de cargar la busqueda de los parametros"""
        callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        # self.model_used_parameters['pl_trainer_kwargs']['callbacks'] = callback

        self.model_used_parameters["input_chunk_length"] = trial.suggest_int(
            "input_chunk_length", 4, 12
        )
        self.model_used_parameters["output_chunk_length"] = trial.suggest_int(
            "output_chunk_length", 1, 4
        )
        # self.model_used_parameters["batch_size"] = trial.suggest_int("batch_size", 1, 1024)
        self.model_used_parameters["optimizer_kwargs"]["lr"] = trial.suggest_float(
            "lr", 5e-5, 1e-3, log=True
        )

        if self.model_name == "NBeatsModel":
            # parametros de busqueda Nbeats
            self.model_used_parameters["num_blocks"] = trial.suggest_int("num_blocks", 1, 10)
            self.model_used_parameters["num_stacks"] = trial.suggest_int("num_stacks", 1, 15)
            self.model_used_parameters["layer_widths"] = trial.suggest_categorical(
                "layer_widths", [128, 256, 512]
            )
            self.model_used_parameters["generic_architecture"] = trial.suggest_categorical(
                "generic_architecture", [False, True]
            )
            self.model_used_parameters["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
            self.model_used_parameters["batch_size"] = trial.suggest_int("batch_size", 1, 1024)

        elif self.model_name == "TCNModel":
            # Parmaetros de busqueda  TCNModel:
            self.model_used_parameters["dilation_base"] = trial.suggest_int("dilation_base", 2, 4)
            self.model_used_parameters["weight_norm"] = trial.suggest_categorical(
                "weight_norm", [False, True]
            )
            self.model_used_parameters["kernel_size"] = trial.suggest_int(
                "kernel_size", 3, self.model_used_parameters["input_chunk_length"] - 1
            )
            self.model_used_parameters["num_filters"] = trial.suggest_int("num_filters", 6, 12)

        elif self.model_name == "BlockRNNModel":
            # Parmaetros de busqueda  BlockRNNModel:
            self.model_used_parameters["hidden_dim"] = trial.suggest_int("hidden_dim", 1, 15)
            self.model_used_parameters["n_rnn_layers"] = trial.suggest_int("n_rnn_layers", 1, 15)
            self.model_used_parameters["batch_size"] = trial.suggest_int("batch_size", 1, 1024)

        elif self.model_name == "TransformerModel":
            # Parmaetros de busqueda  TransformerModel:
            self.model_used_parameters["batch_size"] = trial.suggest_int("batch_size", 1, 1024)
            # self.model_used_parameters['d_model'] = trial.suggest_int("d_model", 4, 32)
            # self.model_used_parameters['nhead'] = trial.suggest_int("nhead", 4, 16)
            self.model_used_parameters["num_encoder_layers"] = trial.suggest_int(
                "num_encoder_layers", 1, 12
            )
            self.model_used_parameters["num_decoder_layers"] = self.model_used_parameters[
                "num_encoder_layers"
            ]
            self.model_used_parameters["dim_feedforward"] = trial.suggest_int(
                "dim_feedforward", 1, 256
            )

        elif self.model_name == "TFTModel":
            # Parmaetros de busqueda  TFTModel:
            self.model_used_parameters["hidden_size"] = trial.suggest_int("hidden_size", 1, 256)
            self.model_used_parameters["lstm_layers"] = trial.suggest_int("lstm_layers", 1, 5)
            self.model_used_parameters["num_attention_heads"] = trial.suggest_int(
                "num_attention_heads", 1, 8
            )
            self.model_used_parameters["batch_size"] = trial.suggest_int("batch_size", 1, 1024)

        elif self.model_name == "DLinealModel" or self.model_name == "NLinearModel":
            # Parmaetros de busqueda  DLinealModel o NLinearModel:
            self.model_used_parameters["batch_size"] = trial.suggest_int("batch_size", 1, 1024)

        # =======================Entrenamiento del modelo=====================================
        model = self.build_fit_model(enable_callback=True)

        # Evaluate how good it is on the validation set
        # preds = model.predict(series=self.train, n=len(self.val))
        # =====================Validate Error=================================================
        preds = model.predict(
            series=self.train,
            n=self.model_used_parameters["output_chunk_length"],
            past_covariates=self.train_cov,
        )

        smapes = smape(self.val, preds, n_jobs=-1, verbose=True)
        smape_val = np.mean(smapes)
        mapes = mape(self.val, preds, n_jobs=-1, verbose=True)
        # r2_scores = r2_score(actual_series=self.val, pred_series=preds)

        # print(f"r2_score: {r2_score}")
        print(f"mape :{mapes}")
        print(f"smapes: {smapes}")

        # return r2_scores if r2_scores != np.nan else float("inf")
        return smape_val if smape_val != np.nan else float("inf")

    def print_callback(self, study, trial):
        """Metodo para mostrar los valores de entrenamiento"""
        print(f"Current value: {trial.value}, Current params: {trial.params}")
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

    def retrain(self):
        """metodo para reentrenar un modelo"""

        # study = optuna.create_study(direction="maximize")
        study = optuna.create_study(
            sampler=optuna.samplers.QMCSampler(), direction="minimize"  # PartialFixedSampler
        )

        # optimizacion por diferencia de tiempo
        # study.optimize(self.optimize, timeout=6000, callbacks=[self.print_callback])

        # optimizacion por numero de intentos
        study.optimize(
            self.optimize,
            n_trials=2,
            # n_jobs=-1,
            callbacks=[self.print_callback],
        )

        # Finally, print the best value and best hyperparameters:
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
        return study

    def update_parameters(self, study):
        """update_parameters Metodo para actulizar parametros despues del entrenamiento
        Args:
            study (_type_): _description_
        """
        for new_data in study.best_trial.params.items():
            key, val = new_data
            if key == "lr":
                self.model_used_parameters["optimizer_kwargs"][key] = val
            else:
                self.model_used_parameters[key] = val
