import torch
from darts.metrics import smape
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import optuna
import tqdm
from optuna.integration import PyTorchLightningPruningCallback
import numpy as np
import pandas as pd

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
    'RNNModel': RNNModel,
    'BlockRNNModel': BlockRNNModel,  # Complejo
    'NBeatsModel': NBEATSModel,
    'TCNModel': TCNModel,  # Muchos datos detalles
    'TransformerModel': TransformerModel,
    'TFTModel': TFTModel,  # Correciones
    'DLinealModel': DLinearModel,
    'NLinearModel': NLinearModel,
    'ExponentialSmoothing': ExponentialSmoothing,  # Malo
    'FFT': FFT  # Malo
}

Parameters_model = {
    'RNNModel': ModelRNN,
    'BlockRNNModel': ModelBlockRNN,
    'NBeatsModel': ModelNBEATSModel,
    'TCNModel': ModelTCNModel,
    'TransformerModel': ModelTransformerModel,
    'TFTModel': ModelTFTModel,
    'DLinealModel': ModelDLinearModel,
    'NLinearModel': ModelNlinearModel,
    # 'ExponentialSmoothing':ModelExponentialSmoothing,
    # 'FFT':ModelFFT
}


NR_DAYS = 80
DAY_DURATION = 24 * 4  # 15 minutes frequency

# all_series_fp32 = [
#     s[-(NR_DAYS * DAY_DURATION) :].astype(np.float32) for s in tqdm(all_series)
# ]


class ModelHyperparameters:
    '''
    Metodo para hiperparametrizar modelos y buscar sus parametros
    '''

    def __init__(self, model_name, data, split) -> None:
        # Nombre del modelo
        self.model_name = model_name

        # Datos del modelo
        self.data = data
        temp_df  = data.pd_dataframe().reset_index()
        # Separacion de datos para entrenamiento
        percent_split = int(temp_df.shape[0] * split/100)
        self.split_value =temp_df.iloc[percent_split].values[0]

        self.train, self.val = self.data.split_after(
            pd.Timestamp(self.split_value.strftime('%Y%m%d')))

        # Cargar los parametros del modelo seleccionado
        model_dict = Parameters_model[self.model_name]
        self.model_used_parameters = model_dict().__dict__

    def build_fit_model(self, likelihood=None, callbacks=None):
        '''Metodo para entrenar y prepara datos de los modelos'''

        torch.manual_seed(42)

        # throughout training we'll monitor the validation loss for early stopping
        early_stopper = EarlyStopping(
            "val_loss", min_delta=0.001, patience=3, verbose=True)
        if callbacks is None:
            callbacks = [early_stopper]
        else:
            callbacks = [early_stopper] + callbacks

        # detect if a GPU is available
        if torch.cuda.is_available():
            pl_trainer_kwargs = {
                "accelerator": "gpu",
                # "gpus": -1,
                # "auto_select_gpus": True,
                'enable_progress_bar': True,
                # "callbacks": callbacks,
            }
            num_workers = 4
        else:
            pl_trainer_kwargs = {"callbacks": callbacks}
            num_workers = 0

        # add_encoders={
        # 'cyclic': {'past': ['month']},
        # # 'datetime_attribute': {'future': ['hour', 'dayofweek']},
        # # 'position': {'past': ['relative'], 'future': ['relative']}
        # }

        # optionally also add the day of the week (cyclically encoded) as a past covariate
        # encoders = {"cyclic": {"past": ["dayofweek"]}} if include_dayofweek else None

        if self.model_name != 'FFT':
            self.model_used_parameters['pl_trainer_kwargs'] = pl_trainer_kwargs
            # self.model_used_parameters['add_encoders'] = add_encoders

        inst_model = Modelos[self.model_name]
        model_prepare = inst_model(**self.model_used_parameters)

        # when validating during training, we can use a slightly longer validation
        # set which also contains the first input_chunk_length time steps
        # model_val_set = scaler.transform(
        #     [s[-((2 * self.val_len) + in_len) : -self.val_len] for s in all_series_fp32]
        # )

        # train the model
        MAX_SAMPLES_PER_TS = 10
        model_prepare.fit(
            series=self.train,
            val_series=self.val,
            max_samples_per_ts=MAX_SAMPLES_PER_TS,
            num_loader_workers=num_workers,
            verbose=True
        )
        # reload best model over course of training
        model_prepare = inst_model.load_from_checkpoint(
            self.model_used_parameters['model_name'])

        return model_prepare

    def objective(self, trial):
        '''Metodo encargado de cargar la busqueda de los parametros'''
        callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]

        self.model_used_parameters['pl_trainer_kwargs']['callbacks'] = callback

        days_in = 30
        self.model_used_parameters['input_chunk_length'] = trial.suggest_int(
            "input_chunk_length", 5, 10)
        self.model_used_parameters['output_chunk_length'] = trial.suggest_int(
            "output_chunk_length", 1, 5)
        self.model_used_parameters['num_blocks'] = trial.suggest_int(
            "num_blocks", 1, 5)
        # self.model_used_parameters['num_layers']  = trial.suggest_int("num_layers", 1, 5) #CUIDADO CON ESTE PARAMETROS
        self.model_used_parameters['num_stacks'] = trial.suggest_int(
            "num_stacks", 1, 10)
        # self.model_used_parameters['kernel_size']  = trial.suggest_int("kernel_size", 5, 25)
        # self.model_used_parameters['num_filters'] = trial.suggest_int("num_filters", 5, 25)
        # self.model_used_parameters['weight_norm'] = trial.suggest_categorical("weight_norm", [False, True])
        # self.model_used_parameters['dilation_base'] = trial.suggest_int("dilation_base", 2, 4)
        self.model_used_parameters['generic_architecture'] = trial.suggest_categorical(
            "generic_architecture", [False, True])
        self.model_used_parameters['dropout'] = trial.suggest_float(
            "dropout", 0.0, 0.4)
        self.model_used_parameters['optimizer_kwargs']['lr'] = trial.suggest_float(
            "lr", 5e-5, 1e-3, log=True)
        # self.model_used_parameters['include_dayofweek'] = trial.suggest_categorical("dayofweek", [False, True])
        model = self.build_fit_model(self.model_name)

        # Evaluate how good it is on the validation set
        preds = model.predict(series=self.train, n=len(self.val))
        smapes = smape(self.val, preds, n_jobs=-1, verbose=True)
        smape_val = np.mean(smapes)

        return smape_val if smape_val != np.nan else float("inf")

    def print_callback(study, trial):
            print(f"Current value: {trial.value}, Current params: {trial.params}")
            print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

    # def 
    #     study = optuna.create_study(direction="minimize")
    #     study.optimize(hyperparametrizacion.objective, timeout=7200, callbacks=[print_callback])

    #     # We could also have used a command as follows to limit the number of trials instead:
    #     # study.optimize(objective, n_trials=100, callbacks=[print_callback])

    #     # Finally, print the best value and best hyperparameters:
    #     print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")