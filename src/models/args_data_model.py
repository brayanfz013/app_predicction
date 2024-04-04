"""
Codigo para almacenas los argumentos de los diferentes
parametros de los modelos
"""

import os
import copy
from pathlib import PosixPath
from typing import Union, Optional
from dataclasses import dataclass, field
from darts.utils.likelihood_models import QuantileRegression


def default_field(obj):
    """Metodo base para tener un dciccionario como metodo dataclass inicial"""
    return field(default_factory=lambda: copy.copy(obj))


modelos_list_used = [
    "BlockRNNModel",
    "NBeatsModel",
    "TCNModel",
    "TransformerModel",
    "TFTModel",
    "DLinealModel",
    "NLinearModel",
]

quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]


@dataclass
class ParamsPostgres:
    """Listado de parametros para ejecutar HandleDBsql"""

    connection_params: Optional[Union[str, os.PathLike, PosixPath]]
    query_read: Optional[Union[str, os.PathLike, PosixPath]]
    query_write: Optional[Union[str, os.PathLike, PosixPath]]
    logger_file: Optional[Union[str, os.PathLike, PosixPath]]
    names_table_columns: Optional[Union[str, os.PathLike, PosixPath]]
    filter_columns: Optional[Union[str, os.PathLike, PosixPath]]


@dataclass
class AlertsData:
    """
    The class "AlertsData" represents different types of alerts
    and their corresponding integer values.
    """

    alerta_bajacantidad: int
    alerta_tiempodeventabajo: int
    alerta_cambiosbruscos: int
    alerta_inventarioinactivo: int
    alerta_varianzapreciosproveedores: int
    alerta_seguimientotendencias: int
    alerta_demandaestacional: int


@dataclass
class Parameters:
    """Modelo de dato para datos importados con el yaml"""

    data_source: dict
    logs_file: str
    connection_data_source: dict
    filter_data: dict
    query_template: dict
    query_template_write: dict
    type_data: dict
    type_data_out: dict
    scale: bool
    first_train: bool = True
    optimize: bool = True
    forecast_val: int = 1
    exp_time_cache: int = 8600000
    # model_parameters: Optional[Union[str, os.PathLike, PosixPath]]


@dataclass
class ParamsRedis:
    """Listado de parametro para se usado en redis"""

    connection_params: Optional[Union[str, os.PathLike, PosixPath]]


@dataclass
class ParamPlainText:
    """Listado de parametros para se usado en uso de archivo de texto plano"""

    file_path: Optional[Union[str, os.PathLike, PosixPath]]


@dataclass
class ModelRNN:
    """parametros iniciales para el modelo RNN"""

    model_name: Optional[str] = "model_version_ModelRNN"
    model: str = "LSTM"
    hidden_dim: int = 20
    dropout: float = 0.1
    batch_size: int = 16
    n_epochs: int = 300  # 2
    optimizer_kwargs: dict = default_field({"lr": 1e-3})
    log_tensorboard: bool = True
    random_state: int = 42
    training_length: int = 20
    input_chunk_length: int = 14
    force_reset: bool = True
    save_checkpoints: bool = True
    pl_trainer_kwargs: dict = default_field(
        {"accelerator": "gpu", "enable_progress_bar": True})


@dataclass
class ModelBlockRNN:
    """Parametro de modelo BlockRNN"""

    model_name: Optional[str] = "model_version_ModelBlockRNN"
    model: str = "GRU"
    input_chunk_length: int = 4
    output_chunk_length: int = 1
    hidden_dim: int = 10
    n_rnn_layers: int = 1
    batch_size: int = 32
    n_epochs: int = 200  # 2
    dropout: float = 0.1
    nr_epochs_val_period: int = 1
    optimizer_kwargs: dict = default_field({"lr": 1e-3})
    log_tensorboard: bool = True
    random_state: int = 42
    force_reset: bool = True
    save_checkpoints: bool = True
    pl_trainer_kwargs: dict = default_field(
        {"accelerator": "gpu", "enable_progress_bar": True})


@dataclass
class ModelExponentialSmoothing:
    """Parametro iniciales para el modelo ExponentialSmoothing"""

    seasonal_periods: int = 120


@dataclass
class ModelTCNModel:
    """Parametros iniciales para el modelo TCNModel"""

    model_name: Optional[str] = "model_version_ModelTCNModel"
    input_chunk_length: int = 4
    output_chunk_length: int = 1
    optimizer_kwargs: dict = default_field({"lr": 1e-3})
    n_epochs: int = 200  # 2
    dropout: float = 0.1
    dilation_base: int = 2
    weight_norm: int = True
    kernel_size: int = 3
    num_filters: int = 6
    nr_epochs_val_period: int = 1
    random_state: int = 0
    force_reset: bool = True
    save_checkpoints: bool = True
    pl_trainer_kwargs: dict = default_field(
        {"accelerator": "gpu", "enable_progress_bar": True})


@dataclass
class ModelFFT:
    """Parametros iniciales para el modelo FFT"""

    nr_freqs_to_keep: int = 20
    trend: str = "poly"


@dataclass
class ModelTransformerModel:
    """Parametros iniciales para el modelo Transformer"""

    model_name: Optional[str] = "model_version_ModelTransformerModel"
    input_chunk_length: int = 4
    output_chunk_length: int = 1
    optimizer_kwargs: dict = default_field({"lr": 1e-3})
    batch_size: int = 32
    n_epochs: int = 200  # 2
    nr_epochs_val_period: int = 10
    d_model: int = 16
    nhead: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    activation: str = "relu"
    random_state: int = 42
    force_reset: bool = True
    save_checkpoints: bool = True
    pl_trainer_kwargs: dict = default_field(
        {"accelerator": "gpu", "enable_progress_bar": True})


@dataclass
class ModelNBEATSModel:
    """Parametros para el modelo NBeats"""

    model_name: Optional[str] = "model_version_ModelNBEATSModel"
    input_chunk_length: int = 4
    output_chunk_length: int = 1
    generic_architecture: bool = True
    optimizer_kwargs: dict = default_field({"lr": 1e-3})
    num_stacks: int = 10
    num_blocks: int = 2
    num_layers: int = 3
    layer_widths: int = 256
    n_epochs: int = 300  # 2
    nr_epochs_val_period: int = 10
    batch_size: int = 1024
    force_reset: bool = True
    save_checkpoints: bool = True
    pl_trainer_kwargs: dict = default_field(
        {"accelerator": "gpu", "enable_progress_bar": True})


@dataclass
class ModelTFTModel:
    """Parametros iniciales para modelo TFT"""

    model_name: Optional[str] = "model_version_ModelTFTModel"
    input_chunk_length: int = 4
    output_chunk_length: int = 1
    optimizer_kwargs: dict = default_field({"lr": 1e-3})
    hidden_size: int = 64
    lstm_layers: int = 1
    num_attention_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 16
    n_epochs: int = 300  # 2
    add_relative_index: bool = False
    add_encoders: bool = default_field(
        {"cyclic": {"future": ["weekofyear", "month"]}})
    likelihood: int = default_field(QuantileRegression(
        quantiles=quantiles
    ))  # QuantileRegression is set per default
    # loss_fn:int=MSELoss(),
    random_state: int = 42
    force_reset: bool = True
    save_checkpoints: bool = True
    pl_trainer_kwargs: dict = default_field(
        {"accelerator": "gpu", "enable_progress_bar": True})


@dataclass
class ModelDLinearModel:
    """Parametros iniciales para modelo Dlinear"""

    model_name: Optional[str] = "model_version_ModelDLinearModel"
    input_chunk_length: int = 4
    output_chunk_length: int = 1
    shared_weights: bool = False
    optimizer_kwargs: dict = default_field({"lr": 1e-3})
    # kernel_size:int= 7
    const_init: bool = True
    use_static_covariates: bool = True
    batch_size: int = 16
    n_epochs: int = 300  # 2
    force_reset: bool = True
    random_state: int = 42
    save_checkpoints: bool = True
    pl_trainer_kwargs: dict = default_field(
        {"accelerator": "gpu", "enable_progress_bar": True})


@dataclass
class ModelNlinearModel:
    """Parametros iniciales para model Nlinear"""

    model_name: Optional[str] = "model_version_ModelNLinearModel"
    input_chunk_length: int = 4
    output_chunk_length: int = 1
    shared_weights: bool = False
    optimizer_kwargs: dict = default_field({"lr": 1e-3})
    # kernel_size:int = 7
    const_init: bool = True
    use_static_covariates: bool = True
    batch_size: int = 16
    n_epochs: int = 300  # 2
    force_reset: bool = True
    random_state: int = 42
    save_checkpoints: bool = True
    pl_trainer_kwargs: dict = default_field(
        {"accelerator": "gpu", "enable_progress_bar": True})
