# #==============================================================================
# model_futcov = RNNModel(
#     model="LSTM",
#     hidden_dim=20,
#     batch_size=8,
#     n_epochs=100,
#     random_state=0,
#     training_length=36,
#     input_chunk_length=24,
#     force_reset=True,
# )

# model_futcov.fit(
#     series=[train_air, train_milk],
#     future_covariates=[air_train_covariates, milk_train_covariates],
#     verbose=False,
# )


# my_model = RNNModel(
#     model="LSTM",
#     hidden_dim=20,
#     dropout=0,
#     batch_size=16,
#     n_epochs=300,
#     optimizer_kwargs={"lr": 1e-3},
#     model_name="Air_RNN",
#     log_tensorboard=True,
#     random_state=42,
#     training_length=20,
#     input_chunk_length=14,
#     force_reset=True,
#     save_checkpoints=True,
# )

# my_model.fit(
#     train_transformed,
#     future_covariates=covariates,
#     val_series=val_transformed,
#     val_future_covariates=covariates,
#     verbose=True,
# )

# model_en = RNNModel(
#     model="LSTM",
#     hidden_dim=20,
#     n_rnn_layers=2,
#     dropout=0.2,
#     batch_size=16,
#     n_epochs=10,
#     optimizer_kwargs={"lr": 1e-3},
#     random_state=0,
#     training_length=300,
#     input_chunk_length=300,
#     likelihood=GaussianLikelihood(),
# )
# #==============================================================================
# model_pastcov = BlockRNNModel(
#     model="LSTM",
#     input_chunk_length=24,
#     output_chunk_length=12,
#     n_epochs=100,
#     random_state=0,
# )

# model_pastcov.fit(
#     series=[train_air, train_milk],
#     past_covariates=[air_train_covariates, milk_train_covariates],
#     verbose=False,
# )

# my_model_sun = BlockRNNModel(
#     model="GRU",
#     input_chunk_length=125,
#     output_chunk_length=36,
#     hidden_dim=10,
#     n_rnn_layers=1,
#     batch_size=32,
#     n_epochs=100,
#     dropout=0.1,
#     model_name="sun_GRU",
#     nr_epochs_val_period=1,
#     optimizer_kwargs={"lr": 1e-3},
#     log_tensorboard=True,
#     random_state=42,
#     force_reset=True,
# )

# #==============================================================================
# model = FFT(required_matches=set(), nr_freqs_to_keep=None)
# model.fit(train)
# pred_val = model.predict(len(val))



# model = FFT(nr_freqs_to_keep=None)
# model.fit(train)
# pred_val = model.predict(len(val))

# model = FFT(nr_freqs_to_keep=20)
# model.fit(train)
# pred_val = model.predict(len(val))


# model = FFT(trend="poly")
# model.fit(train)
# pred_val = model.predict(len(val))

# #==============================================================================



# pred_series_ets = ExponentialSmoothing(seasonal_periods=120).historical_forecasts(
#     series_sp_transformed,
#     start=pd.Timestamp("19401001"),
#     forecast_horizon=36,
#     stride=10,
#     retrain=True,
#     last_points_only=True,
#     verbose=True,
# )

# #==============================================================================


# model_air = TCNModel(
#     input_chunk_length=13,
#     output_chunk_length=12,
#     n_epochs=500,
#     dropout=0.1,
#     dilation_base=2,
#     weight_norm=True,
#     kernel_size=5,
#     num_filters=3,
#     random_state=0,
# )

# model_sun = TCNModel(
#     input_chunk_length=250,
#     output_chunk_length=36,
#     n_epochs=100,
#     dropout=0,
#     dilation_base=2,
#     weight_norm=True,
#     kernel_size=3,
#     num_filters=6,
#     nr_epochs_val_period=1,
#     random_state=0,
# )

# model_air.fit(
#     series=train,
#     past_covariates=train_month,
#     val_series=val,
#     val_past_covariates=val_month,
#     verbose=True,
# )

# deeptcn = TCNModel(
#     input_chunk_length=30,
#     output_chunk_length=20,
#     kernel_size=2,
#     num_filters=4,
#     dilation_base=2,
#     dropout=0,
#     random_state=0,
#     likelihood=GaussianLikelihood(),
# )


# #==============================================================================

# my_model = TransformerModel(
#     input_chunk_length=12,
#     output_chunk_length=1,
#     batch_size=32,
#     n_epochs=200,
#     model_name="air_transformer",
#     nr_epochs_val_period=10,
#     d_model=16,
#     nhead=8,
#     num_encoder_layers=2,
#     num_decoder_layers=2,
#     dim_feedforward=128,
#     dropout=0.1,
#     activation="relu",
#     random_state=42,
#     save_checkpoints=True,
#     force_reset=True,
# )

# my_model.fit(series=train_scaled, val_series=val_scaled, verbose=True)

# #==============================================================================

# model_nbeats = NBEATSModel(
#     input_chunk_length=30,
#     output_chunk_length=7,
#     generic_architecture=True,
#     num_stacks=10,
#     num_blocks=1,
#     num_layers=4,
#     layer_widths=512,
#     n_epochs=100,
#     nr_epochs_val_period=1,
#     batch_size=800,
#     model_name="nbeats_run",
# )

# model_nbeats.fit(train, val_series=val, verbose=True)

# model_nbeats = NBEATSModel(
#     input_chunk_length=30,
#     output_chunk_length=7,
#     generic_architecture=False,
#     num_blocks=3,
#     num_layers=4,
#     layer_widths=512,
#     n_epochs=100,
#     nr_epochs_val_period=1,
#     batch_size=800,
#     model_name="nbeats_interpretable_run",
# )

# #==============================================================================

# # default quantiles for QuantileRegression
# quantiles = [
#     0.01,
#     0.05,
#     0.1,
#     0.15,
#     0.2,
#     0.25,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.75,
#     0.8,
#     0.85,
#     0.9,
#     0.95,
#     0.99,
# ]
# input_chunk_length = 24
# forecast_horizon = 12
# my_model = TFTModel(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=forecast_horizon,
#     hidden_size=64,
#     lstm_layers=1,
#     num_attention_heads=4,
#     dropout=0.1,
#     batch_size=16,
#     n_epochs=300,
#     add_relative_index=False,
#     add_encoders=None,
#     likelihood=QuantileRegression(
#         quantiles=quantiles
#     ),  # QuantileRegression is set per default
#     # loss_fn=MSELoss(),
#     random_state=42,
# )

# my_model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)



# input_chunk_length_ice = 36

# # use `add_encoders` as we don't have future covariates
# my_model_ice = TFTModel(
#     input_chunk_length=input_chunk_length_ice,
#     output_chunk_length=forecast_horizon_ice,
#     hidden_size=32,
#     lstm_layers=1,
#     batch_size=16,
#     n_epochs=300,
#     dropout=0.1,
#     add_encoders={"cyclic": {"future": ["month"]}},
#     add_relative_index=False,
#     optimizer_kwargs={"lr": 1e-3},
#     random_state=42,
# )

# # fit the model with past covariates
# my_model_ice.fit(
#     train_ice_transformed, past_covariates=covariates_heat_transformed, verbose=True
# )