from pathlib import Path
import json
import os
import numpy as np


def regla_de_3(y_0: int, x_1: int, y_1: int):
    """
    plantas por hectarea = salida
    tamano del area del lote = X1
    plantas por muestreo = Y
    tamano del area muestra = Y1


    salida         Y0
    _______  =  _______

      X1           Y1



    salida      Y * X1
            = __________

                  Y1
    """

    return float((y_0*x_1)/y_1)


def instruction_per_cores(list_instruction: list, core: int = os.cpu_count()):
    '''instruction_per_cores Funcion para separa una lista en multiples sublista
    usado para cuando se necesita hacer procesamiento con todos los nucleos

    Args:
        list_instruction (list): Lista  con las instrucciones o comnados completos 
        core (int, optional): Numero de nucleos usados para la taresa. Defaults to os.cpu_count().

    Returns:
        _type_: _description_
    '''
    init = 0
    end = 0
    result = []
    for _ in range(init, len(list_instruction), core):
        init = end
        end = core + init
        result.append(list_instruction[init:end])
    return result


def select_best_model(file: str | Path):
    """
    The function `select_best_model` reads model metrics from a JSON file and selects the model with the
    lowest total error based on MAPE, SMAPE, and R2 metrics.

    Args:
      file (str | Path): The `file` parameter in the `select_best_model` function is expected to be a
    string or a `Path` object representing the path to a JSON file containing model metrics data. The
    function reads this file, extracts the model metrics, calculates the total error for each model
    based on MAPE,

    Returns:
      The function `select_best_model` returns the best model based on the total error calculated from
    the error metrics (MAPE, SMAPE, R2) in the provided data file.
    """

    with open(file, "r", encoding="utf-8") as file_json:
        data_metric: dict = json.load(file_json)

    best_model = None
    min_error = float('inf')  # Initialize with a very large number

    models: dict = {}
    for key, val in data_metric.items():
        models[key] = val[key]

    for model in models:
        mape = models[model]['MAPE']
        smape = models[model]['SMAPE']
        r2 = models[model]['R2']

        # Calculate absolute values for error metrics
        abs_mape = abs(mape)
        abs_smape = abs(smape)
        abs_r2 = abs(r2)

        # Calculate total error
        total_error = abs_mape + abs_smape + abs_r2

        # Update best model if the total error is lower
        if total_error < min_error:
            min_error = total_error
            best_model = model

    return best_model
