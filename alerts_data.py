# -*- coding: utf-8 -*-
# =============================================================================
__author__ = "Brayan Felipe Zapata "
__copyright__ = "Copyright 2007, The Cogent Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Brayan Felipe Zapata"
__email__ = "bzapata@smgsoftware.com"
__status__ = "Production"
# =============================================================================
'''
Codigo para aplicar criterios de filtrado a la base de datos, generando
alerta para lo diferentes modelos en un base de datos escribiendolos en
en postgres o redis
'''
# =============================================================================
import os
import yaml
import logging
import numpy as np
from scipy import stats
from pathlib import Path
from src.features.features_fix_data import ColumnsNameHandler
from src.lib.class_load import LoadFiles
from src.features.features_postgres import HandleDBpsql
from src.lib.factory_data import HandleRedis, SQLDataSourceFactory, get_data
from src.models.args_data_model import Parameters, AlertsData
from src.data.logs import LOGS_DIR
from src.features.filter_data import (AlertaPorBajaCantidad,
                                      AlertaPorTiempoDeVentaBajo,
                                      AlertaPorCambiosBruscosEnLasVentas,
                                      AlertaPorInventarioInactivo,
                                      #   AlertasPorVariacionPreciosProveedores,
                                      AlertaPorSeguimientoTendencias,
                                      AlertaPorDemandaEstacional,
                                      AlertaObserver,
                                      Inventario)