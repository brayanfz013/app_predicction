"""
Codigo para almacenas los argumentos de los diferentes
parametros de los modelos
"""
import os
from pathlib import PosixPath
from typing import Union, Optional
from dataclasses import dataclass,field

@dataclass
class ParamsPostgres:
    '''Listado de parametros para ejecutar HandleDBsql'''
    connection_params : Optional[Union[str, os.PathLike, PosixPath]]
    query_read : Optional[Union[str, os.PathLike, PosixPath]]
    query_write :Optional[Union[str, os.PathLike, PosixPath]]
    logger_file :Optional[Union[str, os.PathLike, PosixPath]]
    names_table_columns : Optional[Union[str, os.PathLike, PosixPath]]