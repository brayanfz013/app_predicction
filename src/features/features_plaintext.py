''' 
Metodo basico para guarcar y cargar archivos de texto plano siendo estos csv y ytxt

'''

import logging
import logging.config
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from ..data.logs import LOGS_DIR
except ImportError as error:
    path_folder = os.path.dirname(__file__)
    path_folder = str(Path(path_folder).parents)

    def search_subfolders(path: str):
        '''Funcion para agregar rutas al path de ejecucion'''
        folder = []
        for root, dirs, _ in os.walk(path, topdown=False):
            for name in dirs:
                if name == '':
                    print(f"[INFO] carpeta omitida: {name}")
                else:
                    folder.append(os.path.join(root, name))
        return folder

    for i in search_subfolders(path_folder):
        sys.path.insert(0, i)
    from data.logs import LOGS_DIR


class HandlePlainText:
    ''' Metodos base para leer archivos de texto plano'''

    def __init__(self, logfile='logging.conf'):

        # Constructor que permite inicializar los parametros
        logging.config.fileConfig(os.path.join(LOGS_DIR, logfile))
        self.log = logging.getLogger('POSTGRES')

        self.log.debug("Instancia libreria")

    def read_plain_csv(self, path_file: str, columns: list, **args):
        '''read_plain_text Metodo para cargar un archivo de texto plano csv

        Args:
            path_file (str): ruta del archivo en memoria del csv
            columns (list): Lista de columnas que se van a cargar en dataframe

        Returns:
            _type_: Dataframe de datos cargados
        '''
        return pd.read_csv(path_file, columns, **args)

    def write_plain_csv(self, save_path: str, data_frame: pd.DataFrame):
        '''write_plain_text Metodo base para guardasr un archivo en texto plano
        proveniente de un dataframe

        Args:
            save_path (str): Rutan donde se guardara un archivos
            data_frame (pd.DataFrame): Datos que se van a guardar
        '''
        return data_frame.to_csv(save_path)

    def read_plain_txt(self, save_path: str) -> list:
        '''read_plain_txt Lectura de archivo de text plano

        Args:
            save_path (str): Ruta del archivo el cual se quiere leer

        Returns:
            _type_: Lista con archivo de texto plano
        '''
        # Abrir un archivo en modo de lectura (r) y leer las líneas
        with open(save_path, 'r', encoding='utf-8') as file:
            lineas = file.readlines()  # lee todas las líneas y las guarda en una lista
        return lineas

    def writte_plain_txt(self, data: list, save_path: str):
        '''writte_plain_txt metodo para escribir una lista
        en un archivo de texto planos

        Args:
            save_path (str): Ruta donde se guardara la informacion
        '''
        # Abrir el archivo en modo de lectura
        with open(save_path, 'r', encoding='utf-8') as file:
            for line in data:
                file.write(line+'\n')

    def show_lines_plain_txt(self, lines):
        '''show_lines_plain_txt Lectura de archivo de texto plano

        Args:
            lines (_type_): Listado de lineas leidas 
        '''
        # Imprimir cada línea
        for linea in lines:
            # strip() se usa para quitar los espacios en blanco y saltos de línea al principio y final
            print(linea.strip())
