''' Clase recopilatoria encargar de cargar y guardar archivos en memorias '''

import os
import json
from pathlib import Path
from typing import Union
import numpy as np
import pickle


class LoadFiles(object):
    '''Clase recopilatoria de diferentes funciones para cargar archivos de carpetas y del sistema'''

    def read_names(self, path_dir):
        '''
        Lee los archivos de una carpeta y los reportana ordenados

        @path_dir:
            Info:Especificacion de la ruta de un directorio para leer archivos
            Dtype:String

        '''
        nombres = []
        for item in os.listdir(path_dir):
            nombres.append(os.path.join(path_dir, item))
        return sorted(nombres)

    def load_path_names(self, path_dir: str, extenciones: list):
        '''
        Lee los archivos de una carpeta y retorna una lista con la ruta completa del archivo
        y otra con solo el nombre del archivo

        @path_dir:
            Info:Especificacion de la ruta de un directorio para leer archivos
            Dtype:String

        @extenciones:
            Info: Lista de string separda por comas con las extenciones que se desean buscar en la carpeta
            Dtype:Lista Strings ['.tiff','.jpg','.jpge','.tif','.png'load_image_path]

        @sample:
            Info: Retorna una lista con la ruta completa del las imagenes
            Dtype:String

        @names:
            Info: Retorna una lista unicamente con los nombres de los archivos
            Dtype: String
        '''

        extenciones = ['.' + i for i in extenciones]
        extenciones = extenciones + [i.upper() for i in extenciones]
        extenciones = sorted(set(extenciones))

        data_path = Path(path_dir)

        if data_path.is_dir():
            sample = []
            names = []

            for i in extenciones:
                t_file = '**/*' + i
                if len(list(data_path.glob(t_file))) == 0:
                    continue
                else:
                    sample.append(list(data_path.glob(t_file)))

            def flatten(lista):
                return [item for sublist in lista for item in sublist]
            sample = flatten(sample)
            sample = sorted(set(sample))

            if not sample:
                print(
                    f'\n[INFO] No hay imagenes con estas extenciones:\n\t{extenciones}\n\tEn la ruta:\n\t{path_dir}\n')
                return [], []

            else:

                if len(sample) != 1:

                    for name in sample:
                        names.append(name.parts[-1])
                    sample = np.array([str(i) for i in sample])
                    return sorted(set(sample)), sorted(set(names))
                else:
                    return str(sample[0]), str(sample[0].parts[-1])

        else:
            return print('[INFO], la ruta no existe, favor revisar el directorio')

    def search_load_files_extencion(self, path_search: str, ext: list) -> dict:
        '''search_load_files_extencion Mejora de la funcion load_path_names agregando caracteristicas de
        agrupacion por tipo , elimina repertido y retorna un diccionario con las key por cada extencion

        Args:
            path_search (str): Ruta directorio donde buscar las extenciones
            ext (list): Listado de extenciones para buscar dentro del path

        Returns:
            dict: Diccionario con las key como extenciones y listado por tipo de extencion

        '''
        # Conversion de str a tipo Path
        data_path = Path(path_search)

        if data_path.is_dir():
            # conversion de las extenciones a diccionario para facilitar la busqueda
            search = dict([(i, (f'.{i.upper()}', f'.{i}'))
                          for i in sorted(set(ext))])

            # Diccionario sobre el cual se retornan los tipos
            sample = dict([(i, []) for i in sorted(set(ext))])

            for key, values in search.items():
                for ext in values:
                    t_file = '**/*' + ext
                    search_file = list(data_path.glob(t_file))
                    if search_file:
                        names_file = []
                        file_path = []
                        for found_file in search_file:
                            file_path.append(str(found_file))
                            names_file.append(found_file.name.split('.')[0])
                        sample[key].append(names_file)
                        sample[key].append(file_path)
        else:
            print('[INFO], la ruta no existe, favor revisar el directorio')

        return sample

    def save_dic_to_txt(self, dict_data: dict, path_to_save: str, file_name: str) -> None:
        '''save_dic_to_txt Guardar un diccionario en un archivos txt

        Args:
            dict_data (dict): Diccionario el cual se desea almacenar
            path_to_save (str): Ruta en memoria donde se almacena el archivo
            file_name (str): Nombre del archivos
        '''
        path_file = Path(os.path.join(path_to_save, file_name)
                         ).with_suffix('.txt')

        with open(path_file, "w", encoding='utf-8') as savefile:
            savefile.write(str(dict_data))

    def save_dict_to_json(self, dict_data: dict, path_to_save: str, file_name: str) -> None:
        '''save_dict_to_json _summary_

        Args:
            dict_data (dict):  Diccionario el cual se desea almacenar
            path_to_save (str): Ruta en memoria donde se almacena el archivo
            file_name (str): Nombre del archivos
        '''

        path_file = Path(
            os.path.join(path_to_save, file_name)
        ).with_suffix('.json')

        with open(path_file, "w", encoding='utf-8') as file_save:
            json.dump(dict_data, file_save)

    def is_empty(self, path):
        """
        Funcion  para determinar si una carpeta se encuentra vacia

        Si no esta vacio retorna 1, Si esta vacio retorna 0

        @path: ruta de la carpeta que se quiere comprobar si esta vacia o no

        """

        if os.path.exists(path) and not os.path.isfile(path):

            # Checking if the directory is empty or not

            if not os.listdir(path):
                print("Empty directory")
                value = 0

            else:
                print("Not empty directory")
                value = 1
        else:
            print("The path is either for a file or not valid")
            value = 0

        return value

    def json_to_dict(self, json_file: str) -> Union[dict, list]:
        '''
        Lee los archivos de una carpeta y retorna una lista con la ruta completa del archivo
        y otra con solo el nombre del archivo

        @json:
            Info:Especificacion de la ruta de un directorio para leer archivos
            Dtype:String

        @data:
            Info: Retorna una lista con la ruta completa del las imagenes
            Dtype:String

        @nKeys_dictames:
            Info: Retorna una lista con las Keys del diccionario
            Dtype: List

        '''
        # path = os.path.join(json_file)
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data, list(data.keys())

    def delete_files_in_folder(self, path_dir: Union[str, os.PathLike]) -> None:
        '''delete_files_in_folder Funcion para una carpeta dada eliminar su contenido
        sin importar lo que tenga adentro

        Args:
            path_dir (Union[str,os.PathLike]): Ruta de la carpeta a la cual se le desea
            eliminar el contenido
        '''
        for item in os.listdir(path_dir):
            os.remove(os.path.join(path_dir, item))

    def save_scaler(self, scaler, path):
        '''Metodo para guardar el escalador para hacer predicciones futuras'''
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)

    def load_scaler(self, path):
        '''Metodo para cargar los datos serializado del scaler'''
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    
    def save_strategy(self,pickle_strategy:str,pathfile:dict):
        '''save_strategy Metodo para guardar diccionario de funciones en 
        pytho en este caso un diccionario con metodo para depurar datos

        Args:
            pickle_strategy (str): Diccionario con valores que son funciones
            replace = {
                        int:lambda x: int(float(x.replace(',',''))),
                        float:lambda x: float(x.replace(',',''))
                    }

            pathfile (dict): Ruta donde se guarda los datos del diccionario
        '''
        # Ahora los datos están guardados en 'strategy.pkl'
        with open(pickle_strategy, 'wb') as file:
            pickle.dump(pathfile, file)

    def load_strategy(self,pickle_strategy):
        '''load_strategy Metodo para cargar informacion de funciones de un diccionario

        Args:
            pickle_strategy (_type_): Rutan donde se encuentra guardada la informacion

        Returns:
            _type_: archivos cargados como diccionario
        '''
        # Podemos cargar los datos del archivo de esta manera
        with open(pickle_strategy, 'rb') as file:
            loaded_replace = pickle.load(file)
        return loaded_replace