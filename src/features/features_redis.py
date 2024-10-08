'''
Codigo con funciones base para enviar informacion al servidor de redis

'''
import json
import logging
import logging.config
import os
import yaml
import socket
import sys
from configparser import ConfigParser
from pathlib import Path
import pickle
import redis
import pandas as pd
try:
    from ..data.logs import LOGS_DIR
except ImportError as error:
    print(error)
    PATH_FOLDER = str(Path(os.path.dirname(__file__)).parents)

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

    for i in search_subfolders(PATH_FOLDER):
        sys.path.insert(0, i)
    from data.logs import LOGS_DIR


class HandleRedis(object):
    """
    Libreria para realizar creacion de base de datso y conexiones a una base de datos usando SQLalquemy
    """

    def __init__(self, logfile='data/config/logging.conf'):

        path_folder = str(Path(os.path.dirname(__file__)).parents[0])
        logs_file = str(Path(path_folder).joinpath(logfile))

        # Constructor que permite inicializar los parametros
        logging.config.fileConfig(os.path.join(LOGS_DIR, logs_file))
        self.log = logging.getLogger('datasource')

    def file_ini_(self, filename: str = 'database', section: str = 'redis'):
        '''file_ini_ Metodo para cargar parametros cuando la extencion del archivo
        de parametros es ini

        Args:
            filename (str, optional): Nombre de ruta con extencion .ini. Defaults to 'database'.
            section (str, optional): Tipo de conexion con la base de datos. Defaults to 'redis'.

        Raises:
            Exception: Error de conexion por no encontrar parametros
            Exception: Erroe de conexion por no encontrar el nombre de la conexion

        Returns:
            _type_: Diccionario con los parametros de conexion
        '''
        # creacion de un "Parser"
        parser = ConfigParser()

        # lectura de archivo de configuracion
        parser.read(filenames=filename)

        data_parameters = {}
        # Extracion de parametros del archivo , para convertilo en dict()
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                if param[1]:
                    data_parameters[param[0]] = param[1]
                else:
                    raise ValueError(
                        f'Error configuration file {filename} \n missing value for fiel: {param[0]}')
        else:
            raise ValueError(
                f'Section {section} not found in the {filename} file')

        return data_parameters

    def file_yaml(self, filename: str, section: str = 'redis'):
        '''file_yaml Metodo rapido para cargar los archivos de la base detaso
        cuando se tiene un yaml usando la clave de connection_data_source

        Args:
            filename (str): Ruta del archivo de la fuente de datos

        Returns:
            _type_: Diccionario con los criterios de conexion
        '''
        with open(filename, 'r', encoding='utf-8') as file:
            load_yaml = yaml.safe_load(file)

        return load_yaml['connection_data_source'][section]

    def get_config_file(self, filename: str | dict = 'database', section: str = 'redis'):
        """
        This Python function retrieves configuration parameters from either an INI or YAML file or a
        dictionary based on the specified filename and section.

        Args:
          filename (str | dict): The `filename` parameter can either be a string representing the name of
        the configuration file (with default value 'database') or a dictionary containing configuration
        settings. Defaults to database
          section (str): The `section` parameter in the `get_config_file` method is used to specify the
        section within the configuration file (specified by `filename`) from which the parameters should be
        retrieved. Defaults to redis

        Returns:
          The function `get_config_file` returns the parameters based on the specified filename and section.
        The parameters are retrieved from either an INI file, a YAML file, or a dictionary based on the
        input provided.
        """

        if isinstance(filename, str):
            if Path(filename).suffix == '.ini':
                parameters = self.file_ini_(filename=filename, section=section)

            elif Path(filename).suffix == '.yaml':
                parameters = self.file_yaml(filename=filename, section=section)

                raise ValueError(
                    f"Archivo no válido: {filename}. Sólo se permiten archivos .ini o .yaml.")

        elif isinstance(filename, dict):
            parameters = filename[section]

        return parameters

    def create_data(self, data: dict, config: str):
        '''create_data Funcion para crear y enviar datos a un servidor de redis
        usando un diccionario y verificando conectividad

        Args:
            data (dict):Diccionarion con la informacion ser enviada, debe ser formato Json
            config (str): Ruta del archivos de las configuraciones para la conexion con redis
        '''
        try:
            parameters_connection = self.get_config_file(
                config, section='redis')
            with redis.Redis(**parameters_connection) as connection:
                # self.log.debug("Conexion exitosa")
                with connection.pipeline() as pipe:
                    for h_id, hat in data.items():
                        pipe.hmset(h_id, hat)
                    pipe.execute()
            # self.log.debug("Datos enviados a redis")
        except (redis.exceptions.DataError, redis.exceptions.AuthenticationError) as redis_error:
            self.log.error(redis_error)
            # print(redis_error)

    def get_alldata(self, file_name: str, config: str):
        """get_posicition Funcion para traer toda la lista de datos de una collecion de redis

        Args:
            file_name (str): Nombre / Key que se desea traer
            config (str): Ruta del archivos de las configuraciones para la conexion con redis

        Returns:
            _type_: _description_
        """
        try:
            parameters_connection = self.get_config_file(
                config, section='redis')
            with redis.Redis(**parameters_connection) as connection:
                data = connection.hgetall(file_name)
                unidict = dict((k.decode('utf8'), v.decode('utf8'))
                               for k, v in data.items())
                # self.log.debug("Extracion de datos completa")
            return unidict
        except (redis.exceptions.DataError, redis.exceptions.AuthenticationError) as redis_error:
            self.log.error(redis_error)
            # print(redis_error)

    def get_single_value(self, dict_key: str, file_name: str, config: str):
        """get_posicition Funcion para traer toda la lista de datos de una collecion de redis

        Args:
            file_name (str): Nombre / Key que se desea traer
            config (str): Ruta del archivos de las configuraciones para la conexion con redis

        Returns:
            _type_: _description_
        """
        try:
            parameters_connection = self.get_config_file(
                config, section='redis')
            with redis.Redis(**parameters_connection) as connection:
                data = connection.hget(dict_key, file_name)
                data = int(data.decode('utf8'))
                # self.log.debug("Extracion de datos completa")
            return data
        except (redis.exceptions.DataError, redis.exceptions.AuthenticationError) as redis_error:
            self.log.error(redis_error)
            # print(redis_error)

    def set_single_value(self, dict_key: str, file_name: str, value: int, config: str | dict,):
        """
        The `set_single_value` function sets a single value in a Redis hash using the provided dictionary
        key, file name, value, and configuration.

        Args:
          dict_key (str): The `dict_key` parameter is the key of the dictionary in Redis where you want to
        set the value. It is a string that represents the key of the dictionary.
          file_name (str): The `file_name` parameter is a string that represents the name or key of the file
        you want to set the value for in Redis.
          value (int): The `value` parameter in the `set_single_value` function is an integer that
        represents the value you want to set for a specific key in a Redis hash.
          config (str): The `config` parameter is the path to the configuration file that contains the
        connection details for Redis. It is used to establish a connection with Redis and perform operations
        on it.

                get_posicition Funcion para traer toda la lista de datos de una collecion de redis

        Args:
            file_name (str): Nombre / Key que se desea traer
            config (str): Ruta del archivos de las configuraciones para la conexion con redis

        Returns:
            _type_: _description_
        """

        try:
            parameters_connection = self.get_config_file(
                config, section='redis')
            with redis.Redis(**parameters_connection) as connection:
                connection.hset(dict_key, file_name, value)
                # self.log.debug("Extracion de datos completa")
        except (redis.exceptions.DataError, redis.exceptions.AuthenticationError) as redis_error:
            self.log.error(redis_error)
            # print(redis_error)

    def set_dict_data(self, hash_name: str, dict_data: dict, config: str):
        """set_dict_data Metodo para enviar un diccionario de datos a redis

        Args:
            hash (str): Nombre del hash donde se guardara la informacion del diccionario
            dict_data (dict): data del diccionario tiene que estar denotado por un
            config (str): Ruta del archivos de las configuraciones para la conexion con redis
        """
        try:
            # Conversion de los indicadores de keys como strings
            dict_data[hash_name] = {
                str(key): value for key, value in dict_data[hash_name].items()}

            parameters_connection = self.get_config_file(
                config, section='redis')
            with redis.Redis(**parameters_connection) as connection:
                for key_data, _val_data_line in dict_data[hash_name].items():
                    serialize = json.dumps(_val_data_line)
                    connection.hset(hash_name, key_data, serialize)
                # self.log.debug("Extracion de datos completa")
        except (redis.exceptions.DataError, redis.exceptions.AuthenticationError) as redis_error:
            self.log.error(redis_error)
            # print(redis_error)

    def get_dict_data(self, hash_name: str, config: str):
        """
        The function `get_dict_data` retrieves data from a Redis hash and returns it as a dictionary.

        Args:
          hash_name (str): The `hash_name` parameter is a string that represents the name of the hash in
        Redis from which you want to retrieve data.
          config (str): The `config` parameter is the path to the configuration file that contains the
        necessary information for connecting to Redis.

        Returns:
          a dictionary object named `data`.

          get_dict_data _summary_

        Args:
            hash_name (str): _description_
            dict_data (dict): _description_
            config (str): Ruta del archivos de las configuraciones para la conexion con redis
        """

        data: dict = {hash_name: {}}
        try:
            parameters_connection = self.get_config_file(
                config, section='redis')
            with redis.Redis(**parameters_connection) as connection:
                for key in connection.hkeys(hash_name):
                    # Recuperar la cadena del hash de Redis
                    value_str = connection.hget(hash_name, key)
                    # Convertir la cadena a un diccionario
                    data[hash_name][key] = json.loads(value_str)

                    # self.log.debug("Extracion de datos completa")
        except (redis.exceptions.DataError, redis.exceptions.AuthenticationError) as redis_error:
            self.log.error(redis_error)
            # print(redis_error)

        return data

    def check_connection(self, config: str):
        '''Metodo basico para verificar la conexion a la base de datos de redis
            Args:
            config (str): Ruta del archivos de las configuraciones para la conexion con redis
        '''
        try:
            parameters_connection = self.get_config_file(
                config, section='redis')
            with redis.Redis(**parameters_connection) as connection:
                print(connection.ping())

        except (redis.exceptions.DataError, redis.exceptions.AuthenticationError, redis.ConnectionError) as redis_error:
            self.log.error(redis_error)

    def get_cache_data(self, hash_name: str, config: dict):
        """
        The function `get_cache_data` retrieves data from a Redis cache using a given hash name and
        configuration.

        Args:
          hash_name (str): The `hash_name` parameter is a string that represents the name
          of the hash or key in the Redis cache from which you want to retrieve data.
          config (str): The `config` parameter is a string that represents the name or path of the
        configuration file. It is used to retrieve the connection parameters for the Redis server.

        Returns:
          the data retrieved from the cache, which is either a deserialized object or None if there
         is no data in the cache.
        """

        parameters_connection = self.get_config_file(
            config, section='redis')

        with redis.Redis(**parameters_connection) as connection:
            data = connection.get(hash_name)
            if data is not None:
                data = pickle.loads(data)
            else:
                self.log.debug("No hay datos en cache")
                data = None
        return data

    def set_cache_data(self, hash_name: str, old_dataframe: pd.DataFrame, new_dataframe: pd.DataFrame, exp_time: int, config: dict):
        """
        The `set_cache_data` function is used to cache database data in Redis, allowing
        for the storage and retrieval of old and new dataframes.

        Args:
            hash_name (str): The `hash_name` parameter is a string that represents the name of
            the hash/key used to store the information in the Redis cache.
            old_dataframe (pd.DataFrame): The `old_dataframe` parameter is a Pandas DataFrame that
            contains the previous data that was stored in the cache. It is used to check if there
            is any existing data in the cache.
            new_dataframe (pd.DataFrame): The `new_dataframe` parameter is a Pandas DataFrame
            that contains the new data that you want to add to the cache.
            exp_time (int): The `exp_time` parameter represents the expiration time in seconds for the
            cached data in Redis. After this time has passed, the data will be automatically
            removed from the cache.
            config (str): The `config` parameter is a string that represents the parameters for
            connecting to Redis. It is used to retrieve the connection parameters from a
            configuration file.

        Returns:
          The method `set_cache_data` returns the updated DataFrame `update_frame`.
        """

        try:
            parameters_connection = self.get_config_file(
                config, section='redis')

            with redis.Redis(**parameters_connection) as connection:
                old_df_bytes = connection.get(hash_name)

                # Si existe un DataFrame antiguo, deserialízalo
                if old_df_bytes is not None:
                    self.log.debug('Recuperando data existente')
                    old_df = pickle.loads(old_df_bytes)
                elif old_dataframe is not None:
                    # Si no existe un DataFrame antiguo, crea uno vacío con las mismas columnas
                    self.log.debug('Insertando nueva data')
                    old_df = old_dataframe.copy()
                    connection.set(hash_name, pickle.dumps(
                        old_dataframe), ex=exp_time)
                else:
                    self.log.debug("No hay datos en cache")
                    old_df = None

            # Verficar si existe nueva informacion por rellenar
            if new_dataframe is not None and isinstance(old_df, pd.DataFrame):
                # Combina el DataFrame antiguo con el nuevo
                # update_frame = old_df.append(new_dataframe, ignore_index=True).drop_duplicates()
                self.log.debug('Verificacion si existe nueva informacion')
                update_frame = pd.concat([old_df, new_dataframe])

                # Serializa el DataFrame a bytes usando pickle
                # Guarda los bytes en Redis
                connection.set(hash_name, pickle.dumps(
                    update_frame), ex=exp_time)
            elif isinstance(old_df, pd.DataFrame):
                update_frame = old_df.copy()
            else:
                update_frame = None
            return update_frame
            # self.log.debug("Extracion de datos completa")
        except (redis.exceptions.DataError, redis.exceptions.AuthenticationError, redis.ConnectionError) as redis_error:
            self.log.error(redis_error)
            # print(redis_error)

    def search_public_ip(self):
        '''Funcion para buscar la ip publica en una conexion'''
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as search:
            search.connect(("8.8.8.8", 80))
            ip_publica = search.getsockname()[0]
        return ip_publica


# if __name__ == '__main__':

    # CONECTION_FILE = '/home/smg/car/carrito/config/redis.ini'

    # redis_handler = HandleRedis()
    # position = redis_handler.get_alldata('posicion',CONECTION_FILE)
    # bussy_index =[]
    # for index,data in enumerate(position.items()):
    #     key,val = data
    #     if val != str(0):
    #         bussy_index.append(index+1)

    # print(bussy_index)
    # ================================================================================================

    # #Archivo de conexiones
    # CONECTION_FILE = '/home/smg/car/carrito/config/redis.ini'
    # #Archivo de esquema
    # INIT_SET_DATA = '/home/smg/car/carrito/data/initial_data.json'

    # redis_handler = HandleRedis()
    # loads = LoadFiles()

    # # Cargar datos para enviar
    # init_redis_data = loads.json_to_dict(INIT_SET_DATA)[0]
    # print(init_redis_data)

    # #Escritura inicila de los datos de almacenamiento de posicion
    # data = redis_handler.get_alldata('posicion', CONECTION_FILE)

    # print(data,type(data))
    # pprint(data)

    # data_sim  = [4]
    # siguiente = data_sim[0]
    # position_server = int(list(data.values())[data_sim[0]-1])
    # position_name = list(data.keys())[data_sim[0]-1]

    # print(f'data redis {position_name}' )
    # print(f'data listado {siguiente}')

    # if position_server:
    #     print('ocupado')
    #     data_sim.insert(0,siguiente-1)
    # else:
    #     print('libre')
    # print(data_sim)
    # ================================================================================================
    # #Archivo de conexiones
    # CONECTION_FILE = '/home/smg/car/carrito/config/redis.ini'
    # #Archivo de esquema
    # INIT_SET_DATA = '/home/smg/car/carrito/data/initial_data.json'

    # redis_handler = HandleRedis()
    # loads = LoadFiles()

    # # Cargar datos para enviar
    # init_redis_data = loads.json_to_dict(INIT_SET_DATA)[0]

    # # Escritura inicila de los datos de almacenamiento de posicion
    # redis_handler.create_data(init_redis_data, CONECTION_FILE)

    # # #Registrar un solo valor en Redis
    # # redis_handler.set_single_value('posicion','POS4',11,CONECTION_FILE)
