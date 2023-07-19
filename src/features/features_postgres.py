import logging
import logging.config
import yaml
import os
import sys
from configparser import ConfigParser
from pathlib import Path
import json
import pandas as pd
import psycopg2

try:
    from ..data.logs import LOGS_DIR
except ImportError as error:
    path_folder = os.path.dirname(__file__)
    path_folder = Path(path_folder).parents[0]
    omitir = ''

    def search_subfolders(path: str):
        '''Funcion para agregar rutas al path de ejecucion'''
        folder = []
        for root, dirs, _ in os.walk(path, topdown=False):
            for name in dirs:
                if name == omitir:
                    print(f"[INFO] carpeta omitida: {name}")
                else:
                    folder.append(os.path.join(root, name))
        return folder

    for i in search_subfolders(path_folder):
        sys.path.insert(0, i)
    from data.logs import LOGS_DIR

class HandleDBpsql(object):
    """
    Libreria para realizar creacion de base de datso y conexiones a una base de datos usando SQLalquemy    
    """

    def __init__(self, logfile='data/config/logging.conf'):
        path_folder = os.path.dirname(__file__)
        path_folder = str(Path(path_folder).parents[0])
        logs_file = str(Path(path_folder).joinpath(logfile))

        # Constructor que permite inicializar los parametros
        logging.config.fileConfig(os.path.join(LOGS_DIR, logs_file))
        self.log = logging.getLogger('POSTGRES')


    def file_ini_(self, filename: str = 'database', section: str = 'postgresql'):
        '''file_ini_ Metodo para cargar parametros cuando la extencion del archivo 
        de parametros es ini

        Args:
            filename (str, optional): Nombre de ruta con extencion .ini. Defaults to 'database'.
            section (str, optional): Tipo de conexion con la base de datos. Defaults to 'postgresql'.

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

    def file_yaml(self, filename: str, section: str = 'postgresql'):
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

    def get_config_file(self, filename:dict | str = 'database', section: str = 'postgresql'):
        """
        Lee el archivo de configuracion con los parametros a la base de datos
        se tiene que seleccion el motor de base de datos.

        Retorna un diccionario con la lectura de los parametros dentro del archivo
        de conexion 

        """
        if isinstance(filename,str):
            if Path(filename).suffix == '.ini':
                parameters = self.file_ini_(filename=filename, section=section)

            elif Path(filename).suffix == '.yaml':
                parameters = self.file_yaml(filename=filename, section=section)

                raise ValueError(
                    f"Archivo no válido: {filename}. Sólo se permiten archivos .ini o .yaml.")

        elif isinstance(filename,dict):
            parameters = filename[section]

        return parameters

    def check_connection(self, connection_parameters: str):
        """ Connect to the PostgreSQL database server """
        conn = None
        try:
            # read connection parameters
            params = self.get_config_file(connection_parameters)

            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(**params)

            # create a cursor
            cur = conn.cursor()

        # execute a statement
            print('PostgreSQL database version:')
            cur.execute('SELECT version()')

            # display the PostgreSQL database server version
            db_version = cur.fetchone()
            print(db_version)

        # close the communication with the PostgreSQL
            cur.close()
            self.log.info('Query de verificacion de conexion de base de datos')
        except (Exception, psycopg2.DatabaseError) as error_check:
            self.log.info(error_check)
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')


    def read_parameters_query(self, file_parametro: str):
        '''Funcion para leer un json y convertirlo en un diccionario
        para ser usado posteriormente en el metodo fix_dict_query'''

        with open(file_parametro, 'r', encoding='utf-8') as file:
            data = json.load(file)

        data['columns'] = list(data['columns'].values())

        return data
    
   
    def fix_dict_query(self, tabla: str, columnas: list, order: str, where: str):
        '''fix_dict_query Funcion para prepara los parametros del diccionario busqueda 
        para la funcion prepare_query_replace_value, demanrea que se pueda hacer querys
        '''

        data_replace = {
            'table': tabla,
            'columns': ', '.join(['"' + columna + '"' for columna in columnas]),
            '_insert': ', '.join(['%s' for _ in columnas]),
            'order': order,
            'where': where
        }
        return data_replace

    def query_data(self, connection, sql):
        """
        Realizar un query de insercion de datos en funcion de los datos
        Retorna 1 si la operacion se completa exitosamente 
        Retorna 0 si la operacion no se completa exitosamente
        """
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                # cursor.commit()
                self.log.info('[INFO] Ejecucion de consulta realizada')
                return 1
        except psycopg2.DatabaseError as error_quey_data:
            self.log.info('[ERROR] Fallo de ejecucion del query')
            self.log.info(error_quey_data)
            return 0

    def prepare_query_replace_value(self, sql_file: str, data_replace: dict):
        '''prepare_query Cargar y modificar Query y preparala para enviarla
        modificando valores del string generado por usando un diccionario 
        depormedio

        Args:
            sql_file (str): Ruta de archivo de SQL con la query a realizar
            data_replace (dict): _description_

        Returns:
            _type_: String con la transformacion del query
        '''

        try:
            with open(sql_file, encoding='utf-8') as f:
                lines = f.readlines()
                sql_statements = ''
                for lin in lines:
                    sql_statements += lin
                sql_statements = sql_statements.replace('\n', ' ')
                sql_statements = sql_statements.replace('\t', ' ')

            for field, replace in data_replace.items():
                sql_statements = sql_statements.replace(field, str(replace))
            return sql_statements

        except Exception as erro_replace_value:
            self.log.error('Fallo de preparacion de la query')
            self.log.error(erro_replace_value)

    def prepare_query(self, sql_file: str):
        '''prepare_query Cargar y modificar Query y preparala para enviarla

        Args:
            sql_file (str): Ruta de archivo de SQL con la query a realizar

        Returns:
            _type_: String con la transformacion del query
        '''

        try:
            with open(sql_file, encoding='utf-8') as f:
                lines = f.readlines()
                sql_statements = ''
                for lin in lines:
                    sql_statements += lin
                sql_statements = sql_statements.replace('\n', ' ')
                sql_statements = sql_statements.replace('\t', ' ')
            return sql_statements
        except Exception as error_prepare_query:
            self.log.error('Fallo de preparacion de la query')
            self.log.error(error_prepare_query)

    def send_query(self, connection_parameters: str, query: str):
        '''send_query Enviar a ejecutar un Query a la base de datso

        Args:
            connection_parameters (str): parametros de conexion a la base de datos
            query (str): query lista para enviarla a la base de datos
        '''

        conn = None
        try:
            # read the connection parameters
            params = self.get_config_file(connection_parameters)
            # connect to the PostgreSQL server
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            # create table one by one
            cur.execute(query)
            # close communication with the PostgreSQL database server
            cur.close()
            # commit the changes
            conn.commit()
            self.log.debug("Query enviada")
        except (Exception, psycopg2.DatabaseError) as error_send:
            self.log.error(error_send)
        finally:
            if conn is not None:
                conn.close()
            self.log.debug("Finalizacion query")

    def insert_data_from_csv(self, connection_parameters: str, query: str, data_path: str):
        '''insert_data_from_csv Funcion para insertar un archivos csv a un tabla 
        especificada 

        Args:
            connection_parameters (str): Ruta del archivo con parametros de conexion
            query (str): Query usada para insertar la informacion en la DB
            data_path (str): Ruta del archivo de texto plano que se desea insertar en
            postgres
        '''
        conn = None
        try:
            # read database configuration
            params = self.get_config_file(connection_parameters)
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()

            data_to_send = pd.read_csv(data_path)
        
            for data_row in data_to_send.values:
                # execute the INSERT statement
                cur.execute(query, (data_row))

            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
            self.log.info('Query de insercion de archivo csv finalizada')
        except (Exception, psycopg2.DatabaseError) as error_insert_data_csv:
            self.log.error('Fallo de insercion de la query')
            self.log.error(error_insert_data_csv)
        finally:
            if conn is not None:
                conn.close()

    def insert_data_from_dataframe(self, connection_parameters: str, query: str, dataframe: pd.DataFrame):
        '''insert_data_from_csv Funcion para insertar un archivos csv a un tabla 
        especificada 

        Args:
            connection_parameters (str): Ruta del archivo con parametros de conexion
            query (str): Query usada para insertar la informacion en la DB
            data_path (str): Ruta del archivo de texto plano que se desea insertar en
            postgres
        '''
        conn = None
        try:
            # read database configuration
            params = self.get_config_file(connection_parameters)
    
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
    
            # create a new cursor
            cur = conn.cursor()
    
            for data_row in dataframe.values:
                # execute the INSERT statement
                cur.execute(query, (data_row))
                print('insertando')
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
            self.log.info('Query de insercion de archivo csv finalizada')
        except (Exception, psycopg2.DatabaseError) as error_insert_data_frame:
            self.log.error('Fallo de insercion de la query')
            self.log.error(error_insert_data_frame)
        finally:
            if conn is not None:
                conn.close()


    def insert_data(self, connection_parameters: str, query: str, data: tuple):
        '''insert_data Insertar un unico valor segun los una query

        Args:
            connection_parameters (str): Parametros de conexion a la base de datso
            query (str): query lista para enviarla a la base de datos
            data (tuple): datos que se tienen que insertar en la base de datos

        Returns:
            _type_: _description_
        '''

        conn = None
        try:
            # read database configuration
            params = self.get_config_file(connection_parameters)
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(query, (data))

            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
            self.log.info('Query de insercion unica finalizada')
        except (Exception, psycopg2.DatabaseError) as error_insert_data:
            self.log.error('Fallo de insercion de la query')
            self.log.error(error_insert_data)
        finally:
            if conn is not None:
                conn.close()

    def get_last_row(self, connection_parameters: str, query: str):
        '''get_last_row Metodo para obtener el ultimo valor de una base de datos

        Args:
            connection_parameters (str): parametros de conexion
            query (str): _description_

        Returns:
            _type_: _description_
        '''
        conn = None
        try:
            # read database configuration
            params = self.get_config_file(connection_parameters)
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(query)

            ultima_fila = cur.fetchone()

            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()

            return list(ultima_fila)
        except (Exception, psycopg2.DatabaseError) as error_get_last:
            self.log.error(error_get_last)
        finally:
            if conn is not None:
                conn.close()

    def get_table(self, connection_parameters: str, query: str):
    # def get_table(self, params, query: str):
        '''get_table Funcion para extraer la tabla completa de una base de datos
        usando una query

        Args:
            connection_parameters (str): Parametros de conexion a la base de datso
            query (str): query lista para extraer la base de datos
        '''
        conn = None
        try:
            # read database configuration
            params = self.get_config_file(connection_parameters)
            
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(query)

            ultima_fila = cur.fetchall()

            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()

            return pd.DataFrame(list(ultima_fila))
        except (Exception, psycopg2.DatabaseError) as error_get_table:
            self.log.error(error_get_table)
        finally:
            if conn is not None:
                conn.close()
