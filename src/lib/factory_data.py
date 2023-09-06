'''
Codigo usado para extraer la informacion para la predicciones de la informacion
'''
# Esta es la interfaz abstracta para las operaciones
import os
from pathlib import Path
import pandas as pd
try:
    # from src.features.features_redis import HandleRedis
    from src.features.features_postgres import HandleDBpsql
    from src.models.args_data_model import ParamsPostgres, Parameters
except ImportError as Error:
    # from features_redis import HandleRedis
    from features_postgres import HandleDBpsql
    from args_data_model import Parameters

from abc import ABC, abstractmethod


class DataSource(ABC):
    '''DataSource Metodos base para los objetos de datos '''

    @abstractmethod
    def read(self):
        '''read Metodo base de lectura '''

    @abstractmethod
    def write(self, data):
        '''write Metodo de escritura base '''

    @abstractmethod
    def create(self):
        '''Metodo para crear tablas en base de datos'''

# Implementaciones concretas de la interfaz para cada tipo de fuente de datos
# The `SQLPostgres` class is a Python class that provides methods for manipulating data in a
# PostgreSQL database.
class SQLPostgres(DataSource):
    '''Metodo para manipulacion de datos de postgres'''

    def __init__(self, **parametros) -> None:
        _file_path = Path(__file__).parents[2]
        self.parametro = Parameters(**parametros)
        self.data_source = HandleDBpsql()
        self.query_read = _file_path.joinpath(
            'src/data/querys/get_table.sql').as_posix()
        self.query_write = _file_path.joinpath(
            'src/data/querys/insert_data.sql').as_posix()
        self.query_create = _file_path.joinpath(
            'src/data/querys/new_table.sql').as_posix()

    def read(self):
        """
        The `read` method reads data from a data source using a query template and returns the resulting
        table.

        Returns:
          The method is returning the result of the `get_table` method from the `data_source` object.
        metodo base para hacer lectura de los datos
        """
        fix_dict_query = self.data_source.fix_dict_query(
            self.parametro.query_template['table'],
            list(self.parametro.query_template['columns'].values()),
            self.parametro.query_template['order'],
            self.parametro.query_template['where']
        )

        query = self.data_source.prepare_query_replace_value(
            sql_file=self.query_read,
            data_replace=fix_dict_query)

        return self.data_source.get_table(
            connection_parameters=self.parametro.connection_data_source,
            query=query
        )

    def write(self, data: pd.DataFrame):
        """
        The `write` method is a base method for writing data to a PostgreSQL database using a template
        query.

        Args:
          data (pd.DataFrame): The `data` parameter is a pandas DataFrame that contains the data to be
        written to a PostgreSQL database.
        metodo base para hacer escritura de los datos en postgres
        """

        fix_dict_query = self.data_source.fix_dict_query(
            self.parametro.query_template_write['table'],
            list(self.parametro.query_template_write['columns'].values()),
            self.parametro.query_template_write['order'],
            self.parametro.query_template_write['where']
        )

        query = self.data_source.prepare_query_replace_value(
            sql_file=self.query_write,
            data_replace=fix_dict_query)

        self.data_source.insert_data_from_dataframe(
            connection_parameters=self.parametro.connection_data_source,
            dataframe=data,
            query=query
        )

    def create(self):
        """
        The `create` method is used to create tables in a PostgreSQL database based on the column names
        and data types specified in a YAML file.

        Metodo para crear tablas
        """


        conver_postgrest = {
            'date': 'DATE',
            'integer': 'NUMERIC(15,3)',
            'float': 'DECIMAL(12,3)',
            'string': 'VARCHAR(50)',
        }
        # print(self.parametro)
        convert_value = {}

        for key_, val_ in self.parametro.type_data_out.items():
            convert_value[key_] = conver_postgrest[val_]

        # Aquí es donde construirás las declaraciones para las columnas.
        column_declarations = []

        # Obtén los nombres de las columnas y los tipos de datos del yaml.
        # column_names = self.parametro.query_template_write['columns']
        data_types = self.parametro.type_data_out


        # Ahora, crea las declaraciones para las columnas usando los nombres y los tipos de datos.
        for key, val in data_types.items():
            column_declarations.append(f'"{key}" {conver_postgrest[val]}')

        fix_data_dict = self.parametro.query_template_write.copy()
        fix_data_dict['columns'] = ",\n".join(column_declarations)

        # self.parametro.query_template_write['columns'] = ",\n".join(column_declarations)
        # self.parametro.query_template_write['table'] = str(self.parametro.filter_data['filter_1_feature'])

        query = self.data_source.prepare_query_replace_value(
            sql_file=self.query_create,
            data_replace=fix_data_dict
        )

        self.data_source.send_query(
            connection_parameters=self.parametro.connection_data_source,
            query=query
        )


class SQLserver(DataSource):
    '''Metodo de manipulacion de datos de sql server'''
    def __init__(self) -> None:
        """ ToDo"""

    def read(self):
        """ ToDo"""


    def write(self, data):
        """ ToDo"""

    def create(self):
        """ ToDo"""


class NoSQLRedis(DataSource):
    '''Metodo para manipulacion de datos de redis'''

    def read(self):
        return "Datos leídos desde la base de datos NoSQL"

    def write(self, data):
        return "Datos escritos en la base de datos NoSQL"

    def create(self):
        return "Fuente de datos en la base de datos NoSQL"


class PlainTextFileDataSource(DataSource):
    '''Metodo para manipulacion de datos tipo texto plano'''

    def read(self):
        return "Datos leídos desde el archivo de texto plano"

    def write(self, data):
        return "Datos escritos en el archivo de texto plano"


class APIinterface(DataSource):

    def __init__(self) -> None:
        """ ToDo"""

    def read(self):
        """ ToDo"""



# Esta es la interfaz abstracta para la fábrica
class AbstractDataSourceFactory(ABC):
    '''Fabrica para obtener fuentes de datos'''
    @abstractmethod
    def create_data_source(self):
        '''Creacion de la fuente de datos seleccionada'''

# Implementaciones concretas de la fábrica para cada tipo de fuente de datos


class SQLDataSourceFactory(AbstractDataSourceFactory):
    '''Fabricante de metodos SQL '''

    def __init__(self, **parametros) -> None:
        self.parametro = parametros

    def create_data_source(self) -> DataSource:
        return SQLPostgres(**self.parametro)


class NoSQLDataSourceFactory(AbstractDataSourceFactory):
    '''Fabricante de metodos NoSQL '''

    def __init__(self, **parametros) -> None:
        self.parametro = parametros


    def create_data_source(self) -> DataSource:
        return NoSQLRedis()


# class PlainTextFileDataSourceFactory(AbstractDataSourceFactory):
#     '''Fabricante de metodos de texto plano '''

#     def create_data_source(self) -> DataSource:
#         return PlainTextFileDataSource()

# Usando la fábrica
def get_data(factory: AbstractDataSourceFactory) -> None:
    '''Metodo para hacer la lectura de la infomacion'''
    data_source = factory.create_data_source()
    return data_source.read()


def set_data(factotory: AbstractDataSourceFactory, data) -> None:
    '''Metodo para hacer la escritura en la base de datos'''
    data_source = factotory.create_data_source()
    return data_source.write(data)


def create_table(factory: AbstractDataSourceFactory) -> None:
    '''Metodo para hacer la lectura de la infomacion'''
    data_source = factory.create_data_source()
    return data_source.create()


if __name__ == "__main__":
    print("Probando el código con la base de datos SQL:")
    get_data(SQLDataSourceFactory())

    print("\nProbando el código con la base de datos NoSQL:")
    get_data(NoSQLDataSourceFactory())

    # print("\nProbando el código con el archivo de texto plano:")
    # get_data(PlainTextFileDataSourceFactory())
