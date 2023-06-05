'''
Codigo usado para extraer la informacion para la predicciones de la informacion
'''

from src.features.features_redis import HandleRedis
from src.features.features_postgres import HandleDBpsql
from src.models.args_data_model import ParamsPostgres

from abc import ABC, abstractmethod

# Esta es la interfaz abstracta para las operaciones


class DataSource(ABC):
    '''DataSource Metodos base para los objetos de datos '''

    @abstractmethod
    def read(self):
        '''read Metodo base de lectura '''

    @abstractmethod
    def write(self, data):
        '''write Metodo de escritura base '''

# Implementaciones concretas de la interfaz para cada tipo de fuente de datos
class SQLPostgres(DataSource):
    '''Metodo para manipulacion de datos de postgres'''

    def __init__(self, **parametros) -> None:
        self.parametro = ParamsPostgres(**parametros)
        self.data_source = HandleDBpsql(parametros['logger_file'])

    def read(self):
        '''metodo base para hacer lectura de los datos'''
        parameter_query = self.data_source.read_parameters_query(self.parametro.names_table_columns)
        fix_dict_query = self.data_source.fix_dict_query(
                        parameter_query['table'],
                        parameter_query['columns'],
                        parameter_query['order'],
                        parameter_query['where']
                        )
        query = self.data_source.prepare_query_replace_value(
            sql_file=self.parametro.logger_file, data_replace=fix_dict_query)
        
        return self.data_source.get_table(
            connection_parameters=self.parametro.connection_params, query=query)

    def write(self):

        '''metodo base para hacer escritura de los datos en postgres'''
        # query = self.data_source.prepare_query_replace_value()
        return "Datos escritos en la base de datos SQL"


class NoSQLRedis(DataSource):
    '''Metodo para manipulacion de datos de redis'''

    def read(self):
        return "Datos leídos desde la base de datos NoSQL"

    def write(self, data):
        return "Datos escritos en la base de datos NoSQL"


class PlainTextFileDataSource(DataSource):
    '''Metodo para manipulacion de datos tipo texto plano'''

    def read(self):
        return "Datos leídos desde el archivo de texto plano"

    def write(self, data):
        return "Datos escritos en el archivo de texto plano"


# Esta es la interfaz abstracta para la fábrica
class AbstractDataSourceFactory(ABC):
    '''Fabrica para obtener fuentes de datos'''
    @abstractmethod
    def create_data_source(self):
        '''Creacion de la fuente de datos seleccionada'''


# Implementaciones concretas de la fábrica para cada tipo de fuente de datos
class SQLDataSourceFactory(AbstractDataSourceFactory):
    '''Fabricante de metodos SQL '''

    def create_data_source(self) -> DataSource:
        return SQLPostgres()


class NoSQLDataSourceFactory(AbstractDataSourceFactory):
    '''Fabricante de metodos NoSQL '''

    def create_data_source(self) -> DataSource:
        return NoSQLRedis()


class PlainTextFileDataSourceFactory(AbstractDataSourceFactory):
    '''Fabricante de metodos de texto plano '''

    def create_data_source(self) -> DataSource:
        return PlainTextFileDataSource()

# Usando la fábrica


def get_data(factory: AbstractDataSourceFactory) -> None:
    '''Metodo para hacer la lectura de la infomacion'''
    data_source = factory.create_data_source()
    print(data_source.read())
    print(data_source.write("datos"))


if __name__ == "__main__":
    print("Probando el código con la base de datos SQL:")
    get_data(SQLDataSourceFactory())

    print("\nProbando el código con la base de datos NoSQL:")
    get_data(NoSQLDataSourceFactory())

    print("\nProbando el código con el archivo de texto plano:")
    get_data(PlainTextFileDataSourceFactory())
