# coding=utf-8
''' 
Metodo de preparacion de los datos de cada empresa
Varia en funcion de gobernanza de los datos y de la 
certeza que se tenga sobre los tipo de datos importado
'''

from abc import ABC, abstractmethod
import pandas as pd

try:
    from src.features.features_fix_data import PrepareData
    from src.lib.class_load import LoadFiles
except ImportError:
    from features_fix_data import PrepareData
    from class_load import LoadFiles


# Definir la interfaz de la estrategia
class DataCleaningStrategy(ABC):
    # @abstractmethod
    # def __init__(self,parameters) -> None:
        # self.parameters = parameters

    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

# Estrategia de limpieza básica: rellena los valores faltantes con la media
class MeanImputation(DataCleaningStrategy):
    '''MeanImputation Metodo de imputacion de parametros variables para cada 
    columna

    Args:
        DataCleaningStrategy (_type_): _description_
    '''

    def __init__(self,
                 replace_dtypes: list,
                 strategy_imputation: dict,
                 preprocess_function: dict,
                 **parameters
                 ) -> None:
        self.replace_dtypes = replace_dtypes
        self.strategy_imputation = strategy_imputation
        self.preprocess_function = preprocess_function
        self.parameters =  parameters

    def clean(self, data):
        handle_data = PrepareData(data,**self.parameters['query_template'])
        _, trans_col = handle_data.get_dtypes_columns_update(
            self.replace_dtypes)
        handle_data.apply_transformations(trans_col, self.preprocess_function)
        handle_data.fill_na_columns(self.strategy_imputation)
        handle_data.update_dtypes(trans_col)
        # return data.fillna(data.mean())
        return handle_data


# Estrategia de limpieza para outliers: reemplaza los outliers con el valor promedio del rango intercuartil
class OutliersToIQRMean(DataCleaningStrategy):
    '''OutliersToIQRMean Metodo base para remocion de outlines'''
    def __init__(self, **parameters) -> None:
        '''__init__ _summary_

        Args:
            filter_params (str): Archivo con los parametros para filtrado y removecion 
            de outliners
        '''    
        self.parameters = parameters
        # handle_loader = LoadFiles()
        # self.parameters_filter = handle_loader.json_to_dict(self.parameters["filter_columns"])[0]
        self.parameters_filter = parameters['filter_data']
        
    def clean(self, data):
        '''Metodo para remover los outlines'''

        handle_data = PrepareData(data,**self.parameters['query_template'])

        handle_data.filter_column(
            self.parameters_filter['filter_1_column'],
            self.parameters_filter['filter_1_feature'],
            string_filter=False
        )
        handle_data.filter_column(
            self.parameters_filter['filter_2_column'],
            self.parameters_filter['filter_2_feature'],
            string_filter=True
        )
        handle_data.get_expand_date(self.parameters_filter['date_column'])
        handle_data.set_index_col(self.parameters_filter['date_column'])
        handle_data.group_by_time(self.parameters_filter['predict_column'],
                                  frequency_group=self.parameters_filter['group_frequecy'])

        Q1 = handle_data.dataframe.quantile(0.25)
        Q3 = handle_data.dataframe.quantile(0.75)
        IQR = Q3 - Q1
        mean_iqr = (Q1 + Q3) / 2
        data_out = handle_data.dataframe[~((handle_data.dataframe < (Q1 - 1.5 * IQR)) |
                          (handle_data.dataframe > (Q3 + 1.5 * IQR)))]
        return data_out.fillna(mean_iqr)

class DataModel(DataCleaningStrategy):
    '''Clase para scalar y transformar los datos para cualquiermodelo de Darts'''

    def __init__(self,**parameters) -> None:
        # super().__init__()
        self.parameters = parameters
        self.parameters_filter = parameters['filter_data']

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        '''Metodo para transformar y scalar los datos de los modelos'''
        handler_data = PrepareData(data,**self.parameters['query_template'])
        print(handler_data.dataframe)

        time_series = handler_data.transforf_dataframe_dart(
            self.parameters_filter['date_column'],
            self.parameters_filter['predict_column']
            )
        filled_data = handler_data.fill_missing_values(time_series)

        return handler_data.scale_data(filled_data)
  
# Ahora puedes crear una clase de "contexto" que puede utilizar cualquiera de estas estrategias
class DataCleaner:

    def __init__(self, strategy: DataCleaningStrategy):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: DataCleaningStrategy):
        self._strategy = strategy

    def clean(self, data):
        return self._strategy.clean(data)


# if __name__ == '__main__':

    # cleaner = DataCleaner(MeanImputation())
    # clean_data = cleaner.clean('raw_data')

    # cleaner.strategy = OutliersToIQRMean()
    # clean_data = cleaner.clean('raw_data')