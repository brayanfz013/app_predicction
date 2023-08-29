'''
Clases y metodos para limpiar la informacion para hacer la prepracion de los datos para el modelo
'''
import numpy as np
import pandas as pd
import json
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from matplotlib import pyplot as plt
from darts.metrics import r2_score

class ColumnsNameHandler:
    ''' Metodo para manipulas el nombre de las columnas de un dataframe de pandas    '''

    def __init__(self,dataframe:pd.DataFrame,**parameters) -> None:
        
        # self.dataframe =dataframe
        if isinstance(dataframe,pd.DataFrame):
            old = list(dataframe.columns)
            new = list(parameters['columns'].values())

            if new  != old and len(new) == len(old):
                rename_cols = {x:y for x,y in zip(old,new)}  
                self.dataframe = dataframe.rename(columns=rename_cols)
            else:
                self.dataframe = dataframe
        else:
            # with open(parameters["names_table_columns"] , 'r', encoding='utf-8') as file:
            #     names = json.load(file);
            
            names_columns = list(parameters['columns'].values())
            self.dataframe = pd.DataFrame(dataframe,columns=names_columns)

    def apply_transformations(self, transformations:dict,functions_transform:dict):
        '''apply_transformations Metodo para aplicar transformacion a cada columna de un
        dataframe 

        Args:
            transformations (dict): Diccionario con los nombres de las columnas y los
            tipos de datos que a lo que se van a convertir 

            {
                'Fecha Creación': numpy.datetime64,
                'Material (Cod)': int,
                'Peso neto (TON)': int,
                'Solicitate (Cod)': 'object',
                'Cant Pedido UMV': int,
                'Valor Neto': int
            }
            functions_transform (dict): diccionario con los metodos que se van a usar 
            para transformar las columnas 

            {
                int:lambda x: int(float(x.replace(',',''))),
                float:lambda x: float(x.replace(',',''))
                # object:pass
            }
        '''
        for col, dtype_data in transformations.items():
            try:
                self.dataframe[col] = self.dataframe[col].apply(functions_transform[dtype_data])
            except (KeyError,ValueError,AttributeError) as _:
                pass

    def fill_na_columns(self,strategy:dict):
        '''Metodo para rellena las columnas '''

        for dtype, _ in strategy.items():
            cols_to_transform = self.dataframe.select_dtypes(include=[dtype]).columns
            for col in cols_to_transform:
                if self.dataframe[col].isna().sum() != 0:
                    self.dataframe[col].fillna(self.dataframe[col].mode()[0],inplace =True)

    def set_data_column(self,col_name:str,formatp: str = '%m/%d/%Y')->pd.DataFrame:
        '''set_data_column Metodo para colocar la column de timpoe en el formato deseado

        Args:
            dataframe (pd.DataFrame): Dataframe de datos 
            col_name (str): nombre de la columna que se va cambiar
            format (str, optional): Formato de la fecha en funcion como dicta pandas. Defaults to '%m/%d/%Y'.

        Returns:
            pd.DataFrame: Dataframe con el formato de tiempo cambiado
        '''
        self.dataframe[col_name] = pd.to_datetime(self.dataframe[col_name], format=formatp)


    def get_dtypes_columns_update(self,new_type:list)->tuple:
        '''get_dtypes_updates Metodo para obtener los tipos de datos del dataframe
        y actulizarlos segun la lista de las columnas que se extraen

        Args:
            new_type (list): Lista con los valores que se van a actulizar 
            los tipos de datos

        Returns:
            _type_ Diccionario con los tipo de datos de dataframe que se va a actulizar
            {'A': {'int64': ''}, 'B': {'object': ''}, 'C': {'float64': ''}}
        '''
        dict_update = {col: {str(self.dataframe[col].dtype): ""} for col in self.dataframe.columns}
        transform_cols = {val : list(dict_update[val].values())[0] for val in dict_update}
        for i,new_type in zip(transform_cols,new_type):
            transform_cols[i] = new_type
        return (dict_update,transform_cols)

    def update_dtypes(self,update_dtypes:dict):
        '''Metodo para actualizar los tipos de datos de la columnas'''
        for col, dtype_col in update_dtypes.items():
            self.dataframe[col] = self.dataframe[col].astype(dtype_col)
        

    def check_values_range(self,dict_types:dict, min_val:float, max_val:float, dict_out:dict, key:str)->dict:

        """
        Funcion especifica para determinar si un valor esta dentro de un rango especificado
        Funcion ligada a variables_resize

        dict_types: diccionario donde es el rango a donde se desea evaluar
        min_val: valor minimo de un dataframe.describe()
        max_val: valor maximo de un dataframe.describe()
        dict_out: diccionario de salida usado para almacenar los valores los nuevo Dtypes
        key: key de salida usado para almacenar los valores los nuevo Dtypes
        """

        for ld_type  in list(dict_types.keys()):

            up = dict_types[ld_type][1]
            down = dict_types[ld_type][0]

            if down <= min_val < up  and down < max_val <= up:
                dict_out[key] = str(ld_type)
                break

        return dict_out

    def variables_resize(self,dataframe):
        """
        Funcion para minimizar el peso de las variables numericas de un dataframe
        dataframe: Diccionario de descripcion del dataframe de pandas 

        """    

        integer_signed = {
            "int8": {
                0:np.iinfo(np.int8).min,
                1:np.iinfo(np.int8).max
                },

            "int16":{
                0:np.iinfo(np.int16).min,
                1:np.iinfo(np.int16).max
                },

            "int32":{
                0:np.iinfo(np.int32).min,
                1:np.iinfo(np.int32).max
                },

            "int64":{
                0:np.iinfo(np.int64).min,
                1:np.iinfo(np.int64).max
                },
        }


        integer_unsigned = {

            "uint8":{
                0:np.iinfo(np.uint8).min,
                1:np.iinfo(np.uint8).max
                },
            
            "uint16":{
                0:np.iinfo(np.uint16).min,
                1:np.iinfo(np.uint16).max
                },

            "uint32":{
                0:np.iinfo(np.uint32).min,
                1:np.iinfo(np.uint32).max
                },

            "uint64":{
                0:np.iinfo(np.uint64).min,
                1:np.iinfo(np.uint64).max
                },
            }


        floating = {

            "float16":{
                0:np.finfo(np.float16).min,
                1:np.finfo(np.float16).max
                },

            "float32":{
                0:np.finfo(np.float32).min,
                1:np.finfo(np.float32).max
                },

            "float64":{
                0:np.finfo(np.float64).min,
                1:np.finfo(np.float64).max
                }   
            }

        #Seleccion de nombre de columnas
        features = dataframe.describe().columns
        
        #Dtypes de las variables numericas asignadas por pandas
        numerical_dtype = dataframe[features].dtypes

        #Valores maximos y minimos de las variables numericas
        max_values = dataframe.describe().loc['max']
        min_values = dataframe.describe().loc['min']

        new_dtypes = {}

        for cont, key in enumerate(features):

            #Valores maximos y minimos de la caracteristica que itera
            max_ft = max_values[key].astype(numerical_dtype[cont])

            min_ft = min_values[key].astype(numerical_dtype[cont])

            #Preguntar si es flotante, la caracteristica que se esta iterando
            if numerical_dtype[cont] in list(floating.keys()):

                #haga la comparacion con los tipos de datos float "floating"
                new_dtypes = self.check_values_range(floating, min_ft, max_ft, new_dtypes, key)

            #Preguntar si el valor minimo tiene valores negativos 
            elif min_ft < 0:

                #haga la comparacion con los tipo de datos uint : "integer_unsigned"
                new_dtypes = self.check_values_range(integer_unsigned, min_ft, max_ft, new_dtypes, key)

            #Preguntar si el valor minimo tiene valores negativos 
            elif min_ft >= 0:

                #haga la comparacion con los tipo de datos int : "integer_signed"
                new_dtypes = self.check_values_range(integer_signed, min_ft, max_ft, new_dtypes, key)
                
        return new_dtypes


class PrepareData(ColumnsNameHandler):
    '''Metodo para limpiar los datos de un data frame en funcionde una columna'''

    def __init__(self,dataframe:pd.DataFrame,**parameters) -> None:
        super().__init__(dataframe,**parameters)

    def set_index_col(self,column_index:str):
        '''Metodo personalizado para colocar una columna como index del dataframe'''
        self.dataframe.set_index(column_index,inplace=True)
        
    def approx_to_nearest_multiple(self,numero:float,mulitple:int)->int:
        '''approx_to_nearest_multiple Metodo para aproximas un valor a 
        a un multiplo especificado mas cercano

        Args:
            num (_type_): Numero que se desea aproximar
            mulitple (int): Multiplo al cual se requiere aproximar

        Returns:
            _type_: _description_
        '''
        return round(numero / mulitple)*mulitple

    def remove_characater(self,data_frame:pd, colum:str,character_remove:str):
        '''remove_characater Remove characateres diferentes de una columna de un dataframe

        Args:
            data_frame (pd): Dataframe con informacion 
            colum (str): nombre de la columna la cual se modifica
            character_remove (str): caracter que se requiere eliminar

        Returns:
            _type_: dataframe con caracter eliminado
        '''
        data_frame[colum] = data_frame[colum].apply(
            lambda x: int(float(x.replace(character_remove,''))))
        
        return data_frame
    
    def get_expand_date(self,column_date:str):
        '''get_expand_date metodo para expandir la fecha en multiples campos para el analisis

        Args:
            dataframe (pd.DataFrame): Dataframe con la columna fecha para la expancion
            column_date (str): nombre del columna que contiene la fecha

        Returns:
            _type_: Dataframe con la fecha expandida
        '''
        self.dataframe['Dia'] = self.dataframe[column_date].dt.day
        self.dataframe['Mes'] = self.dataframe[column_date].dt.month
        self.dataframe['Año'] = self.dataframe[column_date].dt.year
        self.dataframe['Semana'] = self.dataframe[column_date].dt.isocalendar().week
        self.dataframe['DiaSemana'] = self.dataframe[column_date].dt.day_of_week
        return self.dataframe

    def group_by_time(self,col_group:str,frequency_group:str = 'W'):
        '''group_by_time Metodo para agrupar las columnas de un dataframe dependiendo de la frecuencia
        Args:
            dataframe (pd.DataFrame): Datos para hacer hacer la agrupacion
            col_group (str): nombre de columna para hacer la agrupacion
            frequency_group (str, optional): Periodo de agrupacion Defaults to 'W'.
        '''
        self.dataframe = self.dataframe.groupby([pd.Grouper(freq=frequency_group)])[col_group].sum()
        self.dataframe = pd.DataFrame(self.dataframe)

    def fill_missing_values(self,data_convert:TimeSeries):
        '''scale_data Metodo para escarlar datos transformados en el pipe line de Darts

        Args:
            data_convert (TimeSeries): Listado de datos transformados por darts para ser escalados

        Returns:
            _type_: Serie de tiempo escalado para ser usados en las predicciones, junto con el escalador
        '''
        # Escalizacion de datos usando la libreria Darts
        filler = MissingValuesFiller()
        data_transform = filler.transform(data_convert)
        return data_transform

    def scale_data(self,data):
        # '''Metodo para escalar datos usando darts'''
        '''scale_data Metodo para escalar los datos usando darts, 
        a lo cual se retorna los valores escalados y el escalador 

        Args:
            data (_type_): Datos de series de tiempo de darts

        Returns:
            _type_: Se retorna los datos transformados y el escalador
        '''
        transformer = Scaler()
        train_transformed = transformer.fit_transform(data)
        return train_transformed, transformer

    def transforf_dataframe_dart(self,time_col:str,data_col:str):
        '''transforf_dataframe_dart Metodo para transformar la informacion para usarce en Dars

        Args:
            time_col (str): Columna que tiene la serie de tiempo sobre la cual se predice
            data_col (str): Columna que contiene informacion para las predicciones

        Returns:
            _type_: TimeSeries data convertida para usarla en modelos  de Darts
        '''
        data = self.dataframe.reset_index()
        data_for_model = TimeSeries.from_dataframe(
            data, time_col, [data_col]
        )
        return data_for_model
        
    def filter_column(self,column:str,feature:float,string_filter:bool=True):
        '''filter_column Metodo basico para filtrar los datos de un dataframe

        Args:
            column (str): Columna sobre la cual se aplica el filtro
            feature (typing.Optional): caracteristica sobre la cual se hace la busqueda 
            booleana
            string_filter (bool, optional): Bandera para filtrar entre caracteres numericos y caracteres
            . Defaults to True.

        Returns:
            pd.Dataframe: Retorna un dataframe filtrado en base a los parametros de entradas
        '''

        if string_filter:
            mask = self.dataframe[column].str.contains(str(feature))
            self.dataframe = self.dataframe[mask]

        else:
            mask = self.dataframe[column] == float(feature)
            self.dataframe = self.dataframe[mask]

    def split_data(self,data:TimeSeries,time_stamp:str):
        '''split_data Metodo para hacer un split de tiempo en base una fecha

        Args:
            data (TimeSeries): Datos convertidos previamente en series de tiempo de 
            darst
            time_stamp (str): string del fecha para hacer la particion de los datso
            '20230328'

        Returns:
            _type_: Datos 
        '''
        return data.split_after(pd.Timestamp(time_stamp))

    def metrics_column(self,dataframe:pd.DataFrame):
        '''metrics_column Calcular metricas en un columna

        Args:
            dataframe (pd.DataFrame): Columan de un dataframe sobre la cual se quieren extraer
            los valores seleccionados

        Returns:
            _type_: Diccionario con las metricas extraidas
        '''

        return {'Rango': round( dataframe.max() - dataframe.min(),3),
            'Varianza' : round( dataframe.var(),3),
            'Desviacion_estandar': round( dataframe.std(),3),
            'Coeficiente_varianza': round( (dataframe.std()/dataframe.mean())*100,3),#Resultaod en porcentaje)
            'Quantile Q1': round( dataframe.quantile(0.25),3),
            'Quantile Q3': round( dataframe.quantile(0.75),3),
            'InterQuantile' : round( dataframe.quantile(0.75)-dataframe.quantile(0.25),3),
            'Desviacion_media_absoluta': round((dataframe-dataframe.mean()).abs().mean(),3)
            }

    def display_forecast(self,pred_series, ts_transformed, forecast_type:str, start_date:str=None):
        '''display_forecast Metodo para hacer graficas y predicciones sobre datos a partir de 
        de la transformaciones

        Args:
            pred_series (_type_): Listado de predicciones realizada para ser mostradas
            ts_transformed (_type_): Serie de datos originales para hacer la comparativas
            forecast_type (_type_): Nombre de la grafica que va a tener
            start_date (_type_, optional):inicio de fecha para separar los datos para hacer el calculo de error
            Defaults to None.
        '''
        plt.figure(figsize=(8, 5))
        if start_date:
            ts_transformed = ts_transformed.drop_before(start_date)
        ts_transformed.univariate_component(0).plot(label="actual")
        pred_series.plot(label=("historic " + forecast_type + " forecasts"))
        plt.title(
            "R2: {}".format(r2_score(ts_transformed.univariate_component(0), pred_series))
        )
        plt.legend()

if __name__ == '__main__':

    from scipy import stats

    #Reempplazo de los tipo de datos por lo cuales 
    #se va a cambiar el dataframe
    new_types_ =[np.datetime64,int,int,'object',int,int]

    #Estrategias para imputar los datos faltantes de NA 
    strategy_ = {
        int:np.mean,
        float:np.mean,
        object:stats.mode
    }

    #metodo para transformar los tipo de datos
    replace = {
        int:lambda x: int(float(x.replace(',',''))),
        float:lambda x: float(x.replace(',',''))
    }

    data_for_process = pd.DataFrame()

    HandleColumns = ColumnsNameHandler(data_for_process)

    Columns_type = HandleColumns.get_dtypes_columns_update(new_types_)

    Columns_type = HandleColumns.data_type(Columns_type)

    for i,new_dtp in zip(Columns_type,new_types_):
        Columns_type[i] = new_dtp

    HandleColumns.apply_transformations(Columns_type,replace)

    HandleColumns.fill_na_columns(strategy_)

    HandleColumns.update_dtypes(Columns_type)