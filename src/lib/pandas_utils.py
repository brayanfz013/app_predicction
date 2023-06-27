import numpy as np

def update_Dtypes_dataframe (dict, list) -> dict:
    """
    Dado un dataframe con unos dtypes definidos, se usa una lista para actulizar
    los tipos de datos del dataframe dado 
    
    dict: diccionario del dataframe a actualizar

    list: lista con los nuevos dtypes para actualizar 
    """
    
    cont = 0

    if len(dict) == len(list):

        for key in dict.keys():
  
            dict.update({key:list[cont]})
            cont += 1
            
        return dict
            
    else:

        raise Exception("Diccionario y lista no tienen las mismas dimensiones")


def convert_categorical_data(dicti, lista) -> dict:

    """
    Funsion para convertir columnas de un dataframe en categoricas 
    Tiene como parametro de entrada el diccionario a convertir 
    y una lista con las keys del dataset que se van a convertir en categoricas 
    """

    for key in lista:

        dicti[key] = dicti[key].astype("category")

    return dicti


def check_values_range(dict, min, max, dict_out, key):

    """
    Funcion especifica para determinar si un valor esta dentro de un rango especificado
    Funcion ligada a variables_resize

    dict: diccionario donde es el rango a donde se desea evaluar
    min: valor minimo de un dataframe.describe()
    max: valor maximo de un dataframe.describe()
    dict_out: diccionario de salida usado para almacenar los valores los nuevo Dtypes
    key: key de salida usado para almacenar los valores los nuevo Dtypes

    """

    for LDtype  in list(dict.keys()):

        up = dict[LDtype][1]
        down = dict[LDtype][0]

        if down <= min < up  and down < max <= up:

            dict_out[key] = str(LDtype) 

            break

    return dict_out



def variables_resize(dataframe):

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
            new_dtypes = check_values_range(floating, min_ft, max_ft, new_dtypes, key)

        #Preguntar si el valor minimo tiene valores negativos 
        elif min_ft < 0:

            #haga la comparacion con los tipo de datos uint : "integer_unsigned"
            new_dtypes = check_values_range(integer_unsigned, min_ft, max_ft, new_dtypes, key)

        #Preguntar si el valor minimo tiene valores negativos 
        elif min_ft >= 0:

            #haga la comparacion con los tipo de datos int : "integer_signed"
            new_dtypes = check_values_range(integer_signed, min_ft, max_ft, new_dtypes, key)
            
    return new_dtypes



def update_dtype_dataframe(dict,newtypes):

    """
    dict: diccionario al cual se le desean convertir las variables
    newtypes: diccionario con los nuevos tipos de datos que se deea convertir 
    """

    return dict.astype(newtypes)