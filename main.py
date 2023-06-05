'''Codigo para la ejecucion  y llamando para hacer predicciones en los modelos'''


from app_prediction.src.lib.factory_data import client_code, SQLDataSourceFactory, NoSQLDataSourceFactory,PlainTextFileDataSourceFactory


print("Probando el código con la base de datos SQL:")
client_code(SQLDataSourceFactory())

# print("\nProbando el código con la base de datos NoSQL:")
# client_code(NoSQLDataSourceFactory())

# print("\nProbando el código con el archivo de texto plano:")
# client_code(PlainTextFileDataSourceFactory())
