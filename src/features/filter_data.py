'''
Codigo para generar un sistema de alertas usando patrones de diseno, se aplica tanto a los datos
de inventario real como a los datos de las predicciones de los modelos

Se usan el patron de diseno observer patter junto strategy patter para generar un sistema de alertas
que pueda se modificable en el tiempo
'''

from abc import ABC, abstractmethod
import datetime
from datetime import datetime as dtime
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from src.features.features_redis import HandleRedis

handler_redis = HandleRedis()

class Subject:
    '''The `Subject` class allows objects to attach, detach, and notify
       observers of changes in its state.
       '''

    def __init__(self):
        self._observers = []

    def attach(self, observer):
        """
        The function "attach" adds an observer to a list of observers.

        Args:
          observer: The "observer" parameter is an object that wants to be notified
          of changes in the state of the object that this method belongs to.
        """
        self._observers.append(observer)

    def detach(self, observer):
        """
        The detach function removes an observer from the list of observers.

        Args:
          observer: The observer parameter is an object that is currently
          observing or listening to changes in the subject.
        """
        self._observers.remove(observer)

    def notify(self):
        """
        The `notify` function iterates over a list of observers and calls their `update`
        method, passing in the current object.
        """
        for observer in self._observers:
            observer.update(self)


class Observer(ABC):
    '''Observer Constructor abstracto base para el observador '''
    @abstractmethod
    def update(self, subject):
        '''Actuliazar los observadores '''


class Alertas(ABC):
    '''Clase abstracta sistema de alertas'''
    @abstractmethod
    def eval(self, data):
        '''Constructor para evaluar alertas'''

# ===================================================================
#               Listado de alertas a evaluar en los datos
# ===================================================================


class AlertaPorBajaCantidad(Alertas):
    '''
    # The class "AlertaPorBajaCantidad" is a subclass of "Alertas" and it defines a method "eval" that
    # prints an alert message if the quantity of a product is below 50.
    '''

    def __init__(self, cantidad: str, item: str = 'modulo', column: str = '', config: dict = None) -> None:
        self.cantidad = cantidad
        self.column = column
        self.item = item
        self.config = config

    def eval(self, data):
        # Evalua el ultimo valor de los datos
        if data[self.column][-1] < self.cantidad:
            # aqui se genera el codigo para o el indicador de alerta para escribir en la base
            # de datos
            print("Alerta Activa: Baja cantidad de:", self.item)
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name="AlertaBajaCantidad",
                value=1)
        else:
            print("Alerta Desactivada: Baja cantidad de:", self.item)
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name="AlertaBajaCantidad",
                value=2)


class AlertasPorVariacionPreciosProveedores(Alertas):
    '''Alerta para cuando existe variacion en los precios de los proveedores'''


class AlertaPorVarianza(Alertas):
    '''Alerta por alta varianza en los ultimos valores'''
    # Esta alerta se aplica a los datos agrupados por semanas  aplicada multiples meses

    def __init__(self, threshold=0.1, item: str = 'modulo', column: str = '', config: dict = None):
        # threshold es el umbral que determina cuánto debe cambiar el coeficiente de variación
        # para que se dispare la alerta.
        self.item = item
        self.threshold = threshold
        self.column = column
        self.config = config

    def eval(self, data):
        # Evalua el ultimo valor de los datos
        if data[self.column][-1] < self.threshold:
            # aqui se genera el codigo para o el indicador de alerta para escribir en la base
            # de datos
            print("Alerta: Varianza cantidad de:", self.item)
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorVarianza',
                value=1
            )
        else:
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorVarianza',
                value=2
            )


class AlertaPorTiempoDeVentaBajo(Alertas):
    '''
    The `AlertaPorTiempoDeVentaBajo` class triggers an alert if a product is sold in fewer days than a
    specified threshold.
    '''

    def __init__(self, min_dias_venta: int = 7, item: str = 'modulo', config: dict = None):
        # min_dias_venta es el número mínimo de días que un producto debe durar en el inventario
        # antes de que se dispare la alerta
        self.min_dias_venta = min_dias_venta
        self.item = item
        self.config = config
        self.dias_ventas_ = None

    def eval(self, data):
        """
        The function evaluates the number of days it took to sell a product and triggers an alert if it
        is below a certain threshold.

        Args:
          init_date: The `init_date` parameter is the starting date of the sales period. It should be
        provided in the format 'YYYY-MM-DD'.
          end_date: The `end_date` parameter represents the date when the sales period ends.
          item: The 'item' parameter represents the name or identifier of the product being evaluated.
        """
        # Convierte las fechas de inicio y fin a objetos datetime
        if not isinstance(data.index[0], datetime.date):
            init_date = dtime.strptime(data.index[0], '%Y-%m-%d')
            end_date = dtime.strptime(data.index[-1], '%Y-%m-%d')
        else:
            init_date = data.index[0]
            init_date = datetime.datetime.combine(init_date, datetime.time())
            end_date = data.index[-1]
            end_date = datetime.datetime.combine(end_date, datetime.time())

        # Calcula la diferencia en días entre las fechas
        dias_venta = (end_date - init_date).days
        # Si la cantidad de días de venta es menor al umbral, se dispara la alerta
        if dias_venta < self.min_dias_venta:
            print(
                f"Alerta: El producto '{self.item}' se vendió en solo {dias_venta} días.")
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorTiempoDeVentaBajo',
                value=1
            )
        else:
            print(
                f"Alerta Desactivada: El producto '{self.item}' se vendió en solo {dias_venta} días.")
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorTiempoDeVentaBajo',
                value=2
            )


class AlertaPorCambiosBruscosEnLasVentas(Alertas):
    '''Alerta por cambio brusco en ventas'''

    def __init__(self, umbral_varianza, umbral_desviacion_estandar, item: str = 'modulo', config: str = None):
        self.umbral_varianza = umbral_varianza
        self.umbral_desviacion_estandar = umbral_desviacion_estandar
        self.item = item
        self.config = config

    def eval(self, data):
        varianza = data['varianza'].values[0]
        desviacion_estandar = data['desviacion_estandar'].values[0]
        if varianza > self.umbral_varianza:
            print(
                f"Alerta: Cambio brusco en ventas detectado. Varianza: {varianza}")
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorCambiosBruscosVarianza',
                value=1
            )
        else:
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorCambiosBruscosVarianza',
                value=2
            )
        if desviacion_estandar > self.umbral_desviacion_estandar:
            print(
                f"Alerta: Cambio brusco en ventas detectado. Desviación Estándar: {desviacion_estandar}")
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorCambiosBruscosDesviacionEstandar',
                value=1
            )
        else:
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorCambiosBruscosDesviacionEstandar',
                value=2
            )


class AlertaPorInventarioInactivo(Alertas):
    '''Alerta por tiempo inventario inactivo'''
    # Esta alerta se aplica a los datos por dias (sin agrupacion)

    def __init__(self, max_dias_inactivo=30, item: str = 'modulo', previus_val: float = 0, config: str = None):
        super().__init__()
        # max_dias_inactivo es el número máximo de días que un producto puede estar inactivo
        # antes de que se dispare la alerta
        self.max_dias_inactivo = max_dias_inactivo
        self.previus_stock = previus_val
        self.item = item
        self.config = config

    def eval(self, data):
        # Si es la primera vez que evaluamos, simplemente guardamos el valor actual de cantidad
        self.previus_stock = None if self.previus_stock == 'None' else self.previus_stock

        if self.previus_stock is None:
            print('Condicional valor previo')
            self.previus_stock = data['predicion'].values[-1]

        # '''Las fechas init_date y end_date deben ser cadenas en el formato 'AAAA-MM-DD'.'''
        # Convierte las fechas de inicio y fin a objetos datetime
        now = dtime.now()

        if not isinstance(data.index[-1], datetime.date):
            end_date = dtime.strptime(data.index[-1], '%Y-%m-%d')
        else:
            end_date = data.index[-1]

        end_date = datetime.datetime.combine(end_date, datetime.time())

        # Calcula la diferencia en días entre las fecha
        dias_inactivo = (now - end_date).days

        # Si la cantidad de días inactivos supera el umbral, se dispara la alerta
        if dias_inactivo > self.max_dias_inactivo and float(self.previus_stock) == float(data['predicion'].values[-1]):
            print(
                f"Alerta: El producto '{self.item}' ha estado inactivo por {dias_inactivo} días.")
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorInventarioInactivo',
                value=1
            )
        else:
            print(
                f"Alerta Desactivasda: El producto '{self.item}' ha estado inactivo por {dias_inactivo} días.")
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorInventarioInactivo',
                value=2
            )


class AlertaPorSeguimientoTendencias:
    '''
    # The `AlertaPorSeguimientoTendencias` class is designed to evaluate the trend of a specific item
    # based on the coefficient of variation in a given dataset and generate an alert if the trend changes
    # significantly.
    '''

    def __init__(self, threshold=0.1, item='modulo', config: dict = None):
        self.item = item
        self.threshold = threshold
        self.config = config

    def eval(self, data: pd.DataFrame):
        """
        The `eval` function compares the current and previous values of a coefficient of variation in a
        DataFrame and prints an alert if the relative change exceeds a threshold.

        Args:
          data (pd.DataFrame): The `data` parameter is expected to be a pandas DataFrame containing the
        sales data. It is assumed that the DataFrame is sorted by date, with the last row representing the
        current values and the second-to-last row representing the previous values. The DataFrame should
        have a column named 'coeficiente_varianza
        """
        # Suponiendo que los datos están ordenados por fecha y que la última fila contiene los valores actuales
        # y la penúltima fila contiene los valores anteriores
        if data.shape[0] >=2:
            current_row = data.iloc[-1]
            previous_row = data.iloc[-2]

            coeficiente_varianza = current_row['coeficiente_varianza']
            previous_coeficiente_varianza = previous_row['coeficiente_varianza']

            cambio = coeficiente_varianza - previous_coeficiente_varianza
            cambio_relativo = abs(cambio) / previous_coeficiente_varianza

            if cambio_relativo > self.threshold:
                tendencia = "a la alza" if cambio > 0 else "a la baja"
                print(
                    f"Alerta: Cambio significativo en la tendencia de ventas detectado ({tendencia}).")
                print("Coeficiente de variación anterior:",
                    previous_coeficiente_varianza)
                print("Coeficiente de variación actual:", coeficiente_varianza)

                handler_redis.set_single_value(
                    dict_key=self.item,
                    config=self.config,
                    file_name='AlertaPorSeguimientoTendencias',
                    value=1
                )
            else:
                print('No hay cambios')
                handler_redis.set_single_value(
                    dict_key=self.item,
                    config=self.config,
                    file_name='AlertaPorSeguimientoTendencias',
                    value=2
                )

            if cambio > 0:
                handler_redis.set_single_value(
                    dict_key=self.item,
                    config=self.config,
                    file_name='AlertaPorSeguimientoTendenciasCreciente',
                    value=2
                )
            else:
                handler_redis.set_single_value(
                    dict_key=self.item,
                    config=self.config,
                    file_name='AlertaPorSeguimientoTendenciasCreciente',
                    value=3
                )
        else:
            print('Datos insuficientes')
            handler_redis.set_single_value(
                    dict_key=self.item,
                    config=self.config,
                    file_name='AlertaPorSeguimientoTendencias',
                    value=4
                )

class AlertaPorDemandaEstacional(Alertas):
    '''Alerta para saber cuando una producot esta en demanda estacional'''
    # Esta alerta se aplica a los datos agrupados por semanas aplicada multiples meses

    def __init__(self, item, threshold, column_time: str = '', column_value: str = '', config: dict = None) -> None:
        self.threshold = threshold
        self.item = item
        self.column_time = column_time
        self.column_value = column_value
        self.config = config

    def eval(self, data):
        # Realiza la descomposición estacional
        # puedes ajustar el período según tus necesidades
        if data.shape[0] <= 8:
            print(
                f"Alerta: Demanda estacional detectada para el artículo {self.item}. Datos insuficiente"
            )
            handler_redis.set_single_value(
                dict_key=self.item,
                config=self.config,
                file_name='AlertaPorDemandaEstacional',
                value=4
            )
        else:
            result = seasonal_decompose(
                data[self.column_value],
                model='additive',
                period=4
            )
            # Supongamos que quieres disparar una alerta si la amplitud de la componente estacional supera un umbral
            seasonal_amplitude = result.seasonal.max() - result.seasonal.min()
            if seasonal_amplitude > self.threshold:  # ajusta este umbral según tus necesidades
                print(
                    f"Alerta: Demanda estacional detectada para el artículo {self.item}. Amplitud: {seasonal_amplitude}")
                handler_redis.set_single_value(
                    dict_key=self.item,
                    config=self.config,
                    file_name='AlertaPorDemandaEstacional',
                    value=1
                )
            else:
                print('No hay cambios')
                handler_redis.set_single_value(
                    dict_key=self.item,
                    config=self.config,
                    file_name='AlertaPorDemandaEstacional',
                    value=2
                )
# ===================================================================
#          Clase concreta de Observer para utilizar la alerta
# ===================================================================


class AlertaObserver(Observer):
    '''
    The `AlertaObserver` class is an implementation of the Observer pattern that updates an
       `Alertas` object based on changes in a subject.
    '''

    def __init__(self, alerta: Alertas) -> None:
        self.alerta = alerta

    def update(self, subject):
        self.alerta.eval(subject.data)


class Inventario(Subject):
    '''
    La clase `Inventario` representa un inventario con varios elementos.
    '''

    def __init__(self, inventario) -> None:
        '''
        Args:
          inventario: Una lista de diccionarios que representan los elementos en el inventario.
                      Cada diccionario debe tener claves que coincidan con los datos que las alertas necesitan evaluar.
        '''
        super().__init__()
        self.inventario = inventario

    def evaluar_historico(self, init_data, end_date):
        '''Evalúa todos los elementos en el inventario con todas las alertas.'''
        try:
            self.inventario = self.inventario[init_data:end_date]
        except TypeError as date_error:
            print('[Error] Fecha fuera de rango:',date_error)
        for alerta in self._observers:
            # Aquí puedes pasar los datos necesarios dependiendo de cómo estén estructuradas tus alertas
            if isinstance(alerta, AlertaPorBajaCantidad):
                if alerta.alerta.eval(self.inventario[end_date:]):
                    self.notify()
            if isinstance(alerta, AlertaPorInventarioInactivo):
                if alerta.alerta.eval(self.inventario[end_date:]):
                    self.notify()
            if alerta.alerta.eval(self.inventario):
                self.notify()

    def evaluar_metricas(self):
        '''Evaluar las alertas '''

        for alerta in self._observers:
            if alerta.alerta.eval(self.inventario):
                self.notify()

# if __name__ == "__main__":
# ===================================================================
#     alerta_concreta = AlertaPorBajaCantidad()
#     observer_concreto = AlertaObserver(alerta_concreta)

#     DATA_SOURCE_TEST = None
#     inventario = Invertario(data_source=DATA_SOURCE_TEST,modelo = 'ModeloA',cantidad = 100)
#     inventario.attach(observer_concreto)

#     inventario.vender(60)
#     inventario.vender(20)

# ===================================================================
# Implementacion y uso de alertas para cambio brusco en las ventas
# data = {'varianza': 100, 'desviacion_estandar': 15}
# alerta_cambios_bruscos = AlertaPorCambiosBruscosEnLasVentas(umbral_varianza=50, umbral_desviacion_estandar=10)
# observer_cambios_bruscos = AlertaObserver(alerta_cambios_bruscos)

# # Asumiendo que tienes una instancia de la clase Inventario
# inventario.attach(observer_cambios_bruscos)

# # Ahora, puedes evaluar la alerta con los datos
# alerta_cambios_bruscos.eval(data)


# ===================================================================
# Ejemplo de uso de dias inactivo
# alerta_inventario_inactivo = AlertaPorInventarioInactivo()
# alerta_inventario_inactivo.eval('2022-01-01', '2022-02-15', 'SME5-90-24')


# ===================================================================
# Ejemplo de uso de Tiempo de venta bajo
# alerta_tiempo_venta_bajo = AlertaPorTiempoDeVentaBajo()
# alerta_tiempo_venta_bajo.eval('2022-01-01', '2022-01-05', 'SME5-90-24')


# ===================================================================
# Crear una alerta para cuando
# alerta_baja_cantidad = AlertaPorBajaCantidad(parameter_alerts.alerta_bajacantidad,'cantidad')
# observer_alerta = AlertaObserver(alerta_baja_cantidad)

# alerta_tiempo_de_venta_bajo = AlertaPorTiempoDeVentaBajo()
# observer_tiempo_de_venta_bajo = AlertaObserver(alerta_tiempo_de_venta_bajo)


# # Crear el inventario inicial
# inventario_inicial = [
#     {'modelo': 'manzana', 'cantidad': 50,'tiempo_de_venta' : 4},
#     {'modelo': 'banana', 'cantidad': 10,'tiempo_de_venta':4},
# ]

# # Crear una instancia de Inventario y adjuntar el observador
# inventario = Inventario(inventario_inicial)
# inventario.attach(observer_alerta)
# inventario.attach(observer_tiempo_de_venta_bajo)

# # Agregar la alerta al inventario
# inventario.agregar_alerta(alerta_baja_cantidad)
# inventario.agregar_alerta(alerta_tiempo_de_venta_bajo)

# # Evaluar el inventario
# inventario.evaluar_inventario()

# ===================================================================
