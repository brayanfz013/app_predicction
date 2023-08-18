''' 
Codigo para generar un sistema de alertas usando patrones de diseno, se aplica tanto a los datos
de inventario real como a los datos de las predicciones de los modelos

Se usan el patron de diseno observer patter junto strategy patter para generar un sistema de alertas
que pueda se modificable en el tiempo
'''

from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta


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
    def __init__(self, cantidad: str,item:str = 'modulo' ,**cols) -> None:
        self.cantidad = cantidad
        self.date = cols['0']
        self.model = cols['1']
        self.quantity = cols['2']
        self.customer = cols['3']

    def eval(self, data):
        #Evalua el ultimo valor de los datos 
        if data[self.quantity][-1] < self.cantidad:
            # aqui se genera el codigo para o el indicador de alerta para escribir en la base
            # de datos
            print("Alerta: Baja cantidad de:", item)

class AlertaPorTiempoDeVentaBajo(Alertas):
    '''
    The `AlertaPorTiempoDeVentaBajo` class triggers an alert if a product is sold in fewer days than a
    specified threshold.
    '''
    
    def __init__(self, min_dias_venta:int=7,item:str = 'modulo' ,**cols):
        # min_dias_venta es el número mínimo de días que un producto debe durar en el inventario
        # antes de que se dispare la alerta
        self.min_dias_venta = min_dias_venta
        self.date = cols['0']
        self.model = cols['1']
        self.quantity = cols['2']
        self.customer = cols['3']

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
        init_date = datetime.strptime(data['init_date'].values[0], '%Y-%m-%d')
        end_date = datetime.strptime(data['end_date'].values[0], '%Y-%m-%d')

        # Calcula la diferencia en días entre las fechas
        dias_venta = (end_date - init_date).days
        
        print('Dias de venta',dias_venta)
        # Si la cantidad de días de venta es menor al umbral, se dispara la alerta
        if dias_venta < self.min_dias_venta:
            print(
                f"Alerta: El producto '{item}' se vendió en solo {dias_venta} días.")


class AlertaPorCambiosBruscosEnLasVentas(Alertas):
    '''Alerta por cambio brusco en ventas'''

    def __init__(self, umbral_varianza, umbral_desviacion_estandar,item:str = 'modulo' ,**cols):
        self.umbral_varianza = umbral_varianza
        self.umbral_desviacion_estandar = umbral_desviacion_estandar
        self.date = cols['0']
        self.model = cols['1']
        self.quantity = cols['2']
        self.customer = cols['3']

    def eval(self, data):
        varianza = data['Varianza'].values[0]
        desviacion_estandar = data['Desviacion_estandar'].values[0]
        if varianza > self.umbral_varianza:
            print(
                f"Alerta: Cambio brusco en ventas detectado. Varianza: {varianza}")

        if desviacion_estandar > self.umbral_desviacion_estandar:
            print(
                f"Alerta: Cambio brusco en ventas detectado. Desviación Estándar: {desviacion_estandar}")


class AlertaPorInventarioInactivo(Alertas):
    '''Alerta por tiempo inventario inactivo'''
    # Esta alerta se aplica a los datos por dias (sin agrupacion)

    def __init__(self, max_dias_inactivo=30,item:str = 'modulo' ,**cols):
        # max_dias_inactivo es el número máximo de días que un producto puede estar inactivo
        # antes de que se dispare la alerta
        self.max_dias_inactivo = max_dias_inactivo
        self.previus_stock = None
        self.date = cols['0']
        self.model = cols['1']
        self.quantity = cols['2']
        self.customer = cols['3']

    def eval(self, data):
        # Si es la primera vez que evaluamos, simplemente guardamos el valor actual de cantidad       
        if self.previus_stock is None:
            self.previus_stock = data['mean'].values[0]
            # return

        '''Las fechas init_date y end_date deben ser cadenas en el formato 'AAAA-MM-DD'.'''
        # Convierte las fechas de inicio y fin a objetos datetime
        now = datetime.now()

        # real_day = datetime.strptime(now, '%Y-%m-%d')
        end_date = datetime.strptime(data['end_date'].values[0], '%Y-%m-%d')
        
        # Calcula la diferencia en días entre las fechas
        dias_inactivo = (now - end_date).days
    
        # Si la cantidad de días inactivos supera el umbral, se dispara la alerta
        if dias_inactivo > self.max_dias_inactivo and self.previus_stock == data['mean'].values[0]:
            print(
                f"Alerta: El producto '{item}' ha estado inactivo por {dias_inactivo} días.")


class AlertasPorVariacionPreciosProveedores(Alertas):
    '''Alerta para cuando existe variacion en los precios de los proveedores'''


class AlertaPorSeguimientoTendencias(Alertas):
    '''Alerta para saber cuando un modelo entra en una tendencias'''
    # Esta alerta se aplica a los datos agrupados por semanas  aplicada multiples meses

    def __init__(self, threshold=0.1,item:str = 'modulo' ,**cols):
        # threshold es el umbral que determina cuánto debe cambiar el coeficiente de variación
        # para que se dispare la alerta.
        self.threshold = threshold
        self.previous_coeficiente_varianza = None
        self.date = cols['0']
        self.model = cols['1']
        self.quantity = cols['2']
        self.customer = cols['3']

    def eval(self, data):
        # Si es la primera vez que evaluamos, simplemente guardamos el coeficiente de variación actual
        varianza = data['Varianza'].values[0]
        desviacion_estandar = data['Desviacion_estandar'].values[0]
        coeficiente_varianza = data['Coeficiente_varianza'].values[0]

        print('Varianza',varianza)
        print('Desviacion estandar',desviacion_estandar)
        print('coeficiente_varianza',coeficiente_varianza)
        
        if self.previous_coeficiente_varianza is None:
            self.previous_coeficiente_varianza = coeficiente_varianza
            # return
        print('self.previous_coeficiente_varianza',self.previous_coeficiente_varianza)
        # Calculamos el cambio relativo en el coeficiente de variación
        cambio_relativo = abs(
            coeficiente_varianza - self.previous_coeficiente_varianza) / self.previous_coeficiente_varianza

        print(cambio_relativo)
        # Si el cambio relativo supera el umbral, se dispara la alerta
        if cambio_relativo > self.threshold:
            print("Alerta: Cambio significativo en la tendencia de ventas detectado.")
            print("Varianza:", varianza)
            print("Desviación estándar:", desviacion_estandar)
            print("Coeficiente de variación anterior:",
                  self.previous_coeficiente_varianza)
            print("Coeficiente de variación actual:", coeficiente_varianza)
        else:
            print('No hay cambios')

        # Actualizamos el coeficiente de variación anterior para la próxima evaluación
        self.previous_coeficiente_varianza = coeficiente_varianza


class AlertaPorDemandaEstacional(Alertas):
    '''Alerta para saber cuando una producot esta en demanda estacional'''
    # Esta alerta se aplica a los datos agrupados por semanas aplicada multiples meses

    def __init__(self, item_to_check, threshold,item:str = 'modulo' ,**cols) -> None:
        super().__init__()
        self.threshold = threshold
        self.item_to_check = item_to_check
        self.date = cols['0']
        self.model = cols['1']
        self.quantity = cols['2']
        self.customer = cols['3']

    def eval(self, data):
        # Realiza la descomposición estacional
        # puedes ajustar el período según tus necesidades
        self.result = seasonal_decompose(
            data, model='additive', period=12)

        # Supongamos que quieres disparar una alerta si la amplitud de la componente estacional supera un umbral
        seasonal_amplitude = self.result.seasonal.max() - self.result.seasonal.min()
        if seasonal_amplitude > self.threshold:  # ajusta este umbral según tus necesidades
            print(
                f"Alerta: Demanda estacional detectada para el artículo {self.item_to_check}. Amplitud: {seasonal_amplitude}")
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
        self.alertas = [] # Lista de objetos Alertas
        # self.observers = [] # Lista de observadores

    def agregar_alerta(self, alerta):
        '''Agrega una alerta a la lista de alertas para evaluar.'''
        self.alertas.append(alerta)

    def evaluar_inventario(self):
        '''Evalúa todos los elementos en el inventario con todas las alertas.'''
        for item in self.inventario:
            for alerta in self.alertas:
                print(alerta)
                if alerta.eval(item): # Aquí puedes pasar los datos necesarios dependiendo de cómo estén estructuradas tus alertas
                    # self.notify(f"Alerta: {alerta.message} para el item {item}")
                    self.notify()

    def evaluar_historico(self,init_data,end_date):
        '''Evalúa todos los elementos en el inventario con todas las alertas.'''
        self.inventario = self.inventario[init_data:end_date]
        for alerta in self.alertas:
            # Aquí puedes pasar los datos necesarios dependiendo de cómo estén estructuradas tus alertas
            if isinstance(alerta,AlertaPorBajaCantidad):
                if alerta.eval(self.inventario[end_date:]): 
                    self.notify()
            if isinstance(alerta,AlertaPorInventarioInactivo):
                if alerta.eval(self.inventario[end_date:]): 
                    self.notify()

    def evaluar_metricas(self):
        '''Evaluar las alertas '''
        # for item in self.inventario:
        for alerta in self.alertas:
            if alerta.eval(self.inventario):
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
