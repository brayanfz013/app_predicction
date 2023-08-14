''' 
Codigo para generar un sistema de alertas usando patrones de diseno, se aplica tanto a los datos
de inventario real como a los datos de las predicciones de los modelos

Se usan el patron de diseno observer patter junto strategy patter para generar un sistema de alertas
que pueda se modificable en el tiempo
'''

from abc import ABC, abstractmethod


class Subject:
    ''' 
    The `Subject` class is a base class for implementing the Observer pattern,
    allowing objects to attach, detach, and notify their observers.
    '''

    def __init__(self, data_source) -> None:
        self._observers = data_source

    def attach(self, observer):
        '''Agregar nuevos objetos para ser observado'''
        self._observers.add_observer(observer)

    def detach(self, observer):
        '''Remove observadores de los motodos'''
        self._observers.remove_observer(observer)

    def notify(self):
        '''Notificacion de observadores '''
        for observer in self._observers.get_observers():
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
    def eval(self,data):
        if data['cantidad'] <50:
            print("Alerta: Baja cantidad de:",data['producto'])

# Alerta por Baja Cantidad: No hay una métrica directamente relacionada 
# con esta alerta en la lista proporcionada.

# Alerta por Tiempo de Venta Bajo: Puedes usar las columnas init_date y end_date
#  para determinar el período de tiempo en que un producto ha estado en el inventario.

# Alerta de Expiración de Producto: No hay una métrica directamente relacionada
# con esta alerta en la lista proporcionada, ya que necesitarías información sobre 
# la fecha de caducidad del producto.

# Alerta por Cambios Bruscos en las Ventas: Las columnas varianza y desviacion_estandar 
# pueden ayudar a detectar fluctuaciones en las ventas. Una alta varianza o desviación 
# estándar podría indicar cambios bruscos.

# Alerta por Inventario Inactivo: Similar a la Alerta por Tiempo de Venta Bajo, 
# puedes utilizar init_date y end_date para identificar productos que no han tenido 
# movimiento en un período determinado.

# Alerta por Variación de Precios de Proveedores: No hay una métrica directamente 
# relacionada con esta alerta en la lista proporcionada, ya que necesitarías información 
# sobre los precios de compra y venta.

# Alerta de Seguimiento de Tendencias de Ventas: La varianza, desviacion_estandar,
#  y coeficiente_varianza podrían usarse para identificar tendencias en las ventas
#  de diferentes productos.

# Alerta por Demanda Estacional: No hay una métrica directamente relacionada con
#  esta alerta en la lista proporcionada. La detección de demanda estacional 
# generalmente requeriría análisis de datos de ventas a lo largo del tiempo, 
# posiblemente en una escala mensual o trimestral.

# Alerta de Seguridad de Inventario: No hay una métrica directamente relacionada 
# con esta alerta en la lista proporcionada. La seguridad del inventario normalmente
#  requiere un seguimiento detallado de los movimientos y ajustes del inventario.

# Alerta de Evaluación de Proveedores: No hay una métrica directamente relacionada 
# con esta alerta en la lista proporcionada, ya que necesitarías información específica 
# sobre los proveedores, como tiempos de entrega, calidad, etc.


# ===================================================================
#          Clase concreta de Observer para utilizar la alerta
# ===================================================================


class AlertaObserver(Observer):
    def __init__(self, alerta:Alertas) -> None:
        self.alerta = alerta

    def update(self, subject):
        self.alerta.eval(subject.data)


class Invertario(Subject):
    def __init__(self, data_source,modelo, cantidad) -> None:
        super().__init__(data_source)
        self.data = {'modelo':modelo, 'cantidad':cantidad}

    def vender(self,cantidad):
        self.data['cantidad'] -= cantidad
        self.notify()


if __name__ == "__main__":
    alerta_concreta = AlertaPorBajaCantidad()
    observer_concreto = AlertaObserver(alerta_concreta)

    DATA_SOURCE_TEST = None
    inventario = Invertario(data_source=DATA_SOURCE_TEST,modelo = 'ModeloA',cantidad = 100)
    inventario.attach(observer_concreto)

    inventario.vender(60)
    inventario.vender(20)