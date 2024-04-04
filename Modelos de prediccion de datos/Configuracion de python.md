---
link: https://www.notion.so/Configuracion-de-python-86598e2eaf2b44108331a3f9cd1ec412
notionID: 86598e2e-af2b-4410-8331-a3f9cd1ec412
---
# Configuracion de python

Se esta usando la version de pyhon 3.10.0 para la ejecucion y codigo del proyecto 

### Instalacion de python

¡Por supuesto! Aquí están los pasos para instalar Python 3.10 en un sistema Linux/Unix desde la terminal:

1. **Actualizar el sistema**:
Antes de instalar cualquier paquete, es una buena práctica actualizar el sistema y los repositorios de paquetes.
    
```bash
	sudo apt update
	sudo apt upgrade
```
    
2. **Instalar las herramientas necesarias**:
Es necesario instalar algunas herramientas requeridas para la construcción y compilación del código fuente.
    
```bash
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl
```
    
3. **Descargar Python 3.10**:
	Puedes obtener la última versión de Python desde el sitio oficial. En este caso, vamos a descargar Python 3.10.

``` bash
	wget <https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz>
```

Descomprime el archivo descargado:

``` bash
tar -xvf Python-3.10.0.tgz
```


4. **Compilar e instalar**:
    Navega al directorio que contiene el código fuente de Python:
    
    ```bash
    cd Python-3.10.0
    ```
    
    Configura el proceso de instalación:

	```bash
	./configure --enable-optimizations
	```

	Inicia el proceso de instalación:

	
	```bash
	sudo make altinstall
	```
    
	Usamos `make altinstall` en lugar de `make install` para evitar reemplazar el Python predeterminado del sistema. Esto instalará Python 3.10 como `python3.10`.
    
5. **Verifica la instalación**:
    
    Una vez finalizada la instalación, puedes verificar que Python 3.10 se haya instalado correctamente:

```bash
python3.10 --version
```
    

Esto debería mostrar `Python 3.10.0` o la versión específica que hayas instalado.

Nota: El proceso de instalación puede variar dependiendo de la distribución específica de Linux/Unix que estés utilizando. Estos pasos están basados en sistemas Debian/Ubuntu. Si estás usando otra distribución (como CentOS, Fedora, etc.), es posible que debas adaptar algunos comandos.

**Nota :** 

Se tiene que tener en consideracion que necesario primero instalar CUDA en el sistema operativo para sacar el maximo provecho a algunas librerias como lo es torch

Los librerias para ejecutar el codigo son las siguientes:

```bash
pip install numpy 
pip install pandas
pip install pyodbc
pip install redis
pip install psycopg2-binary
pip install darts
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install optuna
pip install tqdm
pip install scipy
```