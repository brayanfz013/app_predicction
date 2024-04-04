---
link: https://www.notion.so/Configuracion-de-CUDA-7bd95393614b4463b899e475398f89cc
notionID: 7bd95393-614b-4463-b899-e475398f89cc
---
# Configuracion de CUDA

1. **Asegúrate de tener los requisitos previos.**
    - Necesitarás una versión de Windows 10 o Windows 11 con las últimas actualizaciones instaladas.
    - Necesitarás una distribución de Linux compatible con WSL2, como Ubuntu o Debian.
    - Necesitarás una GPU NVIDIA compatible con CUDA.
    - Necesitarás descargar e instalar los controladores NVIDIA para tu GPU.
2. **Habilita WSL2.**
    - Abre la aplicación Configuración.
    - Haz clic en "Aplicaciones y características".
    - En la lista de aplicaciones, haz clic en "Windows Subsystem for Linux".
    - Haz clic en "Opciones avanzadas".
    - En la sección "Kernel", haz clic en el botón "Habilitar".
3. **Instala una distribución de Linux.**
    - Hay varias formas de instalar una distribución de Linux. Puedes instalarla desde la Tienda de Microsoft, o puedes descargarla e instalarla desde la web del desarrollador.
    - Si estás instalando desde la Tienda de Microsoft, abre la Tienda de Microsoft y busca "Ubuntu". Haz clic en el botón "Instalar" para instalar Ubuntu.
    - Si estás descargando e instalando desde la web del desarrollador, ve al sitio web del desarrollador y descarga la última versión de Ubuntu. Una vez descargada la imagen ISO, puedes usar una herramienta de creación de discos para grabarla en una unidad USB o DVD.
4. **Inicia la distribución de Linux.**
    - Una vez que hayas instalado una distribución de Linux, puedes iniciarla abriendo la aplicación Símbolo del sistema y ejecutando el siguiente comando:
        
        `wsl --distribution <distribution-name> start`
        
    
    Por ejemplo, para iniciar Ubuntu, ejecutarías el siguiente comando:
    `wsl --distribution ubuntu start`
    
5. **Instala los controladores NVIDIA.**
    - Una vez que hayas iniciado la distribución de Linux, puedes instalar los controladores NVIDIA ejecutando el siguiente comando:
        
        `sudo apt-get install nvidia-driver-470`
        
    
    El número de versión del controlador puede variar, así que asegúrate de instalar la versión más reciente.
    

Seleccionar en esta pagina : 

**Operative system**: linux

**Arquitecture**: x86_64

**Distribution**: WSL-ubuntu

**Version**: 2.0

**Installer** Type: deb (local)

[CUDA Toolkit 11.8 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

Seguir las instrucciones de instalacion:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```