---
link: https://www.notion.so/Configurar-base-de-datos-874a5fa047e04b6186e95a78f954e02e
notionID: 874a5fa0-47e0-4b61-86e9-5a78f954e02e
---
# Configurar base de datos

## Redis

Es necesario configurar redis para establecer la base de datos de cache para sobrecargar la base de datos

```bash
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

sudo apt-get update
sudo apt-get install redis
sudo service redis-server start
```

## Postgrest

En caso de que quiera es extraer o ingresar datos en postgres el codigo cuenta con la funcionalidad completa para ejecutar estas acciones

## SQLserver

En caso de que quiera es extraer o ingresar datos en postgres el codigo cuenta **NO** con la funcionalidad completa para ejecutar estas acciones

## Texto plano

En caso de que quiera es extraer o ingresar datos en postgres el codigo cuenta **NO** con la funcionalidad completa para ejecutar estas acciones