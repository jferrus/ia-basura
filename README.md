# ♻️ Simple Trashnet Garbage Classifier IA

Este proyecto es un sistema simple de clasificación automática de residuos basado en Inteligencia Artificial y Visión por Computador. Utiliza un modelo de Deep Learning entrenado con el dataset **Trashnet** para identificar materiales y sugerir el contenedor de reciclaje adecuado.

> [!IMPORTANT]
> **Aviso sobre residuos orgánicos:** Este modelo **no clasifica materia orgánica**. El dataset original de Trashnet no incluye dicha categoría, por lo que el sistema está enfocado exclusivamente en materiales reciclables y restos no orgánicos.

---

## 📋 Descripción del Modelo

El sistema analiza las imágenes y las clasifica en una de las 6 categorías del dataset original. A continuación, se muestra la correspondencia con los contenedores de reciclaje:

| Clase de Trashnet | Material | Contenedor Sugerido |
| :--- | :--- | :--- |
| `glass` | Vidrio | 🟢 **Verde** |
| `paper` | Papel | 🔵 **Azul** |
| `cardboard` | Cartón | 🔵 **Azul** |
| `plastic` | Plástico | 🟡 **Amarillo** |
| `metal` | Metal | 🟡 **Amarillo** |
| `trash` | Otros / Resto | ⚫ **Gris** |

---

## 📂 Estructura del Proyecto

* `ia_basura.py`: Script principal de Python para cargar el modelo y procesar imágenes.
* `ia_basura2.py`: Otro script principal de Python para cargar el modelo y procesar imágenes.
* `generar_modelo_basura.pth`: Script para entrenar el modelo con PyTorch.
* `generar_modelo_basura2.pth`: Script para entrenar otro modelo con PyTorch.
* `imagenes/`: Carpeta para guardar imágenes de prueba.
* `requirements.txt`: Lista de librerías necesarias para la ejecución.
* `README.md`: Documentación del proyecto.

---

## 🚀 Instalación y Uso

### 1. Requisitos Previos
Asegúrate de tener instalado Python 3.x. Es recomendable crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias
Instala las librerías necesarias mediante el archivo de requisitos:
```bash
pip install -r requirements.txt
```
### 3. Entrenar el modelo
El modelo 2 es más complejo, tarda más en generarse y es más exacto. Para entrenar el modelo:
```bash
python generar_modelo_basura.py
o
python generar_modelo_basura2.py

```
### 4. Realizar una Clasificación

Para ejecutar la IA sobre una imagen específica, utiliza el siguiente comando:
```bash
python ia_basura.py
o
python ia_basura2.py
```
Una vez se ejecuta puedes poner el nombre de una imagen del directorio /imagenes de este proyecto sin la extensión, que tiene que ser .jpg de forma obligatoria.

La primera vez que lo ejecutas tarda un rato en responder pero las siguientes veces ya va más rápdio.

### 🧠 Detalles Técnicos

- Dataset de origen: Gary Thung - TrashNet.

- Framework: PyTorch.

- Procesamiento: Las imágenes son reescaladas y normalizadas antes de entrar en la red neuronal para asegurar la precisión de la predicción.

- Limitaciones: Al no contar con datos de "Orgánico", cualquier residuo de este tipo será clasificado dentro de la categoría más similar visualmente (generalmente trash o paper), lo que puede dar lugar a falsos positivos.

Desarrollado para mejorar la eficiencia en la gestión de residuos.