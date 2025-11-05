## Requisitos Previos

Tener instalados:

- Git
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Instrucciones de Instalación

### 1. Clonar el Repositorio

Abrir una terminal y ejecuta:

```bash
git clone https://github.com/crisaraya1983/banco-horizonte.git
cd banco-horizonte
```

### 2. Crear un Entorno Virtual

En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

En macOS y Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las Dependencias

```bash
pip install -r requirements.txt
```

Este comando instalará todas las librerías especificadas en el archivo requirements.txt:
- streamlit
- pandas
- numpy
- folium
- streamlit-folium
- plotly
- scikit-learn
- geopy
- scipy
- openpyxl

## Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador web:
```
http://localhost:8501
```

Si la aplicación no se abre automáticamente, accede manualmente a la URL anterior desde tu navegador.

## Solución de Problemas

### Error: "streamlit command not found"
Asegúrate de que el entorno virtual está activado. En Windows ejecuta `venv\Scripts\activate` y en macOS/Linux ejecuta `source venv/bin/activate`.

### Error: "ModuleNotFoundError"
Reinstala las dependencias con: `pip install -r requirements.txt`

### Puerto 8501 en uso
Si el puerto 8501 está en uso, puedes especificar otro puerto:
```bash
streamlit run app.py --server.port 8502
```
