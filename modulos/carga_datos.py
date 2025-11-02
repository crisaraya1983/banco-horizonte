"""
Módulo de Carga de Datos
========================
Este módulo centraliza toda la lógica de carga, validación y transformación 
de los datos CSV del proyecto. Al crear funciones reutilizables aquí, 
garantizamos que todos los análisis trabajan con datos consistentes.

La idea es que cualquier otro módulo puede importar las funciones de aquí
sin preocuparse por cómo se cargan los datos realmente.
"""

import pandas as pd
import streamlit as st
from pathlib import Path
import os

# Definimos la ruta donde están los datos
# Esto es flexible: funciona tanto en desarrollo como en Streamlit Cloud
RUTA_DATOS = Path(os.getcwd()) / "data"


@st.cache_data
def cargar_sucursales():
    """
    Carga el archivo de sucursales y lo transforma a un formato útil.
    
    Los datos contienen la ubicación física de cada sucursal, su tipo,
    volumen de transacciones y cantidad de empleados. Esta información
    es crítica para entender la cobertura bancaria.
    
    Returns:
        pd.DataFrame: Dataframe con columnas:
            - Ubicación (tupla de latitud, longitud)
            - Tipo de Sucursal
            - Volumen de Transacciones (mes)
            - Número de Empleados
    """
    df = pd.read_csv(
        RUTA_DATOS / "sucursales.csv", 
        sep=";"
    )
    
    # Parseamos la ubicación de string a tupla de coordenadas
    # Por ejemplo: "(-34.6118, -58.4173)" se convierte en (-34.6118, -58.4173)
    df[["Latitud", "Longitud"]] = df["Ubicación"].str.strip("()").str.split(", ", expand=True)
    df["Latitud"] = pd.to_numeric(df["Latitud"])
    df["Longitud"] = pd.to_numeric(df["Longitud"])
    
    return df


@st.cache_data
def cargar_cajeros():
    """
    Carga el archivo de cajeros automáticos.
    
    Los cajeros son puntos de contacto clave para los clientes.
    Necesitamos entender su distribución geográfica y su carga de trabajo
    para optimizar la red de sucursales.
    
    Returns:
        pd.DataFrame: Dataframe con columnas:
            - Ubicación (tupla de latitud, longitud)
            - Volumen de Transacciones Diarias
            - Tipo de Transacciones
    """
    df = pd.read_csv(
        RUTA_DATOS / "cajeros.csv",
        sep=";"
    )
    
    # Mismo parseado de coordenadas
    df[["Latitud", "Longitud"]] = df["Ubicación"].str.strip("()").str.split(", ", expand=True)
    df["Latitud"] = pd.to_numeric(df["Latitud"])
    df["Longitud"] = pd.to_numeric(df["Longitud"])
    
    return df


@st.cache_data
def cargar_clientes():
    """
    Carga el archivo de clientes.
    
    Los clientes son el centro del análisis. Sus ubicaciones, frecuencia de visitas
    y productos adquiridos nos ayudarán a identificar patrones de comportamiento
    y oportunidades de expansión.
    
    Returns:
        pd.DataFrame: Dataframe con columnas:
            - Ubicación de Residencia (tupla de latitud, longitud)
            - Frecuencia de Visitas
            - Productos Financieros Adquiridos
            - Volumen de Transacciones
            - Saldo Promedio de Cuentas
    """
    df = pd.read_csv(
        RUTA_DATOS / "clientes.csv",
        sep=";"
    )
    
    # Parseamos coordenadas de clientes
    df[["Latitud", "Longitud"]] = df["Ubicación de Residencia"].str.strip("()").str.split(", ", expand=True)
    df["Latitud"] = pd.to_numeric(df["Latitud"])
    df["Longitud"] = pd.to_numeric(df["Longitud"])
    
    return df


@st.cache_data
def cargar_productos():
    """
    Carga el archivo de productos financieros.
    
    Este archivo relaciona productos con sucursales, mostrando qué productos
    se ofrecen dónde, cuántos clientes los adquirieron y el volumen de ventas.
    
    Returns:
        pd.DataFrame: Dataframe con columnas:
            - Tipo de Producto
            - Sucursal Donde Se Ofrece
            - Número de Clientes
            - Volumen de Ventas
    """
    df = pd.read_csv(
        RUTA_DATOS / "productos.csv",
        sep=";"
    )
    
    return df


@st.cache_data
def cargar_todos_los_datos():
    """
    Carga todos los datasets en una sola llamada.
    
    Esta función es útil para las páginas que necesitan acceso a múltiples
    datasets de una vez. Al usar @st.cache_data, Streamlit cachea el resultado
    y no vuelve a ejecutar esta función a menos que el código cambie.
    
    Returns:
        tuple: (df_sucursales, df_cajeros, df_clientes, df_productos)
    """
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    productos = cargar_productos()
    
    return sucursales, cajeros, clientes, productos


def validar_datos():
    """
    Realiza validaciones básicas de los datos cargados.
    
    Esto es importante para debugging: nos ayuda a identificar si hay
    problemas en los datos desde el principio.
    
    Returns:
        dict: Diccionario con resultados de validación
    """
    validaciones = {}
    
    try:
        sucursales = cargar_sucursales()
        validaciones["sucursales"] = {
            "estado": "OK",
            "registros": len(sucursales),
            "columnas": list(sucursales.columns)
        }
    except Exception as e:
        validaciones["sucursales"] = {"estado": "ERROR", "mensaje": str(e)}
    
    try:
        cajeros = cargar_cajeros()
        validaciones["cajeros"] = {
            "estado": "OK",
            "registros": len(cajeros),
            "columnas": list(cajeros.columns)
        }
    except Exception as e:
        validaciones["cajeros"] = {"estado": "ERROR", "mensaje": str(e)}
    
    try:
        clientes = cargar_clientes()
        validaciones["clientes"] = {
            "estado": "OK",
            "registros": len(clientes),
            "columnas": list(clientes.columns)
        }
    except Exception as e:
        validaciones["clientes"] = {"estado": "ERROR", "mensaje": str(e)}
    
    try:
        productos = cargar_productos()
        validaciones["productos"] = {
            "estado": "OK",
            "registros": len(productos),
            "columnas": list(productos.columns)
        }
    except Exception as e:
        validaciones["productos"] = {"estado": "ERROR", "mensaje": str(e)}
    
    return validaciones