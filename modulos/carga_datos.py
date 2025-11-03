import pandas as pd
import streamlit as st
from pathlib import Path
import os

# Ruta donde están los datos
RUTA_DATOS = Path(os.getcwd()) / "data"


@st.cache_data
def cargar_sucursales():
    df = pd.read_csv(
        RUTA_DATOS / "sucursales.csv", 
        sep=";"
    )
    
    # Parseamos la ubicación de string a tupla de coordenadas
    df[["Latitud", "Longitud"]] = df["Ubicación"].str.strip("()").str.split(", ", expand=True)
    df["Latitud"] = pd.to_numeric(df["Latitud"])
    df["Longitud"] = pd.to_numeric(df["Longitud"])
    
    return df


@st.cache_data
def cargar_cajeros():
    df = pd.read_csv(
        RUTA_DATOS / "cajeros.csv",
        sep=";"
    )
    df[["Latitud", "Longitud"]] = df["Ubicación"].str.strip("()").str.split(", ", expand=True)
    df["Latitud"] = pd.to_numeric(df["Latitud"])
    df["Longitud"] = pd.to_numeric(df["Longitud"])
    
    return df


@st.cache_data
def cargar_clientes():
    df = pd.read_csv(
        RUTA_DATOS / "clientes.csv",
        sep=";"
    )
    df[["Latitud", "Longitud"]] = df["Ubicación de Residencia"].str.strip("()").str.split(", ", expand=True)
    df["Latitud"] = pd.to_numeric(df["Latitud"])
    df["Longitud"] = pd.to_numeric(df["Longitud"])
    
    return df


@st.cache_data
def cargar_productos():
    df = pd.read_csv(
        RUTA_DATOS / "productos.csv",
        sep=";"
    )
    
    return df


@st.cache_data
def cargar_todos_los_datos():
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    productos = cargar_productos()
    
    return sucursales, cajeros, clientes, productos
