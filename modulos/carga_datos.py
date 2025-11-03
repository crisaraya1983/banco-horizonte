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
    
    # Transformación 1: Parsear la ubicación de string a coordenadas numéricas
    df[["Latitud", "Longitud"]] = df["Ubicación"].str.strip("()").str.split(", ", expand=True)
    df["Latitud"] = pd.to_numeric(df["Latitud"])
    df["Longitud"] = pd.to_numeric(df["Longitud"])
    
    df["Nombre"] = (
        df["Tipo de Sucursal"] + " " +
        df.groupby("Tipo de Sucursal").cumcount().add(1).astype(str)
    )
    
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

@st.cache_data
def obtener_productos_consolidados():

    df_productos = cargar_productos()
    
    df_consolidado = df_productos.groupby('Tipo de Producto').agg({
        'Número de Clientes': 'sum',
        'Volumen de Ventas': 'sum'
    }).reset_index()
    
    df_consolidado.columns = ['Tipo de Producto', 'Total Clientes', 'Total Volumen Ventas']
    
    return df_consolidado


@st.cache_data
def obtener_ubicaciones_con_productos():
  
    df_sucursales = cargar_sucursales()
    df_clientes = cargar_clientes()
    df_productos = cargar_productos()
    
    # Merge 1: Sucursales con Clientes por ubicación
    df_merge1 = df_sucursales.merge(
        df_clientes,
        left_on='Ubicación',
        right_on='Ubicación de Residencia',
        how='inner',
        suffixes=('_sucursal', '_cliente')
    )
    
    # Merge 2: Agregar productos usando Tipo de Sucursal + Producto Financiero
    df_merge2 = df_merge1.merge(
        df_productos,
        left_on=['Tipo de Sucursal', 'Productos Financieros Adquiridos'],
        right_on=['Sucursal Donde Se Ofrece', 'Tipo de Producto'],
        how='left'
    )
    
    # Seleccionar y renombrar columnas finales
    columnas_resultado = [
        'Ubicación',
        'Nombre',
        'Tipo de Sucursal',
        'Latitud',
        'Longitud',
        'Productos Financieros Adquiridos',
        'Número de Clientes',
        'Volumen de Ventas',
        'Volumen de Transacciones',
        'Saldo Promedio de Cuentas'
    ]
    
    df_resultado = df_merge2[columnas_resultado].copy()
    
    df_resultado.columns = [
        'Ubicación',
        'Sucursal',
        'Tipo de Sucursal',
        'Latitud',
        'Longitud',
        'Tipo de Producto',
        'Clientes del Producto',
        'Volumen de Ventas',
        'Volumen de Transacciones Zona',
        'Saldo Promedio Zona'
    ]
    
    return df_resultado


@st.cache_data
def obtener_resumen_por_sucursal():

    df_ubicaciones = obtener_ubicaciones_con_productos()
    df_sucursales = cargar_sucursales()
    
    # Agregar por sucursal
    df_resumen = df_ubicaciones.groupby(['Ubicación', 'Sucursal', 'Tipo de Sucursal']).agg({
        'Tipo de Producto': lambda x: ', '.join(x.unique()),
        'Clientes del Producto': 'sum',
        'Volumen de Ventas': 'sum',
        'Volumen de Transacciones Zona': 'first',
        'Saldo Promedio Zona': 'first'
    }).reset_index()
    
    # Agregar información de sucursal (empleados, transacciones)
    df_resumen = df_resumen.merge(
        df_sucursales[['Ubicación', 'Número de Empleados', 'Volumen de Transacciones (mes)']],
        on='Ubicación',
        how='left'
    )
    
    df_resumen.columns = [
        'Ubicación',
        'Sucursal',
        'Tipo de Sucursal',
        'Productos Ofrecidos',
        'Total Clientes',
        'Total Volumen Ventas',
        'Volumen Transacciones Zona',
        'Saldo Promedio Zona',
        'Empleados',
        'Volumen Transacciones Sucursal Mensual'
    ]
    
    # Reordenar columnas para mejor presentación
    orden_columnas = [
        'Ubicación',
        'Sucursal',
        'Tipo de Sucursal',
        'Empleados',
        'Volumen Transacciones Sucursal Mensual',
        'Productos Ofrecidos',
        'Total Clientes',
        'Total Volumen Ventas',
        'Volumen Transacciones Zona',
        'Saldo Promedio Zona'
    ]
    
    df_resumen = df_resumen[orden_columnas]
    
    return df_resumen

"""
FUNCIÓN PARA AGREGAR A modulos/carga_datos.py

Copia esta función completa al final de tu archivo carga_datos.py
"""

@st.cache_data
def obtener_datos_consolidados():

    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    productos = cargar_productos()
    
    consolidado = sucursales.merge(
        cajeros[['Ubicación', 'Volumen de Transacciones Diarias', 'Tipo de Transacciones']],
        on='Ubicación',
        how='left'
    )
    
    consolidado = consolidado.merge(
        clientes[['Ubicación de Residencia', 'Frecuencia de Visitas', 
                  'Productos Financieros Adquiridos', 'Volumen de Transacciones', 
                  'Saldo Promedio de Cuentas']],
        left_on='Ubicación',
        right_on='Ubicación de Residencia',
        how='left'
    )
    
    productos_preparado = productos.rename(columns={
        'Sucursal Donde Se Ofrece': 'Sucursal_Merge',
        'Tipo de Producto': 'Producto_Merge'
    })
    
    consolidado['Sucursal_Merge'] = consolidado['Tipo de Sucursal']
    consolidado['Producto_Merge'] = consolidado['Productos Financieros Adquiridos']
    
    consolidado = consolidado.merge(
        productos_preparado[['Sucursal_Merge', 'Producto_Merge', 
                            'Número de Clientes', 'Volumen de Ventas']],
        on=['Sucursal_Merge', 'Producto_Merge'],
        how='left'
    )
    
    consolidado.drop(columns=['Sucursal_Merge', 'Producto_Merge', 'Ubicación de Residencia'], 
                     inplace=True)
    
    consolidado.rename(columns={
        'Volumen de Transacciones (mes)': 'Volumen_Transacciones_Sucursal',
        'Volumen de Transacciones Diarias': 'Volumen_Transacciones_Cajero_Diarias',
        'Tipo de Transacciones': 'Tipos_Transacciones_Cajero',
        'Volumen de Transacciones': 'Volumen_Transacciones_Cliente',
        'Número de Clientes': 'Numero_Clientes_Producto',
        'Volumen de Ventas': 'Volumen_Ventas_Producto'
    }, inplace=True)
    
    consolidado = consolidado[[
        'Ubicación',
        'Nombre',
        'Tipo de Sucursal',
        'Número de Empleados',
        'Volumen_Transacciones_Sucursal',
        'Volumen_Transacciones_Cajero_Diarias',
        'Tipos_Transacciones_Cajero',
        'Frecuencia de Visitas',
        'Productos Financieros Adquiridos',
        'Volumen_Transacciones_Cliente',
        'Saldo Promedio de Cuentas',
        'Numero_Clientes_Producto',
        'Volumen_Ventas_Producto',
        'Latitud',
        'Longitud'
    ]]
    
    return consolidado