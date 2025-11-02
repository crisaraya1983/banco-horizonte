"""
Módulo de Visualizaciones Geoespaciales
========================================
Este módulo encapsula toda la lógica de creación de visualizaciones.
Separa la "lógica de presentación" de la "lógica de datos".

La filosofía aquí es que este módulo no se preocupa por cómo se calculan los datos,
solo por cómo presentarlos de forma hermosa e interactiva.

Usamos dos librerías principales:
- Folium: Para mapas interactivos basados en OpenStreetMap
- Plotly: Para gráficos interactivos que el usuario puede explorar
"""

import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


# ============================================================================
# FUNCIONES DE UTILIDAD PARA MAPAS
# ============================================================================

def crear_mapa_base(centro_lat, centro_lon, zoom=6, tiles="OpenStreetMap"):
    """
    Crea un mapa base vacío centrado en las coordenadas especificadas.
    
    Este es el punto de partida para cualquier mapa. Folium proporciona
    diferentes opciones de tiles (capas base) que varían en estilo:
    - OpenStreetMap: Clásico, monocromo
    - CartoDB positron: Limpio, minimalista
    - CartoDB voyager: Colorido, con más detalles
    
    Parámetros:
        centro_lat, centro_lon: Coordenadas del centro del mapa
        zoom: Nivel de zoom inicial (1-18, más alto = más zoom)
        tiles: Estilo de mapa a usar
    
    Returns:
        folium.Map: Objeto de mapa folium
    """
    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=zoom,
        tiles=tiles,
        prefer_canvas=True  # Mejora performance en navegadores
    )
    return mapa


def calcular_centroide(df):
    """
    Calcula el centro geográfico de un conjunto de puntos.
    
    Parámetros:
        df: DataFrame con columnas 'Latitud' y 'Longitud'
    
    Returns:
        tuple: (latitud_promedio, longitud_promedio)
    """
    lat_prom = df['Latitud'].mean()
    lon_prom = df['Longitud'].mean()
    return lat_prom, lon_prom


# ============================================================================
# VISUALIZACIÓN DE SUCURSALES Y CAJEROS
# ============================================================================

def crear_mapa_sucursales_cajeros(sucursales_df, cajeros_df, clientes_df=None):
    """
    Crea un mapa interactivo mostrando todas las sucursales y cajeros automáticos.
    
    Este es probablemente el mapa más importante para entender la cobertura actual
    de la red bancaria. Muestra:
    - Sucursales como marcadores azules grandes
    - Cajeros como marcadores verdes más pequeños
    - Opcionalmente, clientes como puntos rojos pequeños
    
    Los diferentes colores hacen que sea fácil distinguir qué es qué a simple vista.
    
    Parámetros:
        sucursales_df: DataFrame de sucursales
        cajeros_df: DataFrame de cajeros
        clientes_df: DataFrame de clientes (opcional)
    
    Returns:
        folium.Map: Mapa interactivo con todas las ubicaciones
    """
    # Calculamos el centroide de sucursales para centrar el mapa
    centro_lat, centro_lon = calcular_centroide(sucursales_df)
    
    # Creamos el mapa base
    mapa = crear_mapa_base(centro_lat, centro_lon, zoom=7)
    
    # Agregamos sucursales como marcadores azules
    for idx, row in sucursales_df.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"""
            <b>{row['Tipo de Sucursal']}</b><br>
            Transacciones/mes: {row['Volumen de Transacciones (mes)']}<br>
            Empleados: {row['Número de Empleados']}
            """,
            tooltip=f"Sucursal {row['Tipo de Sucursal']}",
            icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
        ).add_to(mapa)
    
    # Agregamos cajeros como marcadores verdes más pequeños
    for idx, row in cajeros_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=6,
            popup=f"""
            <b>Cajero Automático</b><br>
            Transacciones/día: {row['Volumen de Transacciones Diarias']}<br>
            Tipo: {row['Tipo de Transacciones']}
            """,
            tooltip="Cajero ATM",
            color='green',
            fill=True,
            fillColor='lightgreen',
            fillOpacity=0.7,
            weight=2
        ).add_to(mapa)
    
    # Opcionalmente, agregamos clientes como puntos pequeños
    if clientes_df is not None and len(clientes_df) > 0:
        for idx, row in clientes_df.iterrows():
            folium.CircleMarker(
                location=[row['Latitud'], row['Longitud']],
                radius=3,
                popup=f"""
                <b>Cliente</b><br>
                Producto: {row['Productos Financieros Adquiridos']}<br>
                Saldo promedio: ${row['Saldo Promedio de Cuentas']:,.0f}
                """,
                tooltip="Cliente",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.3,
                weight=1
            ).add_to(mapa)
    
    # Agregamos una leyenda
    leyenda = folium.Figure().add_child(folium.LatLngPopup())
    
    return mapa


def crear_mapa_cobertura_clientes(clientes_df, sucursales_df, cajeros_df, 
                                   umbral_sucursal=10.0, umbral_cajero=5.0):
    """
    Crea un mapa que muestra qué clientes están cubiertos y cuáles no.
    
    Este mapa es muy útil para identificar visualmente las áreas desatendidas.
    Los clientes se colorean según si están dentro de los umbrales de distancia:
    - Verde: Cliente bien cubierto (dentro de ambos umbrales)
    - Amarillo: Cliente parcialmente cubierto
    - Rojo: Cliente desatendido (fuera de ambos umbrales)
    
    Parámetros:
        clientes_df: DataFrame de clientes (debe tener distancias calculadas)
        sucursales_df: DataFrame de sucursales
        cajeros_df: DataFrame de cajeros
        umbral_sucursal: Distancia máxima a sucursal (km)
        umbral_cajero: Distancia máxima a cajero (km)
    
    Returns:
        folium.Map: Mapa con clientes coloreados por cobertura
    """
    from modulos.geoespacial import (
        calcular_distancia_a_sucursal_mas_cercana,
        calcular_distancia_a_cajero_mas_cercano
    )
    
    # Calculamos las distancias para cada cliente
    clientes = clientes_df.copy()
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales_df)
    clientes = calcular_distancia_a_cajero_mas_cercano(clientes, cajeros_df)
    
    # Determinamos el estado de cobertura para cada cliente
    def determinar_color_cobertura(row):
        """Determina el color basado en la cobertura del cliente."""
        distancia_sucursal = row['Distancia_a_Sucursal_km'] <= umbral_sucursal
        distancia_cajero = row['Distancia_a_Cajero_km'] <= umbral_cajero
        
        if distancia_sucursal and distancia_cajero:
            return 'green'  # Bien cubierto
        elif distancia_sucursal or distancia_cajero:
            return 'orange'  # Parcialmente cubierto
        else:
            return 'red'  # Desatendido
    
    clientes['Color_Cobertura'] = clientes.apply(determinar_color_cobertura, axis=1)
    
    # Centrado del mapa
    centro_lat, centro_lon = calcular_centroide(clientes)
    mapa = crear_mapa_base(centro_lat, centro_lon, zoom=7)
    
    # Agregamos clientes con colores según cobertura
    for idx, row in clientes.iterrows():
        color = row['Color_Cobertura']
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=5,
            popup=f"""
            <b>Cliente</b><br>
            Producto: {row['Productos Financieros Adquiridos']}<br>
            Dist. a sucursal: {row['Distancia_a_Sucursal_km']:.2f} km<br>
            Dist. a cajero: {row['Distancia_a_Cajero_km']:.2f} km<br>
            Saldo: ${row['Saldo Promedio de Cuentas']:,.0f}
            """,
            tooltip="Click para más detalles",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(mapa)
    
    # Agregamos sucursales y cajeros de referencia
    for idx, row in sucursales_df.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"Sucursal: {row['Tipo de Sucursal']}",
            icon=folium.Icon(color='blue', icon='bank', prefix='fa'),
            opacity=0.7
        ).add_to(mapa)
    
    for idx, row in cajeros_df.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"Cajero ATM",
            icon=folium.Icon(color='gray', icon='money', prefix='fa'),
            opacity=0.7
        ).add_to(mapa)
    
    # Leyenda HTML
    leyenda_html = """
    <div style="position: fixed; 
            bottom: 50px; right: 50px; width: 200px; height: 150px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
    <p style="margin:0;"><b>Leyenda de Cobertura</b></p>
    <p style="margin:5px;"><i class="fa fa-circle" style="color:green"></i> Bien cubierto</p>
    <p style="margin:5px;"><i class="fa fa-circle" style="color:orange"></i> Parcialmente cubierto</p>
    <p style="margin:5px;"><i class="fa fa-circle" style="color:red"></i> Desatendido</p>
    <p style="margin:5px;"><i class="fa fa-bank"></i> Sucursal</p>
    <p style="margin:5px;"><i class="fa fa-money"></i> Cajero ATM</p>
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(leyenda_html))
    
    return mapa


# ============================================================================
# GRÁFICOS CON PLOTLY
# ============================================================================

def crear_grafico_volumen_transacciones(sucursales_df):
    """
    Crea un gráfico de barras mostrando el volumen de transacciones por sucursal.
    
    Este gráfico es útil para identificar rápidamente qué sucursales
    son más activas y tienen mayor demanda.
    
    Parámetros:
        sucursales_df: DataFrame de sucursales
    
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    # Preparamos los datos
    datos = sucursales_df.groupby('Tipo de Sucursal').agg({
        'Volumen de Transacciones (mes)': 'sum',
        'Número de Empleados': 'sum'
    }).reset_index()
    
    # Creamos el gráfico
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=datos['Tipo de Sucursal'],
        y=datos['Volumen de Transacciones (mes)'],
        name='Transacciones/mes',
        marker_color='lightblue',
        text=datos['Volumen de Transacciones (mes)'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Volumen de Transacciones por Tipo de Sucursal",
        xaxis_title="Tipo de Sucursal",
        yaxis_title="Transacciones/mes",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def crear_grafico_empleados_vs_transacciones(sucursales_df):
    """
    Crea un gráfico de dispersión mostrando la relación entre empleados
    y volumen de transacciones.
    
    Este gráfico ayuda a identificar si hay una relación entre cantidad
    de empleados y productividad (transacciones).
    
    Parámetros:
        sucursales_df: DataFrame de sucursales
    
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    fig = px.scatter(
        sucursales_df,
        x='Número de Empleados',
        y='Volumen de Transacciones (mes)',
        color='Tipo de Sucursal',
        size='Volumen de Transacciones (mes)',
        hover_name='Tipo de Sucursal',
        title='Relación: Empleados vs Volumen de Transacciones',
        labels={
            'Número de Empleados': 'Empleados',
            'Volumen de Transacciones (mes)': 'Transacciones/mes'
        },
        template='plotly_white'
    )
    
    fig.update_layout(hovermode='closest')
    return fig


def crear_grafico_productos_por_ubicacion(clientes_df):
    """
    Crea un gráfico de barras apiladas mostrando qué productos se adquieren
    en cada ubicación (basado en ubicación de residencia).
    
    Este gráfico es esencial para el marketing dirigido, mostrando
    patrones de preferencia de productos por región.
    
    Parámetros:
        clientes_df: DataFrame de clientes
    
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    # Agrupamos por ubicación y producto
    datos = clientes_df.groupby(['Ubicación de Residencia', 'Productos Financieros Adquiridos']).size().reset_index(name='Cantidad')
    
    # Convertimos a un formato más legible para el gráfico
    datos['Ubicación'] = datos['Ubicación de Residencia'].astype(str)
    
    fig = px.bar(
        datos,
        x='Ubicación',
        y='Cantidad',
        color='Productos Financieros Adquiridos',
        title='Distribución de Productos Financieros por Ubicación',
        labels={'Cantidad': 'Número de Clientes'},
        template='plotly_white',
        barmode='stack'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode='x unified'
    )
    
    return fig


def crear_grafico_saldo_promedio_por_producto(clientes_df):
    """
    Crea un gráfico mostrando el saldo promedio de cuentas según el producto adquirido.
    
    Este gráfico ayuda a identificar qué productos están asociados con
    clientes más valiosos (con mayor saldo).
    
    Parámetros:
        clientes_df: DataFrame de clientes
    
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    datos = clientes_df.groupby('Productos Financieros Adquiridos').agg({
        'Saldo Promedio de Cuentas': 'mean',
        'Volumen de Transacciones': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=datos['Productos Financieros Adquiridos'],
        y=datos['Saldo Promedio de Cuentas'],
        name='Saldo Promedio',
        marker_color='lightgreen',
        text=[f'${x:,.0f}' for x in datos['Saldo Promedio de Cuentas']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Saldo Promedio de Cuentas por Producto Financiero",
        xaxis_title="Producto Financiero",
        yaxis_title="Saldo Promedio ($)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def crear_grafico_frecuencia_visitas(clientes_df):
    """
    Crea un gráfico mostrando la distribución de frecuencia de visitas
    de los clientes a las sucursales.
    
    Parámetros:
        clientes_df: DataFrame de clientes
    
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    datos = clientes_df['Frecuencia de Visitas'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=datos.index,
            y=datos.values,
            marker_color='steelblue',
            text=datos.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Distribución de Frecuencia de Visitas de Clientes",
        xaxis_title="Visitas/mes",
        yaxis_title="Cantidad de Clientes",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def crear_grafico_transacciones_cajeros(cajeros_df):
    """
    Crea un gráfico mostrando el volumen de transacciones por cajero automático.
    
    Parámetros:
        cajeros_df: DataFrame de cajeros
    
    Returns:
        plotly.graph_objects.Figure: Gráfico interactivo
    """
    # Creamos identidades legibles para los cajeros
    cajeros_df_copy = cajeros_df.copy()
    cajeros_df_copy['Cajero_ID'] = [f"Cajero {i+1}" for i in range(len(cajeros_df_copy))]
    
    fig = px.bar(
        cajeros_df_copy,
        x='Cajero_ID',
        y='Volumen de Transacciones Diarias',
        color='Tipo de Transacciones',
        title='Volumen de Transacciones Diarias por Cajero Automático',
        labels={'Volumen de Transacciones Diarias': 'Transacciones/día'},
        template='plotly_white'
    )
    
    fig.update_layout(hovermode='x unified')
    
    return fig


def crear_grafico_matriz_distancias(matriz_distancias, etiquetas=None):
    """
    Crea un mapa de calor mostrando la matriz de distancias entre puntos.
    
    Esto es útil para optimización logística. Las distancias cortas (oscuras)
    indican que dos puntos están cerca, mientras que distancias largas (claras)
    indican que están lejos.
    
    Parámetros:
        matriz_distancias: Array NumPy 2D con distancias
        etiquetas: Lista de etiquetas para los ejes
    
    Returns:
        plotly.graph_objects.Figure: Heatmap interactivo
    """
    if etiquetas is None:
        etiquetas = [f"Punto {i+1}" for i in range(len(matriz_distancias))]
    
    fig = go.Figure(data=go.Heatmap(
        z=matriz_distancias,
        x=etiquetas,
        y=etiquetas,
        colorscale='Viridis',
        text=np.round(matriz_distancias, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Distancia (km)")
    ))
    
    fig.update_layout(
        title="Matriz de Distancias entre Puntos de Servicio",
        xaxis_title="Origen",
        yaxis_title="Destino",
        height=600,
        width=700
    )
    
    return fig