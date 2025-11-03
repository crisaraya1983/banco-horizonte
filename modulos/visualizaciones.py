import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


# CONFIGURACIÓN BASE DE PLOTLY

# Paleta de colores
COLORES = [
    "#2c5aa0",
    "#3498db",
    "#27ae60",
    "#f39c12",
    "#e74c3c",
    "#9b59b6",
    "#1abc9c",
    "#34495e"
]


def aplicar_tema(fig):
    """
    Aplica el tema visual a cualquier gráfico Plotly.
    """
    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2d3748"
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top"
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(
        gridcolor="#e2e8f0",
        showgrid=True,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#e2e8f0"
    )
    
    fig.update_yaxes(
        gridcolor="#e2e8f0",
        showgrid=True,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#e2e8f0"
    )
    
    return fig


# FUNCIONES DE MAPAS (FOLIUM)

def crear_mapa_base(centro_lat, centro_lon, zoom=6, tiles="OpenStreetMap"):
    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=zoom,
        tiles=tiles,
        prefer_canvas=True
    )
    return mapa


def calcular_centroide(df):
    lat_prom = df['Latitud'].mean()
    lon_prom = df['Longitud'].mean()
    return lat_prom, lon_prom


def crear_mapa_sucursales_cajeros(sucursales_df, cajeros_df, clientes_df=None):
    centro_lat, centro_lon = calcular_centroide(sucursales_df)
    mapa = crear_mapa_base(centro_lat, centro_lon, zoom=7)
    
    # Agregar sucursales
    for idx, row in sucursales_df.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"""
            <b>{row['Nombre']}</b><br>
            Tipo: {row['Tipo de Sucursal']}<br>
            Transacciones/mes: {row['Volumen de Transacciones (mes)']}<br>
            Empleados: {row['Número de Empleados']}
            """,
            tooltip=row['Nombre'],
            icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
        ).add_to(mapa)
    
    # Agregar cajeros
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
    
    # Opcionalmente, agregar clientes
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
    
    return mapa


def crear_mapa_cobertura_clientes(clientes_df, sucursales_df, cajeros_df, 
                                   umbral_sucursal=10.0, umbral_cajero=5.0):
    from modulos.geoespacial import (
        calcular_distancia_a_sucursal_mas_cercana,
        calcular_distancia_a_cajero_mas_cercano
    )
    
    clientes = clientes_df.copy()
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales_df)
    clientes = calcular_distancia_a_cajero_mas_cercano(clientes, cajeros_df)
    
    def determinar_color_cobertura(row):
        """Determina el color basado en la cobertura del cliente."""
        distancia_sucursal = row['Distancia_a_Sucursal_km'] <= umbral_sucursal
        distancia_cajero = row['Distancia_a_Cajero_km'] <= umbral_cajero
        
        if distancia_sucursal and distancia_cajero:
            return 'green'
        elif distancia_sucursal or distancia_cajero:
            return 'orange'
        else:
            return 'red'
    
    clientes['Color_Cobertura'] = clientes.apply(determinar_color_cobertura, axis=1)
    
    centro_lat, centro_lon = calcular_centroide(clientes)
    mapa = crear_mapa_base(centro_lat, centro_lon, zoom=7)
    
    # Agregar clientes con colores según cobertura
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
    
    # Agregar sucursales y cajeros como referencia
    for idx, row in sucursales_df.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"Sucursal: {row['Nombre']}",
            icon=folium.Icon(color='blue', icon='bank', prefix='fa'),
            opacity=0.7,
            tooltip=row['Nombre']
        ).add_to(mapa)
    
    for idx, row in cajeros_df.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"Cajero ATM",
            icon=folium.Icon(color='gray', icon='money', prefix='fa'),
            opacity=0.7
        ).add_to(mapa)
    
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


# GRÁFICOS CON PLOTLY CON ANIMACIONES

def crear_grafico_barras(datos_df, x_col, y_col, titulo="", 
                        color_col=None, mostrar_valores=True, animar=True):
    fig = px.bar(
        datos_df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=titulo,
        color_discrete_sequence=COLORES,
        template="plotly_white"
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=1, color="#ffffff")),
        textposition="outside" if mostrar_valores else "none",
        texttemplate="%{y:,.0f}" if mostrar_valores else "",
        hovertemplate="<b>%{x}</b><br>Valor: %{y:,.0f}<extra></extra>"
    )
    
    if animar:
        fig.update_layout(
            transition=dict(duration=800, easing="cubic-in-out"),
            xaxis=dict(tickangle=-45 if len(datos_df) > 5 else 0),
            title_font_size=16,
            title_font_color="#1a365d",
            height=400
        )
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_lineas(datos_df, x_col, y_col, titulo="", 
                        marker_size=6, mostrar_area=False, animar=True):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=datos_df[x_col],
        y=datos_df[y_col],
        mode='lines+markers',
        name='Tendencia',
        line=dict(
            color='#2c5aa0',
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=marker_size,
            color='#2c5aa0',
            symbol='circle',
            line=dict(color='#ffffff', width=2)
        ),
        fill='tozeroy' if mostrar_area else 'none',
        fillcolor='rgba(44, 90, 160, 0.1)' if mostrar_area else 'rgba(0,0,0,0)',
        hovertemplate='<b>%{x}</b><br>Valor: %{y:,.0f}<extra></extra>',
    ))
    
    if animar:
        fig.update_layout(
            title=titulo,
            xaxis_title=x_col,
            yaxis_title=y_col,
            transition=dict(duration=1000, easing="cubic-in-out"),
            title_font_size=16,
            title_font_color="#1a365d",
            height=400
        )
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_scatter(datos_df, x_col, y_col, titulo="", 
                         size_col=None, color_col=None, animar=True):
    fig = px.scatter(
        datos_df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        title=titulo,
        color_discrete_sequence=COLORES,
        size_max=30,
        template="plotly_white"
    )
    
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color="#ffffff"),
            opacity=0.8
        ),
        hovertemplate="<b>%{x}</b>, %{y:,.0f}<extra></extra>"
    )
    
    if animar:
        fig.update_layout(
            transition=dict(duration=1200, easing="elastic-out"),
            title_font_size=16,
            title_font_color="#1a365d",
            height=400
        )
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_pie(datos, titulo="", mostrar_porcentajes=True):
    if isinstance(datos, dict):
        labels = datos.get("labels", [])
        values = datos.get("values", [])
    else:
        labels = datos.index.tolist()
        values = datos.values.tolist()
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=COLORES[:len(labels)],
            line=dict(color='#ffffff', width=2)
        ),
        textinfo="label+percent" if mostrar_porcentajes else "label",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Valor: %{value}<br>Porcentaje: %{percent}<extra></extra>",
        sort=False
    )])
    
    fig.update_layout(
        title=titulo,
        transition=dict(duration=800, easing="cubic-in-out"),
        showlegend=True,
        title_font_size=16,
        title_font_color="#1a365d",
        height=400
    )
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_heatmap(matriz, etiquetas_x, etiquetas_y, titulo=""):
    fig = go.Figure(data=go.Heatmap(
        z=matriz,
        x=etiquetas_x,
        y=etiquetas_y,
        colorscale="Blues",
        text=np.round(matriz, 2),
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hovertemplate="<b>%{y} vs %{x}</b><br>Valor: %{z:.2f}<extra></extra>",
        colorbar=dict(
            title="Valor",
            thickness=20,
            len=0.7
        )
    ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Origen",
        yaxis_title="Destino",
        transition=dict(duration=800, easing="cubic-in-out"),
        title_font_size=16,
        title_font_color="#1a365d",
        height=600,
        width=700
    )
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_volumen_transacciones(sucursales_df):
    datos = sucursales_df.groupby('Tipo de Sucursal').agg({
        'Volumen de Transacciones (mes)': 'sum'
    }).reset_index()
    
    return crear_grafico_barras(
        datos,
        x_col='Tipo de Sucursal',
        y_col='Volumen de Transacciones (mes)',
        titulo='Volumen de Transacciones por Tipo de Sucursal',
        mostrar_valores=True
    )


def crear_grafico_empleados_vs_transacciones(sucursales_df):
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
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_productos_por_ubicacion(clientes_df):
    datos = clientes_df.groupby(['Ubicación de Residencia', 'Productos Financieros Adquiridos']).size().reset_index(name='Cantidad')
    datos['Ubicación'] = datos['Ubicación de Residencia'].astype(str)
    
    fig = px.bar(
        datos,
        x='Ubicación',
        y='Cantidad',
        color='Productos Financieros Adquiridos',
        title='Distribución de Productos Financieros por Ubicación',
        labels={'Cantidad': 'Número de Clientes'},
        template='plotly_white',
        barmode='stack',
        color_discrete_sequence=COLORES
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode='x unified'
    )
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_saldo_promedio_por_producto(clientes_df):
    datos = clientes_df.groupby('Productos Financieros Adquiridos').agg({
        'Saldo Promedio de Cuentas': 'mean',
        'Volumen de Transacciones': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=datos['Productos Financieros Adquiridos'],
        y=datos['Saldo Promedio de Cuentas'],
        name='Saldo Promedio',
        marker_color=COLORES[0],
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
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_frecuencia_visitas(clientes_df):
    datos = clientes_df['Frecuencia de Visitas'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=datos.index,
            y=datos.values,
            marker_color=COLORES[0],
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
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_transacciones_cajeros(cajeros_df):
    cajeros_copy = cajeros_df.copy()
    cajeros_copy['Cajero_ID'] = [f"Cajero {i+1}" for i in range(len(cajeros_copy))]
    
    fig = px.bar(
        cajeros_copy,
        x='Cajero_ID',
        y='Volumen de Transacciones Diarias',
        color='Tipo de Transacciones',
        title='Volumen de Transacciones Diarias por Cajero Automático',
        labels={'Volumen de Transacciones Diarias': 'Transacciones/día'},
        template='plotly_white',
        color_discrete_sequence=COLORES
    )
    
    fig.update_layout(hovermode='x unified')
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_matriz_distancias(matriz_distancias, etiquetas=None):
    if etiquetas is None:
        etiquetas = [f"Punto {i+1}" for i in range(len(matriz_distancias))]
    
    return crear_grafico_heatmap(
        matriz_distancias,
        etiquetas_x=etiquetas,
        etiquetas_y=etiquetas,
        titulo="Matriz de Distancias entre Puntos de Servicio"
    )