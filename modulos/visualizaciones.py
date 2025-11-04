import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

from modulos.geoespacial import (
        calcular_distancia_a_sucursal_mas_cercana,
        calcular_distancia_a_cajero_mas_cercano
    )


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


# FUNCIONES PARA ANÁLISIS DE COBERTURA

def crear_mapa_cobertura_con_radios(datos_ubicaciones, distancia_km=10.0, zoom_level=7):
 
    if len(datos_ubicaciones) == 0:
        return None
    
    centro_lat = datos_ubicaciones['Latitud'].mean()
    centro_lon = datos_ubicaciones['Longitud'].mean()
    
    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=zoom_level,  # ← USAR EL PARÁMETRO
        tiles="OpenStreetMap",
        prefer_canvas=True
    )
    
    colores = ['#2c5aa0', '#27ae60', '#f39c12', '#e74c3c', '#9b59b6']
    
    for idx, row in datos_ubicaciones.iterrows():
        color = colores[idx % len(colores)]
        
        # Circulo de cobertura
        folium.Circle(
            location=[row['Latitud'], row['Longitud']],
            radius=distancia_km * 1000,
            popup=f"<b>{row['Nombre']}</b><br>Radio: {distancia_km} km",
            tooltip=f"{row['Nombre']} - Cobertura {distancia_km}km",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.15,
            weight=2,
            dashArray='5, 5'
        ).add_to(mapa)
        
        # Marcador central
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=8,
            popup=f"<b>{row['Nombre']}</b>",
            tooltip=row['Nombre'],
            color='white',
            fill=True,
            fillColor=color,
            fillOpacity=0.9,
            weight=3
        ).add_to(mapa)
        
        # Etiqueta
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            icon=folium.DivIcon(html=f"""
                <div style="font-size: 12px; font-weight: bold; color: {color}; 
                            background-color: white; padding: 5px 10px; 
                            border-radius: 5px; border: 2px solid {color};
                            white-space: nowrap;">
                    {row['Nombre']}
                </div>
            """)
        ).add_to(mapa)
    
    return mapa


def crear_grafico_concentracion_clientes(datos_consolidados):

    # Agrupar por ubicación (sucursal) y sumar clientes
    concentracion = datos_consolidados.groupby('Nombre').agg({
        'Numero_Clientes_Producto': 'sum'
    }).reset_index().sort_values('Numero_Clientes_Producto', ascending=False)
    
    concentracion.columns = ['Sucursal', 'Total_Clientes']
    
    fig = px.bar(
        concentracion,
        x='Sucursal',
        y='Total_Clientes',
        title='Concentración de Clientes por Sucursal',
        labels={'Total_Clientes': 'Cantidad de Clientes', 'Sucursal': 'Sucursal'},
        template='plotly_white',
        color='Total_Clientes',
        color_continuous_scale='Blues'
    )
    
    fig.update_traces(
        textposition='outside',
        texttemplate='%{y:,.0f}',
        hovertemplate='<b>%{x}</b><br>Clientes: %{y:,.0f}<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_transacciones_por_ubicacion(datos_consolidados):
 
    transacciones = datos_consolidados.groupby('Nombre').agg({
        'Volumen_Transacciones_Sucursal': 'first'
    }).reset_index().sort_values(
        'Volumen_Transacciones_Sucursal', ascending=False
    )
    
    transacciones.columns = ['Sucursal', 'Transacciones_Mensuales']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=transacciones['Sucursal'],
        y=transacciones['Transacciones_Mensuales'],
        marker=dict(
            color=transacciones['Transacciones_Mensuales'],
            colorscale='Blues', 
            showscale=False,
            line=dict(color='white', width=1)
        ),
        text=transacciones['Transacciones_Mensuales'],
        textposition='outside',
        texttemplate='%{text:,.0f}',
        hovertemplate='<b>%{x}</b><br>Sucursal/mes: %{y:,.0f}<extra></extra>',
        name='Sucursal'
    ))
    
    fig.update_layout(
        title='Volumen de Transacciones de Sucursales (Mensual)',
        xaxis_title='Sucursal',
        yaxis_title='Transacciones/mes',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    fig = aplicar_tema(fig)
    return fig

def crear_grafico_transacciones_cajeros_por_tipo(datos_consolidados):

    cajeros_data = datos_consolidados[
        ['Nombre', 'Volumen_Transacciones_Cajero_Diarias', 'Tipos_Transacciones_Cajero']
    ].drop_duplicates(subset=['Nombre']).reset_index(drop=True)
    
    cajeros_data['Transacciones_Mensuales'] = (
        cajeros_data['Volumen_Transacciones_Cajero_Diarias'] * 30
    )
    
    cajeros_data['Retiro'] = cajeros_data['Transacciones_Mensuales'] / 3
    cajeros_data['Consulta'] = cajeros_data['Transacciones_Mensuales'] / 3
    cajeros_data['Pago'] = cajeros_data['Transacciones_Mensuales'] / 3
    
    cajeros_data = cajeros_data.sort_values(
        'Transacciones_Mensuales', ascending=False
    )
    
    fig = go.Figure()
    
    colores = {'Retiro': '#27ae60', 'Consulta': '#3498db', 'Pago': '#f39c12'}
    
    for tipo in ['Retiro', 'Consulta', 'Pago']:
        fig.add_trace(go.Bar(
            x=cajeros_data['Nombre'],
            y=cajeros_data[tipo],
            name=tipo,
            marker=dict(color=colores[tipo]),
            text=cajeros_data[tipo].astype(int),
            textposition='inside',
            texttemplate='%{text:,.0f}',
            hovertemplate='<b>%{x}</b><br>' + tipo + ': %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Volumen de Transacciones de Cajeros por Tipo (Mensual)',
        xaxis_title='Ubicación',
        yaxis_title='Transacciones/mes',
        xaxis_tickangle=-45,
        barmode='stack',
        template='plotly_white',
        height=500,
        margin=dict(b=100, r=250),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.15,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        font=dict(size=12)
    )
    
    fig = aplicar_tema(fig)
    return fig


def crear_grafico_comparativa_cobertura_clientes(datos_consolidados, distancia_km=10.0):

    # Agrupar datos por sucursal
    comparativa = datos_consolidados.groupby('Nombre').agg({
        'Numero_Clientes_Producto': 'sum',
        'Volumen_Transacciones_Sucursal': 'first',
        'Número de Empleados': 'first'
    }).reset_index()
    
    comparativa.columns = ['Sucursal', 'Clientes', 'Transacciones', 'Empleados']
    
    # Calcular área de cobertura (simplificado como círculo)
    area_cobertura = (3.14159 * distancia_km ** 2)
    comparativa['Densidad_Clientes'] = comparativa['Clientes'] / area_cobertura
    
    fig = px.scatter(
        comparativa,
        x='Transacciones',
        y='Clientes',
        size='Empleados',
        hover_name='Sucursal',
        title='Relación: Clientes vs Transacciones por Sucursal',
        labels={
            'Transacciones': 'Transacciones/mes',
            'Clientes': 'Cantidad de Clientes'
        },
        template='plotly_white',
        color_discrete_sequence=[COLORES[0]]
    )
    
    fig.update_traces(
        marker=dict(opacity=0.7, line=dict(width=2, color='white')),
        hovertemplate='<b>%{hovertext}</b><br>Transacciones: %{x:,.0f}<br>Clientes: %{y:,.0f}<extra></extra>'
    )
    
    fig.update_layout(height=400)
    fig = aplicar_tema(fig)
    return fig

def crear_mapa_segmentacion_geografica(datos_consolidados):
    
    # Agrupar datos por sucursal para obtener info consolidada
    sucursales_data = datos_consolidados.groupby(['Nombre', 'Latitud', 'Longitud']).agg({
        'Numero_Clientes_Producto': 'sum',
        'Volumen_Transacciones_Sucursal': 'first',
        'Número de Empleados': 'first',
        'Tipo de Sucursal': 'first'
    }).reset_index()
    
    # Calcular métrica de eficiencia
    sucursales_data['Clientes_por_Empleado'] = (
        sucursales_data['Numero_Clientes_Producto'] / 
        sucursales_data['Número de Empleados']
    ).round(2)
    
    sucursales_data['Transacciones_por_Cliente'] = (
        sucursales_data['Volumen_Transacciones_Sucursal'] / 
        sucursales_data['Numero_Clientes_Producto']
    ).round(2)
    
    # Centro del mapa
    centro_lat = sucursales_data['Latitud'].mean()
    centro_lon = sucursales_data['Longitud'].mean()
    
    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=7,
        tiles="OpenStreetMap",
        prefer_canvas=True
    )
    
    # Normalizar valores para colores y tamaños
    min_clientes = sucursales_data['Numero_Clientes_Producto'].min()
    max_clientes = sucursales_data['Numero_Clientes_Producto'].max()
    
    min_transacciones = sucursales_data['Volumen_Transacciones_Sucursal'].min()
    max_transacciones = sucursales_data['Volumen_Transacciones_Sucursal'].max()
    
    def obtener_color(transacciones):
        """Escala de colores: Rojo (bajo) -> Amarillo -> Verde (alto)"""
        proporcion = (transacciones - min_transacciones) / (max_transacciones - min_transacciones)
        if proporcion < 0.33:
            return '#e74c3c'  # Rojo - bajo
        elif proporcion < 0.66:
            return '#f39c12'  # Amarillo - medio
        else:
            return '#27ae60'  # Verde - alto
    
    def obtener_tamaño(clientes):
        proporcion = (clientes - min_clientes) / (max_clientes - min_clientes)
        return 35 + (proporcion * 40)
    
    # Agregar marcadores para cada sucursal
    for idx, row in sucursales_data.iterrows():
        color = obtener_color(row['Volumen_Transacciones_Sucursal'])
        tamaño = obtener_tamaño(row['Numero_Clientes_Producto'])
        
        popup_text = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 5px 0; color: #2c5aa0;">{row['Nombre']}</h4>
            <hr style="margin: 5px 0;">
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td><b>Tipo:</b></td>
                    <td>{row['Tipo de Sucursal']}</td>
                </tr>
                <tr>
                    <td><b>Clientes:</b></td>
                    <td>{int(row['Numero_Clientes_Producto'])}</td>
                </tr>
                <tr>
                    <td><b>Transacciones/mes:</b></td>
                    <td>{int(row['Volumen_Transacciones_Sucursal']):,}</td>
                </tr>
                <tr>
                    <td><b>Empleados:</b></td>
                    <td>{int(row['Número de Empleados'])}</td>
                </tr>
                <tr>
                    <td><b>Clientes/Emp:</b></td>
                    <td>{row['Clientes_por_Empleado']}</td>
                </tr>
                <tr>
                    <td><b>Trans/Cliente:</b></td>
                    <td>{row['Transacciones_por_Cliente']}</td>
                </tr>
            </table>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=tamaño,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['Nombre']}: {int(row['Numero_Clientes_Producto'])} clientes",
            color='white',
            fill=True,
            fillColor=color,
            fillOpacity=0.85,
            weight=3
        ).add_to(mapa)
        
        # Etiqueta con nombre
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            icon=folium.DivIcon(html=f"""
                <div style="font-size: 11px; font-weight: bold; 
                            background-color: white; padding: 3px 6px; 
                            border-radius: 3px; border: 2px solid {color};
                            white-space: nowrap;">
                    {row['Nombre'].split()[-1]}
                </div>
            """)
        ).add_to(mapa)
    
    
    return mapa, sucursales_data

def crear_mapa_rutas_mantenimiento(sucursales_df, rutas_df):
 
    # Centro del mapa
    centro_lat = sucursales_df['Latitud'].mean()
    centro_lon = sucursales_df['Longitud'].mean()
    
    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=7,
        tiles="OpenStreetMap",
        prefer_canvas=True
    )
    
    # Colores
    color_principal = '#2c5aa0'
    color_secundaria = '#27ae60'
    color_linea = '#f39c12'
    
    # Agregar Sucursales Principales (marcadores grandes)
    principales = sucursales_df[sucursales_df['Tipo de Sucursal'] == 'Sucursal Principal']
    
    for idx, row in principales.iterrows():
        # Calcular cuántas sucursales atiende
        rutas_desde = rutas_df[rutas_df['Sucursal_Origen'] == row['Nombre']]
        num_atendidas = len(rutas_desde)
        distancia_total = rutas_desde['Distancia_km'].sum()
        
        popup_text = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 5px 0; color: {color_principal};">
                {row['Nombre']}
            </h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0;"><b>Tipo:</b> {row['Tipo de Sucursal']}</p>
            <p style="margin: 5px 0;"><b>Sucursales Atendidas:</b> {num_atendidas}</p>
            <p style="margin: 5px 0;"><b>Distancia Total:</b> {distancia_total:.2f} km</p>
            <p style="margin: 5px 0;"><b>Empleados:</b> {row['Número de Empleados']}</p>
        </div>
        """
        
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['Nombre']} - {num_atendidas} rutas",
            icon=folium.Icon(
                color='blue',
                icon='home',
                prefix='fa'
            )
        ).add_to(mapa)
        
        # Círculo grande alrededor
        folium.Circle(
            location=[row['Latitud'], row['Longitud']],
            radius=2000,
            popup=row['Nombre'],
            color=color_principal,
            fill=True,
            fillColor=color_principal,
            fillOpacity=0.15,
            weight=2
        ).add_to(mapa)
    
    # Agregar Sucursales Secundarias (marcadores pequeños)
    secundarias = sucursales_df[sucursales_df['Tipo de Sucursal'] == 'Sucursal Secundaria']
    
    for idx, row in secundarias.iterrows():
        # Encontrar desde qué principal es atendida
        ruta_info = rutas_df[rutas_df['Sucursal_Destino'] == row['Nombre']]
        
        if len(ruta_info) > 0:
            origen = ruta_info.iloc[0]['Sucursal_Origen']
            distancia = ruta_info.iloc[0]['Distancia_km']
            tiempo = ruta_info.iloc[0]['Tiempo_Estimado_min']
        else:
            origen = "No asignada"
            distancia = 0
            tiempo = 0
        
        popup_text = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 5px 0; color: {color_secundaria};">
                {row['Nombre']}
            </h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0;"><b>Tipo:</b> {row['Tipo de Sucursal']}</p>
            <p style="margin: 5px 0;"><b>Atendida desde:</b> {origen}</p>
            <p style="margin: 5px 0;"><b>Distancia:</b> {distancia:.2f} km</p>
            <p style="margin: 5px 0;"><b>Tiempo estimado:</b> {tiempo} min</p>
            <p style="margin: 5px 0;"><b>Empleados:</b> {row['Número de Empleados']}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['Nombre']}",
            color='white',
            fill=True,
            fillColor=color_secundaria,
            fillOpacity=0.9,
            weight=3
        ).add_to(mapa)
    
    # Agregar líneas de rutas
    for idx, ruta in rutas_df.iterrows():
        # Línea de ruta
        folium.PolyLine(
            locations=[
                [ruta['Lat_Origen'], ruta['Lon_Origen']],
                [ruta['Lat_Destino'], ruta['Lon_Destino']]
            ],
            color=color_linea,
            weight=3,
            opacity=0.7,
            popup=f"""
            <b>Ruta:</b> {ruta['Sucursal_Origen']} → {ruta['Sucursal_Destino']}<br>
            <b>Distancia:</b> {ruta['Distancia_km']:.2f} km<br>
            <b>Tiempo:</b> {ruta['Tiempo_Estimado_min']} min
            """,
            tooltip=f"{ruta['Distancia_km']:.2f} km"
        ).add_to(mapa)
        
        # Flecha direccional (usando un marcador pequeño en el punto medio)
        lat_medio = (ruta['Lat_Origen'] + ruta['Lat_Destino']) / 2
        lon_medio = (ruta['Lon_Origen'] + ruta['Lon_Destino']) / 2
        
        folium.CircleMarker(
            location=[lat_medio, lon_medio],
            radius=4,
            color=color_linea,
            fill=True,
            fillColor=color_linea,
            fillOpacity=1.0,
            weight=2,
            tooltip=f"{ruta['Distancia_km']:.2f} km"
        ).add_to(mapa)
    
    
    return mapa
