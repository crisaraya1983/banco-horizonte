"""
Módulo de Análisis - Páginas de la Aplicación COMPLETO
========================================================
Este módulo contiene la implementación completa de todas las páginas de análisis
del dashboard. Incluye:

1. Análisis de Cobertura Geográfica (Completamente implementado)
2. Segmentación Geográfica (Nuevo)
3. Optimización Logística (Nuevo)
4. Marketing Dirigido (Nuevo)
5. Predicción de Demanda (Nuevo)
6. Análisis de Riesgos (Nuevo)

Cada página es independiente pero reutiliza funciones de los módulos
geoespacial.py y visualizaciones.py para mantener separación de responsabilidades.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

# Importamos nuestros módulos personalizados
from modulos.carga_datos import cargar_sucursales, cargar_cajeros, cargar_clientes, cargar_productos
from modulos.geoespacial import (
    calcular_cobertura_geográfica,
    calcular_distancia_a_sucursal_mas_cercana,
    calcular_distancia_a_cajero_mas_cercano,
    identificar_zonas_desatendidas,
    crear_matriz_distancias,
    agrupar_clientes_por_proximidad,
    calcular_densidad_clientes_por_sucursal,
    calcular_centroide_geográfico,
    distancia_haversine
)
from modulos.visualizaciones import (
    crear_mapa_sucursales_cajeros,
    crear_mapa_cobertura_clientes,
    crear_grafico_volumen_transacciones,
    crear_grafico_empleados_vs_transacciones,
    crear_grafico_productos_por_ubicacion,
    crear_grafico_saldo_promedio_por_producto,
    crear_grafico_frecuencia_visitas,
    crear_grafico_transacciones_cajeros,
    crear_grafico_matriz_distancias
)


# ============================================================================
# PÁGINA 1: ANÁLISIS DE COBERTURA GEOGRÁFICA (COMPLETO)
# ============================================================================

def pagina_analisis_cobertura():
    """
    Página principal de análisis de cobertura geográfica.
    
    Esta página responde preguntas críticas como:
    - ¿Dónde están ubicados nuestros puntos de servicio?
    - ¿Qué tan bien cubrimos geográficamente a nuestros clientes?
    - ¿Existen áreas desatendidas con alta concentración de clientes?
    - ¿Cómo se distribuye la carga de trabajo entre sucursales?
    """
    
    st.markdown('<div class="main-header">📍 Análisis de Cobertura Geográfica</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    En esta sección analizamos cómo se distribuyen geográficamente nuestros puntos de servicio
    (sucursales y cajeros automáticos) en relación con nuestros clientes. Este análisis es
    fundamental para identificar oportunidades de expansión y evaluar la calidad del servicio.
    """)
    
    st.markdown("---")
    
    # Cargamos los datos
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    productos = cargar_productos()
    
    # Métricas de cobertura general
    st.markdown('<div class="subheader-custom">📊 Resumen de Cobertura General</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        umbral_sucursal = st.slider(
            "Distancia máxima aceptable a Sucursal (km)",
            min_value=1.0, max_value=20.0, value=10.0, step=0.5,
            help="Los clientes dentro de esta distancia se consideran cubiertos por sucursal"
        )
    
    with col2:
        umbral_cajero = st.slider(
            "Distancia máxima aceptable a Cajero (km)",
            min_value=1.0, max_value=15.0, value=5.0, step=0.5,
            help="Los clientes dentro de esta distancia se consideran cubiertos por cajero"
        )
    
    cobertura = calcular_cobertura_geográfica(
        clientes, cajeros, sucursales,
        umbral_sucursal=umbral_sucursal,
        umbral_cajero=umbral_cajero
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Cobertura de Sucursales",
            value=f"{cobertura['cobertura_sucursales_pct']:.1f}%",
            delta="de clientes cubiertos"
        )
    
    with col2:
        st.metric(
            label="Cobertura de Cajeros",
            value=f"{cobertura['cobertura_cajeros_pct']:.1f}%",
            delta="de clientes cubiertos"
        )
    
    with col3:
        st.metric(
            label="Cobertura Completa",
            value=f"{cobertura['cobertura_completa_pct']:.1f}%",
            delta="de clientes bien servidos"
        )
    
    with col4:
        st.metric(
            label="Clientes Desatendidos",
            value=cobertura['clientes_sin_cobertura_completa'],
            delta="requieren expansión"
        )
    
    st.markdown("---")
    
    # Mapas interactivos
    st.markdown('<div class="subheader-custom">🗺️ Visualización Geoespacial</div>', 
                unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Distribución de Puntos de Servicio", "Mapa de Cobertura de Clientes"])
    
    with tab1:
        st.markdown("**Distribución de Sucursales, Cajeros y Clientes**")
        try:
            mapa1 = crear_mapa_sucursales_cajeros(sucursales, cajeros, clientes)
            st.components.v1.html(mapa1._repr_html_(), height=600, width=None)
        except Exception as e:
            st.error(f"Error al crear el mapa: {e}")
    
    with tab2:
        st.markdown("**Estado de Cobertura de Clientes**")
        try:
            mapa2 = crear_mapa_cobertura_clientes(
                clientes, sucursales, cajeros,
                umbral_sucursal=umbral_sucursal,
                umbral_cajero=umbral_cajero
            )
            st.components.v1.html(mapa2._repr_html_(), height=600, width=None)
        except Exception as e:
            st.error(f"Error al crear el mapa: {e}")
    
    st.markdown("---")
    
    # Zonas desatendidas
    st.markdown('<div class="subheader-custom">⚠️ Identificación de Zonas Desatendidas</div>', 
                unsafe_allow_html=True)
    
    clientes_analisis = clientes.copy()
    clientes_analisis = calcular_distancia_a_sucursal_mas_cercana(clientes_analisis, sucursales)
    clientes_analisis = calcular_distancia_a_cajero_mas_cercano(clientes_analisis, cajeros)
    
    desatendidos = identificar_zonas_desatendidas(
        clientes_analisis, sucursales, 
        umbral_km=umbral_sucursal
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Clientes en Zonas Desatendidas",
            value=len(desatendidos),
            delta=f"de {len(clientes)}"
        )
    
    with col2:
        if len(clientes) > 0:
            porcentaje = (len(desatendidos) / len(clientes)) * 100
            st.metric(
                label="Porcentaje Desatendido",
                value=f"{porcentaje:.1f}%",
                delta="de la base de clientes"
            )
    
    with col3:
        distancia_prom = desatendidos['Distancia_a_Sucursal_km'].mean() if len(desatendidos) > 0 else 0
        st.metric(
            label="Distancia Promedio",
            value=f"{distancia_prom:.2f} km",
            delta="a sucursal más cercana"
        )
    
    if len(desatendidos) > 0:
        st.markdown("**Clientes en Zonas Desatendidas:**")
        tabla_mostrar = desatendidos[[
            'Ubicación de Residencia',
            'Productos Financieros Adquiridos',
            'Distancia_a_Sucursal_km',
            'Saldo Promedio de Cuentas'
        ]].copy()
        tabla_mostrar.columns = ['Ubicación', 'Producto', 'Distancia (km)', 'Saldo Promedio']
        st.dataframe(tabla_mostrar, use_container_width=True)
    
    st.markdown("---")
    
    # Gráficos de análisis
    st.markdown('<div class="subheader-custom">📈 Análisis de Volumen</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_vol = crear_grafico_volumen_transacciones(sucursales)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        fig_emp = crear_grafico_empleados_vs_transacciones(sucursales)
        st.plotly_chart(fig_emp, use_container_width=True)


# ============================================================================
# PÁGINA 2: SEGMENTACIÓN GEOGRÁFICA
# ============================================================================

def pagina_segmentacion_geografica():
    """
    Análisis de segmentación geográfica de clientes.
    
    Esta página agrupa clientes en zonas geográficas y analiza
    sus características, preferencias de productos y patrones de comportamiento.
    Responde preguntas como: ¿Qué características tiene cada zona? ¿Cuál es
    más valiosa? ¿Qué productos se prefieren en cada región?
    """
    
    st.markdown('<div class="main-header">🎯 Segmentación Geográfica de Clientes</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    En esta sección segmentamos nuestros clientes en zonas geográficas y analizamos
    sus características, productos preferidos y patrones de comportamiento. Esto es
    esencial para diseñar estrategias de marketing y operacionales personalizadas por región.
    """)
    
    st.markdown("---")
    
    # Cargamos datos
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    
    # Número de zonas para segmentación
    num_zonas = st.slider(
        "Número de zonas geográficas a identificar",
        min_value=2, max_value=5, value=3,
        help="Usamos clustering K-means para dividir el territorio en zonas"
    )
    
    # Realizamos clustering geográfico
    coords = clientes[['Latitud', 'Longitud']].values
    kmeans = KMeans(n_clusters=num_zonas, random_state=42, n_init=10)
    clientes['Zona'] = kmeans.fit_predict(coords) + 1
    
    st.markdown("---")
    
    # Análisis por zona
    st.markdown('<div class="subheader-custom">📊 Características por Zona</div>', 
                unsafe_allow_html=True)
    
    # Creamos tabla resumen por zona
    resumen_zonas = []
    for zona in range(1, num_zonas + 1):
        clientes_zona = clientes[clientes['Zona'] == zona]
        if len(clientes_zona) > 0:
            resumen_zonas.append({
                'Zona': f"Zona {zona}",
                'Cantidad de Clientes': len(clientes_zona),
                'Saldo Promedio': f"${clientes_zona['Saldo Promedio de Cuentas'].mean():,.0f}",
                'Frecuencia Visitas': f"{clientes_zona['Frecuencia de Visitas'].mean():.1f}",
                'Volumen Transacciones': int(clientes_zona['Volumen de Transacciones'].mean()),
                'Producto Preferido': clientes_zona['Productos Financieros Adquiridos'].mode()[0] if len(clientes_zona) > 0 else 'N/A'
            })
    
    resumen_df = pd.DataFrame(resumen_zonas)
    st.dataframe(resumen_df, use_container_width=True)
    
    st.markdown("---")
    
    # Mapa de zonas
    st.markdown('<div class="subheader-custom">🗺️ Mapa de Segmentación Geográfica</div>', 
                unsafe_allow_html=True)
    
    st.markdown("**Distribución de Clientes por Zona**")
    
    try:
        import folium
        
        centro_lat = clientes['Latitud'].mean()
        centro_lon = clientes['Longitud'].mean()
        mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=7)
        
        colores = ['red', 'blue', 'green', 'purple', 'orange']
        
        for zona in range(1, num_zonas + 1):
            clientes_zona = clientes[clientes['Zona'] == zona]
            color = colores[(zona - 1) % len(colores)]
            
            for idx, cliente in clientes_zona.iterrows():
                folium.CircleMarker(
                    location=[cliente['Latitud'], cliente['Longitud']],
                    radius=7,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    popup=f"Zona {zona}<br>Producto: {cliente['Productos Financieros Adquiridos']}"
                ).add_to(mapa)
            
            # Centro de la zona
            centro = clientes_zona[['Latitud', 'Longitud']].mean()
            folium.Marker(
                location=[centro['Latitud'], centro['Longitud']],
                popup=f"Centro Zona {zona}",
                icon=folium.Icon(color=color, icon='info-sign', prefix='glyphicon')
            ).add_to(mapa)
        
        st.components.v1.html(mapa._repr_html_(), height=600, width=None)
    except Exception as e:
        st.error(f"Error al crear mapa: {e}")
    
    st.markdown("---")
    
    # Análisis de productos por zona
    st.markdown('<div class="subheader-custom">💼 Análisis de Productos por Zona</div>', 
                unsafe_allow_html=True)
    
    # Gráfico de barras apiladas
    productos_zona = clientes.groupby(['Zona', 'Productos Financieros Adquiridos']).size().reset_index(name='Cantidad')
    productos_zona['Zona'] = 'Zona ' + productos_zona['Zona'].astype(str)
    
    fig_productos = px.bar(
        productos_zona,
        x='Zona',
        y='Cantidad',
        color='Productos Financieros Adquiridos',
        title='Distribución de Productos Financieros por Zona',
        barmode='stack',
        template='plotly_white'
    )
    st.plotly_chart(fig_productos, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de saldo por zona
    st.markdown('<div class="subheader-custom">💰 Valor de Clientes por Zona</div>', 
                unsafe_allow_html=True)
    
    saldo_zona = []
    for zona in range(1, num_zonas + 1):
        clientes_zona = clientes[clientes['Zona'] == zona]
        saldo_zona.append({
            'Zona': f'Zona {zona}',
            'Saldo Promedio': clientes_zona['Saldo Promedio de Cuentas'].mean()
        })
    
    saldo_df = pd.DataFrame(saldo_zona)
    
    fig_saldo = px.bar(
        saldo_df,
        x='Zona',
        y='Saldo Promedio',
        title='Saldo Promedio de Clientes por Zona',
        text='Saldo Promedio',
        template='plotly_white',
        color='Zona'
    )
    fig_saldo.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig_saldo, use_container_width=True)
    
    st.markdown("---")
    
    # Recomendaciones
    st.markdown('<div class="subheader-custom">💡 Recomendaciones por Zona</div>', 
                unsafe_allow_html=True)
    
    for zona in range(1, num_zonas + 1):
        clientes_zona = clientes[clientes['Zona'] == zona]
        saldo_prom = clientes_zona['Saldo Promedio de Cuentas'].mean()
        producto_preferido = clientes_zona['Productos Financieros Adquiridos'].mode()[0]
        
        st.info(f"""
        **Zona {zona}:** 
        - {len(clientes_zona)} clientes con saldo promedio ${saldo_prom:,.0f}
        - Producto preferido: {producto_preferido}
        - Estrategia recomendada: Enfoque en marketing de {producto_preferido.lower()}
        """)


# ============================================================================
# PÁGINA 3: OPTIMIZACIÓN LOGÍSTICA
# ============================================================================

def pagina_optimizacion_logistica():
    """
    Análisis de optimización logística para mantenimiento de cajeros.
    
    Esta página analiza cómo optimizar las rutas de mantenimiento de
    cajeros automáticos para reducir costos operacionales. Resuelve
    versiones simplificadas del problema del vendedor viajero.
    """
    
    st.markdown('<div class="main-header">🚚 Optimización Logística de Cajeros</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    En esta sección optimizamos las rutas de mantenimiento y reposición de efectivo
    para los cajeros automáticos. El objetivo es minimizar la distancia total viajada
    mientras se visitan todos los puntos, reduciendo costos operacionales.
    """)
    
    st.markdown("---")
    
    # Cargamos datos
    cajeros = cargar_cajeros()
    
    st.markdown('<div class="subheader-custom">📊 Resumen de Cajeros</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total de Cajeros", value=len(cajeros))
    
    with col2:
        transacciones_total = cajeros['Volumen de Transacciones Diarias'].sum()
        st.metric(label="Transacciones/día (Total)", value=int(transacciones_total))
    
    with col3:
        distancia_prom = cajeros['Latitud'].std() + cajeros['Longitud'].std()
        st.metric(label="Dispersión Geográfica", value=f"{distancia_prom:.2f}")
    
    st.markdown("---")
    
    # Matriz de distancias
    st.markdown('<div class="subheader-custom">📐 Matriz de Distancias entre Cajeros</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    La matriz muestra las distancias en kilómetros entre cada par de cajeros.
    Esto es la base para calcular rutas óptimas de mantenimiento.
    """)
    
    matriz_dist = crear_matriz_distancias(cajeros)
    fig_matriz = crear_grafico_matriz_distancias(
        matriz_dist,
        etiquetas=[f"Cajero {i+1}" for i in range(len(cajeros))]
    )
    st.plotly_chart(fig_matriz, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de transacciones por cajero
    st.markdown('<div class="subheader-custom">📈 Carga de Trabajo por Cajero</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    La frecuencia de mantenimiento debe correlacionar con el volumen de transacciones.
    Los cajeros con más transacciones necesitan más atención (reposición de efectivo más frecuente).
    """)
    
    fig_cajeros = crear_grafico_transacciones_cajeros(cajeros)
    st.plotly_chart(fig_cajeros, use_container_width=True)
    
    st.markdown("---")
    
    # Recomendaciones de ruta
    st.markdown('<div class="subheader-custom">🗺️ Propuesta de Ruta Óptima</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Basado en la matriz de distancias, esta es una ruta propuesta que minimiza
    la distancia total a recorrer. El orden sugiere visitar los cajeros en la
    secuencia que minimiza "saltos" geográficos grandes.
    """)
    
    # Greedy nearest neighbor algorithm (simplificado)
    visitados = set()
    posicion_actual = 0
    ruta = [0]
    distancia_total = 0
    
    while len(visitados) < len(cajeros):
        visitados.add(posicion_actual)
        # Encuentra el cajero no visitado más cercano
        distancias_candidatos = []
        for j in range(len(cajeros)):
            if j not in visitados:
                distancias_candidatos.append((matriz_dist[posicion_actual, j], j))
        
        if distancias_candidatos:
            dist_min, siguiente = min(distancias_candidatos)
            distancia_total += dist_min
            ruta.append(siguiente)
            posicion_actual = siguiente
    
    # Volver al inicio
    distancia_total += matriz_dist[ruta[-1], ruta[0]]
    
    # Crear tabla de ruta
    ruta_tabla = []
    for i, idx_cajero in enumerate(ruta):
        ruta_tabla.append({
            'Orden': i + 1,
            'Cajero': f"Cajero {idx_cajero + 1}",
            'Latitud': f"{cajeros.iloc[idx_cajero]['Latitud']:.4f}",
            'Longitud': f"{cajeros.iloc[idx_cajero]['Longitud']:.4f}",
            'Transacciones/día': int(cajeros.iloc[idx_cajero]['Volumen de Transacciones Diarias'])
        })
    
    st.dataframe(pd.DataFrame(ruta_tabla), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Distancia Total de Ruta", value=f"{distancia_total:.2f} km")
    with col2:
        st.metric(label="Distancia Promedio por Tramo", value=f"{distancia_total / len(cajeros):.2f} km")
    
    st.markdown("""
    **Interpretación:** Esta ruta minimiza la distancia total viajada. En una operación real,
    necesitarías considerar también factores como horarios de apertura, volumen de efectivo,
    y congestión vial.
    """)


# ============================================================================
# PÁGINA 4: MARKETING DIRIGIDO
# ============================================================================

def pagina_marketing_dirigido():
    """
    Análisis de marketing dirigido basado en geolocalización.
    
    Identifica oportunidades de marketing segmentadas por ubicación y
    producto, permitiendo campañas personalizadas según características
    geográficas y de comportamiento de clientes.
    """
    
    st.markdown('<div class="main-header">📢 Marketing Dirigido por Geolocalización</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    En esta sección diseñamos estrategias de marketing dirigido basadas en
    características geográficas y comportamiento de clientes. Identificamos
    oportunidades de cross-selling y upselling por región.
    """)
    
    st.markdown("---")
    
    # Cargamos datos
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    
    # Calculamos distancias
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales)
    
    st.markdown('<div class="subheader-custom">📊 Análisis de Valor de Clientes</div>', 
                unsafe_allow_html=True)
    
    # Segmentamos clientes por valor (saldo)
    clientes['Segmento_Valor'] = pd.cut(
        clientes['Saldo Promedio de Cuentas'],
        bins=[0, 3000, 5000, float('inf')],
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    # Estadísticas por segmento
    seg_stats = clientes.groupby('Segmento_Valor').agg({
        'Saldo Promedio de Cuentas': 'mean',
        'Frecuencia de Visitas': 'mean',
        'Volumen de Transacciones': 'mean'
    }).round(0)
    
    st.dataframe(seg_stats, use_container_width=True)
    
    st.markdown("---")
    
    # Gráfico de segmentación
    st.markdown('<div class="subheader-custom">🎯 Distribución de Clientes por Segmento</div>', 
                unsafe_allow_html=True)
    
    seg_dist = clientes['Segmento_Valor'].value_counts()
    
    fig_seg = px.pie(
        values=seg_dist.values,
        names=seg_dist.index,
        title='Distribución de Clientes por Segmento de Valor',
        color_discrete_map={'Bajo': '#FF6B6B', 'Medio': '#FFA500', 'Alto': '#4ECDC4'}
    )
    st.plotly_chart(fig_seg, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de productos por segmento
    st.markdown('<div class="subheader-custom">💼 Preferencia de Productos por Segmento</div>', 
                unsafe_allow_html=True)
    
    prod_seg = pd.crosstab(clientes['Segmento_Valor'], clientes['Productos Financieros Adquiridos'])
    
    fig_prod_seg = px.bar(
        prod_seg,
        barmode='group',
        title='Productos Financieros por Segmento de Cliente',
        labels={'value': 'Cantidad de Clientes', 'index': 'Segmento'},
        template='plotly_white'
    )
    st.plotly_chart(fig_prod_seg, use_container_width=True)
    
    st.markdown("---")
    
    # Proximidad vs valor
    st.markdown('<div class="subheader-custom">📍 Análisis: Proximidad vs Valor de Cliente</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Este análisis examina si existe relación entre distancia a la sucursal
    más cercana y el valor del cliente (saldo promedio). Esto es importante
    para decidir dónde invertir en expansión.
    """)
    
    fig_prox_valor = px.scatter(
        clientes,
        x='Distancia_a_Sucursal_km',
        y='Saldo Promedio de Cuentas',
        color='Productos Financieros Adquiridos',
        size='Volumen de Transacciones',
        hover_data=['Frecuencia de Visitas'],
        title='Proximidad a Sucursal vs Valor de Cliente',
        labels={
            'Distancia_a_Sucursal_km': 'Distancia a Sucursal (km)',
            'Saldo Promedio de Cuentas': 'Saldo Promedio ($)'
        },
        template='plotly_white'
    )
    st.plotly_chart(fig_prox_valor, use_container_width=True)
    
    st.markdown("---")
    
    # Recomendaciones de marketing
    st.markdown('<div class="subheader-custom">💡 Estrategias de Marketing Recomendadas</div>', 
                unsafe_allow_html=True)
    
    # Estadísticas para recomendaciones
    clientes_alto_valor = clientes[clientes['Segmento_Valor'] == 'Alto']
    producto_preferido_alto = clientes_alto_valor['Productos Financieros Adquiridos'].mode()[0]
    
    clientes_desatendidos = clientes[clientes['Distancia_a_Sucursal_km'] > 10]
    
    st.info(f"""
    **Estrategia 1 - Enfoque Premium:**
    - Target: {len(clientes_alto_valor)} clientes de alto valor
    - Producto Focus: {producto_preferido_alto}
    - Tácticas: Programas VIP, tasas preferenciales, asesoría personalizada
    """)
    
    st.info(f"""
    **Estrategia 2 - Retención en Zonas Distantes:**
    - Target: {len(clientes_desatendidos)} clientes alejados de sucursal
    - Desafío: Alta fricción para acceder a servicios
    - Tácticas: Impulsar banca digital, cajeros cercanos, cajeros inteligentes
    """)
    
    # Análisis por producto
    for producto in clientes['Productos Financieros Adquiridos'].unique():
        clientes_prod = clientes[clientes['Productos Financieros Adquiridos'] == producto]
        saldo_prom = clientes_prod['Saldo Promedio de Cuentas'].mean()
        
        st.info(f"""
        **{producto}:**
        - Clientes: {len(clientes_prod)}
        - Saldo Promedio: ${saldo_prom:,.0f}
        - Oportunidad: Cross-sell a clientes con otros productos
        """)


# ============================================================================
# PÁGINA 5: PREDICCIÓN DE DEMANDA
# ============================================================================

def pagina_prediccion_demanda():
    """
    Predicción de demanda basada en factores geoespaciales.
    
    Utiliza machine learning simple para predecir el volumen de transacciones
    futuras basado en características geográficas como densidad de clientes,
    proximidad a sucursales, y otros factores.
    """
    
    st.markdown('<div class="main-header">🔮 Predicción de Demanda Geoespacial</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    En esta sección predecimos la demanda futura de servicios bancarios basada en
    factores geoespaciales. Esto ayuda a justificar inversiones en nuevas sucursales
    o expansión de servicios.
    """)
    
    st.markdown("---")
    
    # Cargamos datos
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    
    # Preparamos características
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales)
    clientes = calcular_distancia_a_cajero_mas_cercano(clientes, cajeros)
    
    st.markdown('<div class="subheader-custom">📊 Modelo Predictivo</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    El modelo usa regresión lineal para predecir volumen de transacciones basado en:
    - Distancia a la sucursal más cercana
    - Distancia al cajero automático más cercano
    - Frecuencia de visitas del cliente
    - Saldo promedio de la cuenta
    """)
    
    # Preparamos datos para el modelo
    X = clientes[[
        'Distancia_a_Sucursal_km',
        'Distancia_a_Cajero_km',
        'Frecuencia de Visitas',
        'Saldo Promedio de Cuentas'
    ]].values
    
    y = clientes['Volumen de Transacciones'].values
    
    # Normalizamos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenamos el modelo
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)
    
    # Score del modelo
    score = modelo.score(X_scaled, y)
    
    st.markdown(f"""
    **Rendimiento del Modelo:**
    - R² Score: {score:.3f} (explica {score*100:.1f}% de la varianza)
    """)
    
    st.markdown("---")
    
    # Importancia de features
    st.markdown('<div class="subheader-custom">🔍 Importancia de Factores</div>', 
                unsafe_allow_html=True)
    
    # Calculamos importancia (valor absoluto de coeficientes normalizados)
    feature_names = [
        'Dist. Sucursal',
        'Dist. Cajero',
        'Frecuencia Visitas',
        'Saldo Promedio'
    ]
    
    importancia = np.abs(modelo.coef_)
    importancia_norm = importancia / importancia.sum()
    
    fig_imp = px.bar(
        x=feature_names,
        y=importancia_norm,
        title='Importancia de Factores en Predicción de Transacciones',
        labels={'x': 'Factor', 'y': 'Importancia Relativa'},
        template='plotly_white',
        text=importancia_norm
    )
    fig_imp.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    **Interpretación:**
    - Factores con mayor valor tienen más influencia en la predicción
    - Esto ayuda a entender qué variables son más importantes para la demanda
    """)
    
    st.markdown("---")
    
    # Predicciones para nuevas ubicaciones
    st.markdown('<div class="subheader-custom">🆕 Predicción para Nueva Ubicación</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Simula la demanda esperada si abrimos una sucursal en una nueva ubicación
    con características específicas.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dist_suc = st.number_input("Distancia a sucursal más cercana (km)", value=7.0)
    with col2:
        dist_caj = st.number_input("Distancia a cajero más cercano (km)", value=3.0)
    with col3:
        freq_vis = st.number_input("Frecuencia de visitas esperada (veces/mes)", value=3.0)
    with col4:
        saldo_prom = st.number_input("Saldo promedio esperado ($)", value=5000.0)
    
    # Hacemos predicción
    X_nuevo = np.array([[dist_suc, dist_caj, freq_vis, saldo_prom]])
    X_nuevo_scaled = scaler.transform(X_nuevo)
    prediccion = modelo.predict(X_nuevo_scaled)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Transacciones Predichas",
            value=f"{int(max(0, prediccion))}",
            delta="por cliente"
        )
    
    with col2:
        valor_potencial = prediccion * 20  # Asumiendo $20 de ingreso por transacción
        st.metric(
            label="Ingreso Potencial Estimado",
            value=f"${max(0, valor_potencial):,.0f}",
            delta="por cliente anualmente"
        )
    
    st.markdown("---")
    
    # Recomendaciones
    st.markdown('<div class="subheader-custom">💡 Recomendaciones Basadas en Predicción</div>', 
                unsafe_allow_html=True)
    
    if prediccion > clientes['Volumen de Transacciones'].mean():
        st.success("""
        **✅ Ubicación Prometedora:** La predicción sugiere que esta nueva ubicación
        tendría demanda arriba del promedio. Considera prioritaria para expansión.
        """)
    elif prediccion > clientes['Volumen de Transacciones'].quantile(0.25):
        st.info("""
        **⚠️ Ubicación Promedio:** Demanda en línea con el promedio. Viabilidad dependería
        de otros factores como costo de operación y competencia.
        """)
    else:
        st.warning("""
        **❌ Ubicación Débil:** La predicción sugiere baja demanda. Reconsidera o mejora
        el perfil de la ubicación (mejor accesibilidad, más densidad de clientes).
        """)


# ============================================================================
# PÁGINA 6: ANÁLISIS DE RIESGOS
# ============================================================================

def pagina_analisis_riesgos():
    """
    Análisis de riesgos geoespaciales para el banco.
    
    Identifica vulnerabilidades geográficas como concentración de clientes,
    dependencia de zonas específicas, y riesgos de disrupción regional.
    """
    
    st.markdown('<div class="main-header">⚠️ Análisis de Riesgos Geoespaciales</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    En esta sección identificamos riesgos geográficos que podrían afectar
    las operaciones del banco. Evaluamos concentración de clientes, dependencia
    de zonas específicas, y vulnerabilidades ante disrupciones regionales.
    """)
    
    st.markdown("---")
    
    # Cargamos datos
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    
    # Segmentamos en zonas
    coords = clientes[['Latitud', 'Longitud']].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clientes['Zona'] = kmeans.fit_predict(coords)
    
    st.markdown('<div class="subheader-custom">📊 Concentración de Clientes</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    El análisis de concentración evalúa qué porcentaje de clientes están
    en pocas zonas geográficas. Alta concentración indica alto riesgo: 
    un evento en esa zona afectaría muchos clientes.
    """)
    
    # Análisis de concentración
    clientes_por_zona = clientes['Zona'].value_counts().sort_values(ascending=False)
    concentracion_top3 = (clientes_por_zona.head(3).sum() / len(clientes)) * 100
    
    fig_conc = px.bar(
        y=clientes_por_zona.values,
        x=[f"Zona {i}" for i in clientes_por_zona.index],
        title='Distribución de Clientes por Zona (Análisis de Concentración)',
        labels={'y': 'Cantidad de Clientes', 'x': 'Zona'},
        template='plotly_white',
        color=clientes_por_zona.values,
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_conc, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Concentración en Top 3 Zonas",
            value=f"{concentracion_top3:.1f}%",
            delta="del total de clientes"
        )
    
    with col2:
        num_zonas_67 = 0
        acum = 0
        for val in clientes_por_zona.values:
            acum += val
            num_zonas_67 += 1
            if acum >= len(clientes) * 0.67:
                break
        
        st.metric(
            label="Zonas para 67% de Clientes",
            value=num_zonas_67,
            delta=f"de {len(clientes_por_zona)}"
        )
    
    st.markdown("---")
    
    # Análisis de valor por zona
    st.markdown('<div class="subheader-custom">💰 Concentración de Valor (Saldos)</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Análisis similar pero enfocado en saldos en cuentas. Si la mayoría del valor
    del banco está en pocas zonas, hay riesgo concentrado.
    """)
    
    saldo_por_zona = clientes.groupby('Zona')['Saldo Promedio de Cuentas'].sum().sort_values(ascending=False)
    valor_top3 = (saldo_por_zona.head(3).sum() / saldo_por_zona.sum()) * 100
    
    fig_valor = px.pie(
        values=saldo_por_zona.values,
        names=[f"Zona {i}" for i in saldo_por_zona.index],
        title='Distribución de Saldo Total por Zona',
        labels={'Zona': 'Zona Geográfica'}
    )
    st.plotly_chart(fig_valor, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Valor en Top 3 Zonas",
            value=f"{valor_top3:.1f}%",
            delta="del saldo total"
        )
    
    with col2:
        st.metric(
            label="Saldo Promedio Total",
            value=f"${clientes['Saldo Promedio de Cuentas'].sum():,.0f}",
            delta="depósitos totales"
        )
    
    st.markdown("---")
    
    # Análisis de cobertura de sucursales
    st.markdown('<div class="subheader-custom">🏢 Riesgo de Cobertura de Sucursales</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Evalúa qué sucursales son críticas. Si una sucursal específica atiende
    muchos clientes o mucho valor, su pérdida sería catastrófica.
    """)
    
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales)
    
    clientes_por_sucursal = clientes.groupby('Índice_Sucursal_Cercana').agg({
        'Saldo Promedio de Cuentas': ['sum', 'mean', 'count']
    }).round(0)
    
    clientes_por_sucursal.columns = ['Saldo Total', 'Saldo Promedio', 'Cantidad Clientes']
    clientes_por_sucursal['Sucursal'] = [f"Sucursal {i+1}" for i in clientes_por_sucursal.index]
    
    fig_suc_riesgo = px.bar(
        clientes_por_sucursal,
        x='Sucursal',
        y='Cantidad Clientes',
        color='Saldo Total',
        title='Dependencia por Sucursal (Clientes y Valor)',
        labels={'Cantidad Clientes': 'Clientes', 'Saldo Total': 'Saldo Total ($)'},
        template='plotly_white'
    )
    st.plotly_chart(fig_suc_riesgo, use_container_width=True)
    
    st.markdown("---")
    
    # Matriz de riesgos
    st.markdown('<div class="subheader-custom">⚠️ Matriz de Riesgos Identificados</div>', 
                unsafe_allow_html=True)
    
    riesgos = []
    
    # Riesgo 1: Concentración de clientes
    if concentracion_top3 > 70:
        riesgos.append({
            'Riesgo': 'Alta Concentración Geográfica',
            'Nivel': '🔴 Crítico',
            'Descripción': f'{concentracion_top3:.0f}% de clientes en 3 zonas',
            'Recomendación': 'Diversificar geográficamente, expandir a nuevas zonas'
        })
    elif concentracion_top3 > 50:
        riesgos.append({
            'Riesgo': 'Concentración Moderada',
            'Nivel': '🟡 Mediano',
            'Descripción': f'{concentracion_top3:.0f}% de clientes en 3 zonas',
            'Recomendación': 'Monitorear y planear expansión a zonas diversas'
        })
    
    # Riesgo 2: Valor concentrado
    if valor_top3 > 80:
        riesgos.append({
            'Riesgo': 'Saldo Concentrado en Pocas Zonas',
            'Nivel': '🔴 Crítico',
            'Descripción': f'{valor_top3:.0f}% del valor en 3 zonas',
            'Recomendación': 'Evaluar diversificación de cartera y riesgos sistémicos regionales'
        })
    
    # Riesgo 3: Dependencia de sucursal
    carga_max = clientes_por_sucursal['Cantidad Clientes'].max()
    carga_prom = clientes_por_sucursal['Cantidad Clientes'].mean()
    
    if carga_max > carga_prom * 2:
        riesgos.append({
            'Riesgo': 'Sucursal Crítica Sobrecargada',
            'Nivel': '🟡 Mediano',
            'Descripción': f'Una sucursal maneja {(carga_max/carga_prom):.1f}x el promedio',
            'Recomendación': 'Aumentar recursos o crear sucursal satélite en zona'
        })
    
    if len(riesgos) > 0:
        for i, riesgo in enumerate(riesgos, 1):
            st.warning(f"""
            **{i}. {riesgo['Riesgo']}** {riesgo['Nivel']}
            
            - Situación: {riesgo['Descripción']}
            - Acción Recomendada: {riesgo['Recomendación']}
            """)
    else:
        st.success("""
        **✅ Perfil de Riesgo Saludable**
        
        No se identificaron riesgos geoespaciales críticos. La distribución de 
        clientes y saldos parece suficientemente diversificada.
        """)
    
    st.markdown("---")
    
    # Plan de mitigación
    st.markdown('<div class="subheader-custom">🛡️ Plan de Mitigación de Riesgos</div>', 
                unsafe_allow_html=True)
    
    st.info("""
    **Acciones Recomendadas:**
    
    1. **Monitoreo Continuo:** Revisar esta página mensualmente para detectar cambios
    
    2. **Diversificación:** Enfoque en atraer clientes de nuevas zonas geográficas
    
    3. **Redundancia:** Asegurar que infraestructura crítica no dependa de una sola sucursal
    
    4. **Contingencias:** Tener planes de continuidad si una zona experimenta crisis económica
    
    5. **Expansión Estratégica:** Invertir en nuevas sucursales en zonas actualmente desatendidas
    """)