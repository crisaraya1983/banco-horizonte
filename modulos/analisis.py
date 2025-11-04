import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

from modulos.carga_datos import (
    cargar_sucursales, cargar_cajeros, cargar_clientes, cargar_productos,
    obtener_datos_consolidados
)
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

from modulos.componentes import (
    crear_seccion_encabezado,
    crear_tarjeta_metrica,
    crear_tarjeta_informativa,
    crear_panel_estadisticas,
    crear_indicador_estado,
    crear_linea_separadora
)

from modulos.visualizaciones import (
    crear_mapa_sucursales_cajeros,
    crear_mapa_cobertura_clientes,
    crear_mapa_cobertura_con_radios,
    crear_grafico_concentracion_clientes,
    crear_grafico_transacciones_por_ubicacion,
    crear_grafico_comparativa_cobertura_clientes,
    crear_grafico_volumen_transacciones,
    crear_grafico_empleados_vs_transacciones,
    crear_grafico_productos_por_ubicacion,
    crear_grafico_saldo_promedio_por_producto,
    crear_grafico_frecuencia_visitas,
    crear_grafico_transacciones_cajeros,
    crear_grafico_matriz_distancias
)


# PÁGINA 1: ANÁLISIS DE COBERTURA GEOGRÁFICA

def pagina_analisis_cobertura():
    """
    Página principal de análisis de cobertura geográfica.
    Analiza la cobertura de sucursales y cajeros automáticos en el territorio.
    """
    
    # Cargar datos consolidados
    datos_consolidados = obtener_datos_consolidados()
    
    # Obtener ubicaciones únicas de sucursales
    ubicaciones_sucursales = datos_consolidados[
        ['Nombre', 'Latitud', 'Longitud', 'Tipo de Sucursal', 
         'Número de Empleados', 'Volumen_Transacciones_Sucursal']
    ].drop_duplicates(subset=['Nombre']).reset_index(drop=True)
    
    
    # SECCIÓN 1: MAPA DE COBERTURA CON RADIOS
    
    crear_seccion_encabezado(
        titulo="Mapa de Cobertura de Sucursales y Cajeros"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        distancia_cobertura = st.slider(
            "Radio de cobertura (km)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help="Selecciona el radio de cobertura para cada sucursal"
        )
    
    with col2:
        st.info(
            f"📍 El mapa muestra {len(ubicaciones_sucursales)} sucursales con un radio "
            f"de cobertura de {distancia_cobertura} km"
        )
    
    try:
        mapa_cobertura = crear_mapa_cobertura_con_radios(
            ubicaciones_sucursales,
            distancia_km=distancia_cobertura
        )
        st.components.v1.html(mapa_cobertura._repr_html_(), height=600, width=None)
    except Exception as e:
        st.error(f"Error al generar el mapa: {e}")
    
    st.divider()
    
    # SECCIÓN 2: ANÁLISIS DE CONCENTRACIÓN DE CLIENTES
    
    crear_seccion_encabezado(
        titulo="Análisis de Concentración de Clientes",
        descripcion="Distribución de clientes por sucursal"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_concentracion = crear_grafico_concentracion_clientes(datos_consolidados)
            st.plotly_chart(fig_concentracion, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de concentración: {e}")
    
    with col2:
        # Estadísticas de concentración
        concentracion_stats = datos_consolidados.groupby('Nombre').agg({
            'Numero_Clientes_Producto': 'sum'
        }).reset_index()
        concentracion_stats.columns = ['Sucursal', 'Total_Clientes']
        concentracion_stats = concentracion_stats.sort_values('Total_Clientes', ascending=False)
        
        st.metric(
            label="Sucursal con Mayor Concentración",
            value=concentracion_stats.iloc[0]['Sucursal'],
            delta=f"{int(concentracion_stats.iloc[0]['Total_Clientes'])} clientes"
        )
        
        st.metric(
            label="Total de Clientes",
            value=f"{int(concentracion_stats['Total_Clientes'].sum()):,}"
        )
        
        st.metric(
            label="Promedio por Sucursal",
            value=f"{int(concentracion_stats['Total_Clientes'].mean()):,}"
        )
        
        # Tabla de estadísticas
        st.markdown("**Detalle por Sucursal**")
        st.dataframe(concentracion_stats, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # SECCIÓN 3: ANÁLISIS DE VOLUMEN DE TRANSACCIONES
    
    crear_seccion_encabezado(
        titulo="Volumen de Transacciones por Ubicación",
        descripcion="Análisis del movimiento transaccional en cada sucursal"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_transacciones = crear_grafico_transacciones_por_ubicacion(datos_consolidados)
            st.plotly_chart(fig_transacciones, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de transacciones: {e}")
    
    with col2:
        # Estadísticas de transacciones
        transacciones_stats = datos_consolidados.groupby('Nombre').agg({
            'Volumen_Transacciones_Sucursal': 'first',
            'Volumen_Transacciones_Cajero_Diarias': 'first'
        }).reset_index()
        
        transacciones_stats['Transacciones_Cajero_Mensuales'] = (
            transacciones_stats['Volumen_Transacciones_Cajero_Diarias'] * 22
        )
        transacciones_stats['Total_Transacciones'] = (
            transacciones_stats['Volumen_Transacciones_Sucursal'] + 
            transacciones_stats['Transacciones_Cajero_Mensuales']
        )
        
        transacciones_stats = transacciones_stats.sort_values(
            'Total_Transacciones', ascending=False
        )
        
        st.metric(
            label="Sucursal con Mayor Volumen",
            value=transacciones_stats.iloc[0]['Nombre'],
            delta=f"{int(transacciones_stats.iloc[0]['Total_Transacciones']):,} transacciones/mes"
        )
        
        st.metric(
            label="Total Transacciones/mes",
            value=f"{int(transacciones_stats['Total_Transacciones'].sum()):,}"
        )
        
        st.metric(
            label="Promedio por Sucursal",
            value=f"{int(transacciones_stats['Total_Transacciones'].mean()):,}"
        )
        
        # Tabla de estadísticas
        st.markdown("**Detalle de Transacciones**")
        tabla_transacciones = transacciones_stats[[
            'Nombre', 
            'Volumen_Transacciones_Sucursal',
            'Transacciones_Cajero_Mensuales',
            'Total_Transacciones'
        ]].copy()
        tabla_transacciones.columns = [
            'Sucursal',
            'Sucursal/mes',
            'Cajero/mes',
            'Total/mes'
        ]
        st.dataframe(tabla_transacciones, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # SECCIÓN 4: COMPARATIVA COBERTURA VS CLIENTES
    
    crear_seccion_encabezado(
        titulo="Análisis: Cobertura vs Concentración de Clientes",
        descripcion="Relación entre transacciones y clientes por sucursal"
    )
    
    try:
        fig_comparativa = crear_grafico_comparativa_cobertura_clientes(
            datos_consolidados, 
            distancia_km=distancia_cobertura
        )
        st.plotly_chart(fig_comparativa, use_container_width=True)
    except Exception as e:
        st.error(f"Error en gráfico de comparativa: {e}")



# PÁGINA 2: SEGMENTACIÓN GEOGRÁFICA

def pagina_segmentacion_geografica():
    """
    Segmentación geográfica de clientes y análisis de productos por región.
    """
    
    crear_seccion_encabezado(titulo="Segmentación Geográfica")
    
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    
    num_zonas = st.slider(
        "Número de zonas",
        min_value=2, max_value=5, value=3
    )
    
    coords = clientes[['Latitud', 'Longitud']].values
    kmeans = KMeans(n_clusters=num_zonas, random_state=42, n_init=10)
    clientes['Zona'] = kmeans.fit_predict(coords) + 1
    
    crear_seccion_encabezado(titulo="Características por Zona")
    
    resumen_zonas = []
    for zona in range(1, num_zonas + 1):
        clientes_zona = clientes[clientes['Zona'] == zona]
        if len(clientes_zona) > 0:
            resumen_zonas.append({
                'Zona': f"Zona {zona}",
                'Clientes': len(clientes_zona),
                'Saldo Promedio': f"${clientes_zona['Saldo Promedio de Cuentas'].mean():,.0f}",
                'Visitas/mes': f"{clientes_zona['Frecuencia de Visitas'].mean():.1f}",
                'Transacciones': int(clientes_zona['Volumen de Transacciones'].mean()),
                'Producto Preferido': clientes_zona['Productos Financieros Adquiridos'].mode()[0] if len(clientes_zona) > 0 else 'N/A'
            })
    
    resumen_df = pd.DataFrame(resumen_zonas)
    st.dataframe(resumen_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Mapa de segmentación
    crear_seccion_encabezado(titulo="Distribución Espacial de Zonas")
    
    try:
        import folium
        
        centro_lat = clientes['Latitud'].mean()
        centro_lon = clientes['Longitud'].mean()
        mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=7)
        
        colores = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
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
                    popup=f"Zona {zona}"
                ).add_to(mapa)
            
            centro = clientes_zona[['Latitud', 'Longitud']].mean()
            folium.Marker(
                location=[centro['Latitud'], centro['Longitud']],
                popup=f"Centro Zona {zona}",
                icon=folium.Icon(color='gray', icon='info-sign')
            ).add_to(mapa)
        
        st.components.v1.html(mapa._repr_html_(), height=600, width=None)
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.divider()
    
    # Gráficos de análisis
    crear_seccion_encabezado(titulo="Análisis de Productos y Valor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        productos_zona = clientes.groupby(['Zona', 'Productos Financieros Adquiridos']).size().reset_index(name='Cantidad')
        productos_zona['Zona'] = 'Zona ' + productos_zona['Zona'].astype(str)
        
        fig_productos = px.bar(
            productos_zona,
            x='Zona',
            y='Cantidad',
            color='Productos Financieros Adquiridos',
            title='Distribución de Productos por Zona',
            barmode='stack',
            template='plotly_white'
        )
        st.plotly_chart(fig_productos, use_container_width=True)
    
    with col2:
        saldo_zona = []
        for zona in range(1, num_zonas + 1):
            clientes_zona = clientes[clientes['Zona'] == zona]
            saldo_zona.append({
                'Zona': f'Zona {zona}',
                'Saldo': clientes_zona['Saldo Promedio de Cuentas'].mean()
            })
        
        saldo_df = pd.DataFrame(saldo_zona)
        
        fig_saldo = px.bar(
            saldo_df,
            x='Zona',
            y='Saldo',
            title='Saldo Promedio por Zona',
            template='plotly_white',
            color='Zona'
        )
        fig_saldo.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
        st.plotly_chart(fig_saldo, use_container_width=True)


# OPTIMIZACIÓN LOGÍSTICA

def pagina_optimizacion_logistica():
    """
    Optimización de rutas de mantenimiento para cajeros automáticos.
    """
    
    crear_seccion_encabezado(titulo="Optimización Logística")
    
    cajeros = cargar_cajeros()
    
    crear_seccion_encabezado(titulo="Resumen de Cajeros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Cajeros", len(cajeros))
    
    with col2:
        transacciones_total = cajeros['Volumen de Transacciones Diarias'].sum()
        st.metric("Transacciones/día", int(transacciones_total))
    
    with col3:
        dispersión = cajeros['Latitud'].std() + cajeros['Longitud'].std()
        st.metric("Dispersión Geográfica", f"{dispersión:.2f}")
    
    st.divider()
    
    # Matriz de distancias
    crear_seccion_encabezado(titulo="Matriz de Distancias")
    
    matriz_dist = crear_matriz_distancias(cajeros)
    fig_matriz = crear_grafico_matriz_distancias(
        matriz_dist,
        etiquetas=[f"Cajero {i+1}" for i in range(len(cajeros))]
    )
    st.plotly_chart(fig_matriz, use_container_width=True)
    
    st.divider()
    
    # Carga de trabajo
    crear_seccion_encabezado(titulo="Carga de Trabajo por Cajero")
    
    fig_cajeros = crear_grafico_transacciones_cajeros(cajeros)
    st.plotly_chart(fig_cajeros, use_container_width=True)
    
    st.divider()
    
    # Ruta propuesta
    crear_seccion_encabezado(titulo="Ruta Óptima Propuesta")
    
    visitados = set()
    posicion_actual = 0
    ruta = [0]
    distancia_total = 0
    
    while len(visitados) < len(cajeros):
        visitados.add(posicion_actual)
        distancias_candidatos = []
        for j in range(len(cajeros)):
            if j not in visitados:
                distancias_candidatos.append((matriz_dist[posicion_actual, j], j))
        
        if distancias_candidatos:
            dist_min, siguiente = min(distancias_candidatos)
            distancia_total += dist_min
            ruta.append(siguiente)
            posicion_actual = siguiente
    
    distancia_total += matriz_dist[ruta[-1], ruta[0]]
    
    ruta_tabla = []
    for i, idx_cajero in enumerate(ruta):
        ruta_tabla.append({
            'Orden': i + 1,
            'Cajero': f"Cajero {idx_cajero + 1}",
            'Latitud': f"{cajeros.iloc[idx_cajero]['Latitud']:.4f}",
            'Longitud': f"{cajeros.iloc[idx_cajero]['Longitud']:.4f}",
            'Transacciones/día': int(cajeros.iloc[idx_cajero]['Volumen de Transacciones Diarias'])
        })
    
    st.dataframe(pd.DataFrame(ruta_tabla), use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Distancia Total", f"{distancia_total:.2f} km")
    with col2:
        st.metric("Distancia Promedio", f"{distancia_total / len(cajeros):.2f} km")


# PÁGINA 4: MARKETING DIRIGIDO

def pagina_marketing_dirigido():
    """
    Análisis de marketing dirigido por geolocalización.
    """
    
    crear_seccion_encabezado(titulo="Marketing Dirigido por Geolocalización")
    
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales)
    
    crear_seccion_encabezado(titulo="Análisis de Valor de Clientes")
    
    clientes['Segmento_Valor'] = pd.cut(
        clientes['Saldo Promedio de Cuentas'],
        bins=[0, 3000, 5000, float('inf')],
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    seg_stats = clientes.groupby('Segmento_Valor').agg({
        'Saldo Promedio de Cuentas': 'mean',
        'Frecuencia de Visitas': 'mean',
        'Volumen de Transacciones': 'mean'
    }).round(0)
    
    st.dataframe(seg_stats, use_container_width=True)
    
    st.divider()
    
    crear_seccion_encabezado(titulo="Distribución de Clientes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        seg_dist = clientes['Segmento_Valor'].value_counts()
        fig_seg = px.pie(
            values=seg_dist.values,
            names=seg_dist.index,
            title='Clientes por Segmento de Valor',
            color_discrete_map={'Bajo': '#FF6B6B', 'Medio': '#FFA500', 'Alto': '#4ECDC4'}
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    
    with col2:
        prod_seg = pd.crosstab(clientes['Segmento_Valor'], clientes['Productos Financieros Adquiridos'])
        fig_prod_seg = px.bar(
            prod_seg,
            barmode='group',
            title='Productos por Segmento',
            template='plotly_white'
        )
        st.plotly_chart(fig_prod_seg, use_container_width=True)
    
    st.divider()
    
    crear_seccion_encabezado(titulo="Proximidad vs Valor de Cliente")
    
    fig_prox_valor = px.scatter(
        clientes,
        x='Distancia_a_Sucursal_km',
        y='Saldo Promedio de Cuentas',
        color='Productos Financieros Adquiridos',
        size='Volumen de Transacciones',
        title='Distancia a Sucursal vs Valor de Cliente',
        labels={
            'Distancia_a_Sucursal_km': 'Distancia (km)',
            'Saldo Promedio de Cuentas': 'Saldo Promedio ($)'
        },
        template='plotly_white'
    )
    st.plotly_chart(fig_prox_valor, use_container_width=True)


# PREDICCIÓN DE DEMANDA

def pagina_prediccion_demanda():
    """
    Predicción de demanda basada en factores geoespaciales.
    """
    
    crear_seccion_encabezado(titulo="Predicción de Demanda")
    
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales)
    clientes = calcular_distancia_a_cajero_mas_cercano(clientes, cajeros)
    
    crear_seccion_encabezado(titulo="Modelo Predictivo")
    
    X = clientes[[
        'Distancia_a_Sucursal_km',
        'Distancia_a_Cajero_km',
        'Frecuencia de Visitas',
        'Saldo Promedio de Cuentas'
    ]].values
    
    y = clientes['Volumen de Transacciones'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)
    score = modelo.score(X_scaled, y)
    
    st.metric("R² Score del Modelo", f"{score:.3f}")
    
    st.divider()
    
    crear_seccion_encabezado(titulo="Importancia de Factores")
    
    feature_names = ['Dist. Sucursal', 'Dist. Cajero', 'Frecuencia Visitas', 'Saldo Promedio']
    importancia = np.abs(modelo.coef_)
    importancia_norm = importancia / importancia.sum()
    
    fig_imp = px.bar(
        x=feature_names,
        y=importancia_norm,
        title='Importancia de Factores',
        labels={'x': 'Factor', 'y': 'Importancia Relativa'},
        template='plotly_white'
    )
    fig_imp.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.divider()
    
    crear_seccion_encabezado(titulo="Predicción para Nueva Ubicación")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dist_suc = st.number_input("Dist. a Sucursal (km)", value=7.0)
    with col2:
        dist_caj = st.number_input("Dist. a Cajero (km)", value=3.0)
    with col3:
        freq_vis = st.number_input("Frecuencia Visitas (veces/mes)", value=3.0)
    with col4:
        saldo_prom = st.number_input("Saldo Promedio ($)", value=5000.0)
    
    X_nuevo = np.array([[dist_suc, dist_caj, freq_vis, saldo_prom]])
    X_nuevo_scaled = scaler.transform(X_nuevo)
    prediccion = modelo.predict(X_nuevo_scaled)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Transacciones Predichas", f"{int(max(0, prediccion))}")
    
    with col2:
        valor_potencial = prediccion * 20
        st.metric("Ingreso Estimado Anual", f"${max(0, valor_potencial):,.0f}")


# ANÁLISIS DE RIESGOS

def pagina_analisis_riesgos():
    """
    Análisis de riesgos geoespaciales.
    """
    
    crear_seccion_encabezado(titulo="Análisis de Riesgos")
    
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    
    coords = clientes[['Latitud', 'Longitud']].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clientes['Zona'] = kmeans.fit_predict(coords)
    
    crear_seccion_encabezado(titulo="Concentración de Clientes")
    
    clientes_por_zona = clientes['Zona'].value_counts().sort_values(ascending=False)
    concentracion_top3 = (clientes_por_zona.head(3).sum() / len(clientes)) * 100
    
    fig_conc = px.bar(
        y=clientes_por_zona.values,
        x=[f"Zona {i}" for i in clientes_por_zona.index],
        title='Distribución de Clientes por Zona',
        labels={'y': 'Clientes', 'x': 'Zona'},
        template='plotly_white',
        color=clientes_por_zona.values,
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_conc, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Concentración Top 3 Zonas", f"{concentracion_top3:.1f}%")
    
    with col2:
        num_zonas_67 = 0
        acum = 0
        for val in clientes_por_zona.values:
            acum += val
            num_zonas_67 += 1
            if acum >= len(clientes) * 0.67:
                break
        st.metric("Zonas para 67% Clientes", num_zonas_67)
    
    st.divider()
    
    crear_seccion_encabezado(titulo="Concentración de Valor")
    
    saldo_por_zona = clientes.groupby('Zona')['Saldo Promedio de Cuentas'].sum().sort_values(ascending=False)
    valor_top3 = (saldo_por_zona.head(3).sum() / saldo_por_zona.sum()) * 100
    
    fig_valor = px.pie(
        values=saldo_por_zona.values,
        names=[f"Zona {i}" for i in saldo_por_zona.index],
        title='Distribución de Saldo Total por Zona'
    )
    st.plotly_chart(fig_valor, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Valor en Top 3 Zonas", f"{valor_top3:.1f}%")
    
    with col2:
        st.metric("Saldo Total", f"${clientes['Saldo Promedio de Cuentas'].sum():,.0f}")
    
    st.divider()
    
    crear_seccion_encabezado(titulo="Dependencia por Sucursal")
    
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales)
    
    clientes_por_sucursal = clientes.groupby('Índice_Sucursal_Cercana').agg({
        'Saldo Promedio de Cuentas': ['sum', 'mean', 'count']
    }).round(0)
    
    clientes_por_sucursal.columns = ['Saldo Total', 'Saldo Promedio', 'Cantidad']
    clientes_por_sucursal['Sucursal'] = [f"Sucursal {i+1}" for i in clientes_por_sucursal.index]
    
    fig_suc_riesgo = px.bar(
        clientes_por_sucursal,
        x='Sucursal',
        y='Cantidad',
        color='Saldo Total',
        title='Clientes y Valor por Sucursal',
        labels={'Cantidad': 'Clientes'},
        template='plotly_white'
    )
    st.plotly_chart(fig_suc_riesgo, use_container_width=True)