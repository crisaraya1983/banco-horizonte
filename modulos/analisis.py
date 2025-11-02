import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

from modulos.carga_datos import (
    cargar_sucursales, cargar_cajeros, cargar_clientes, cargar_productos
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


# ANÁLISIS DE COBERTURA GEOGRÁFICA

def pagina_analisis_cobertura():
    """
    Análisis de cobertura geográfica de sucursales y cajeros automáticos.
    """
    
    st.markdown('<div class="main-header">Análisis de Cobertura Geográfica</div>', 
                unsafe_allow_html=True)
    
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    productos = cargar_productos()
    
    # Controles
    col1, col2 = st.columns(2)
    
    with col1:
        umbral_sucursal = st.slider(
            "Distancia máxima a Sucursal (km)",
            min_value=1.0, max_value=20.0, value=10.0, step=0.5
        )
    
    with col2:
        umbral_cajero = st.slider(
            "Distancia máxima a Cajero (km)",
            min_value=1.0, max_value=15.0, value=5.0, step=0.5
        )
    
    cobertura = calcular_cobertura_geográfica(
        clientes, cajeros, sucursales,
        umbral_sucursal=umbral_sucursal,
        umbral_cajero=umbral_cajero
    )
    
    # Métricas principales
    st.markdown('<div class="section-header">Cobertura General</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cobertura Sucursales", f"{cobertura['cobertura_sucursales_pct']:.1f}%")
    
    with col2:
        st.metric("Cobertura Cajeros", f"{cobertura['cobertura_cajeros_pct']:.1f}%")
    
    with col3:
        st.metric("Cobertura Completa", f"{cobertura['cobertura_completa_pct']:.1f}%")
    
    with col4:
        st.metric("Clientes Desatendidos", cobertura['clientes_sin_cobertura_completa'])
    
    st.divider()
    
    # Mapas interactivos
    st.markdown('<div class="section-header">Visualización Geoespacial</div>', 
                unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Distribución de Servicios", "Estado de Cobertura"])
    
    with tab1:
        try:
            mapa1 = crear_mapa_sucursales_cajeros(sucursales, cajeros, clientes)
            st.components.v1.html(mapa1._repr_html_(), height=600, width=None)
        except Exception as e:
            st.error(f"Error: {e}")
    
    with tab2:
        try:
            mapa2 = crear_mapa_cobertura_clientes(
                clientes, sucursales, cajeros,
                umbral_sucursal=umbral_sucursal,
                umbral_cajero=umbral_cajero
            )
            st.components.v1.html(mapa2._repr_html_(), height=600, width=None)
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.divider()
    
    # Zonas desatendidas
    st.markdown('<div class="section-header">Zonas Desatendidas</div>', 
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
        st.metric("Clientes Desatendidos", len(desatendidos))
    
    with col2:
        porcentaje = (len(desatendidos) / len(clientes)) * 100 if len(clientes) > 0 else 0
        st.metric("Porcentaje", f"{porcentaje:.1f}%")
    
    with col3:
        distancia_prom = desatendidos['Distancia_a_Sucursal_km'].mean() if len(desatendidos) > 0 else 0
        st.metric("Distancia Promedio", f"{distancia_prom:.2f} km")
    
    if len(desatendidos) > 0:
        st.markdown("**Clientes en Zonas Desatendidas**")
        tabla_mostrar = desatendidos[[
            'Ubicación de Residencia',
            'Productos Financieros Adquiridos',
            'Distancia_a_Sucursal_km',
            'Saldo Promedio de Cuentas'
        ]].copy()
        tabla_mostrar.columns = ['Ubicación', 'Producto', 'Distancia (km)', 'Saldo']
        st.dataframe(tabla_mostrar, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Análisis de volumen
    st.markdown('<div class="section-header">Análisis de Volumen</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_vol = crear_grafico_volumen_transacciones(sucursales)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        fig_emp = crear_grafico_empleados_vs_transacciones(sucursales)
        st.plotly_chart(fig_emp, use_container_width=True)


# PÁGINA 2: SEGMENTACIÓN GEOGRÁFICA

def pagina_segmentacion_geografica():
    """
    Segmentación geográfica de clientes y análisis de productos por región.
    """
    
    st.markdown('<div class="main-header">Segmentación Geográfica</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="section-header">Características por Zona</div>', 
                unsafe_allow_html=True)
    
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
    st.markdown('<div class="section-header">Distribución Espacial de Zonas</div>', 
                unsafe_allow_html=True)
    
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
    st.markdown('<div class="section-header">Análisis de Productos y Valor</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="main-header">Optimización Logística</div>', 
                unsafe_allow_html=True)
    
    cajeros = cargar_cajeros()
    
    st.markdown('<div class="section-header">Resumen de Cajeros</div>', 
                unsafe_allow_html=True)
    
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
    st.markdown('<div class="section-header">Matriz de Distancias</div>', 
                unsafe_allow_html=True)
    
    matriz_dist = crear_matriz_distancias(cajeros)
    fig_matriz = crear_grafico_matriz_distancias(
        matriz_dist,
        etiquetas=[f"Cajero {i+1}" for i in range(len(cajeros))]
    )
    st.plotly_chart(fig_matriz, use_container_width=True)
    
    st.divider()
    
    # Carga de trabajo
    st.markdown('<div class="section-header">Carga de Trabajo por Cajero</div>', 
                unsafe_allow_html=True)
    
    fig_cajeros = crear_grafico_transacciones_cajeros(cajeros)
    st.plotly_chart(fig_cajeros, use_container_width=True)
    
    st.divider()
    
    # Ruta propuesta
    st.markdown('<div class="section-header">Ruta Óptima Propuesta</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="main-header">Marketing Dirigido por Geolocalización</div>', 
                unsafe_allow_html=True)
    
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales)
    
    st.markdown('<div class="section-header">Análisis de Valor de Clientes</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="section-header">Distribución de Clientes</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="section-header">Proximidad vs Valor de Cliente</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="main-header">Predicción de Demanda</div>', 
                unsafe_allow_html=True)
    
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales)
    clientes = calcular_distancia_a_cajero_mas_cercano(clientes, cajeros)
    
    st.markdown('<div class="section-header">Modelo Predictivo</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="section-header">Importancia de Factores</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="section-header">Predicción para Nueva Ubicación</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="main-header">Análisis de Riesgos</div>', 
                unsafe_allow_html=True)
    
    clientes = cargar_clientes()
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    
    coords = clientes[['Latitud', 'Longitud']].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clientes['Zona'] = kmeans.fit_predict(coords)
    
    st.markdown('<div class="section-header">Concentración de Clientes</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="section-header">Concentración de Valor</div>', 
                unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="section-header">Dependencia por Sucursal</div>', 
                unsafe_allow_html=True)
    
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