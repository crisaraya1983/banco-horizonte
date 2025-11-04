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
    distancia_haversine,
    calcular_rutas_mantenimiento  
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
    crear_grafico_matriz_distancias,
    crear_grafico_transacciones_cajeros_por_tipo,
    crear_mapa_segmentacion_geografica,
    crear_mapa_rutas_mantenimiento  
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

    if 'map_zoom_level' not in st.session_state:
        st.session_state.map_zoom_level = 7
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        distancia_cobertura = st.slider(
            "Radio de cobertura (km)",
            min_value=0,
            max_value=30,
            value=10,
            step=5,
            help="Selecciona el radio de cobertura para cada sucursal"
        )
        zoom_level = st.session_state.get('map_zoom_level', 7)
    
    with col2:
        st.info(
            f"📍 El mapa muestra {len(ubicaciones_sucursales)} sucursales con un radio "
            f"de cobertura de {distancia_cobertura} km"
        )
    
    try:
        mapa_cobertura = crear_mapa_cobertura_con_radios(
            ubicaciones_sucursales,
            distancia_km=distancia_cobertura,
            zoom_level=zoom_level
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
        descripcion="Análisis del movimiento transaccional en sucursales y cajeros"
    )

    # DOS COLUMNAS PARA LOS DOS GRÁFICOS
    col1, col2 = st.columns(2)

    # GRÁFICO 1: TRANSACCIONES DE SUCURSALES
    with col1:
        try:
            fig_transacciones_sucursal = crear_grafico_transacciones_por_ubicacion(datos_consolidados)
            st.plotly_chart(fig_transacciones_sucursal, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de transacciones de sucursal: {e}")

    # GRÁFICO 2: TRANSACCIONES DE CAJEROS POR TIPO
    with col2:
        try:
            fig_transacciones_cajeros = crear_grafico_transacciones_cajeros_por_tipo(datos_consolidados)
            st.plotly_chart(fig_transacciones_cajeros, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de transacciones de cajeros: {e}")

    st.divider()

    # ESTADÍSTICAS (meter en una sola columna debajo)
    crear_seccion_encabezado(
        titulo="Detalle de Transacciones por Ubicación"
    )

    # Crear tabla consolidada
    transacciones_stats = datos_consolidados.groupby('Nombre').agg({
        'Volumen_Transacciones_Sucursal': 'first',
        'Volumen_Transacciones_Cajero_Diarias': 'first'
    }).reset_index()

    transacciones_stats['Transacciones_Cajero_Mensuales'] = (
        transacciones_stats['Volumen_Transacciones_Cajero_Diarias'] * 30
    )

    transacciones_stats = transacciones_stats.sort_values(
        'Volumen_Transacciones_Sucursal', ascending=False
    )

    tabla_transacciones = transacciones_stats[[
        'Nombre', 
        'Volumen_Transacciones_Sucursal',
        'Volumen_Transacciones_Cajero_Diarias',
        'Transacciones_Cajero_Mensuales'
    ]].copy()

    tabla_transacciones.columns = [
        'Ubicación',
        'Sucursal/mes',
        'Cajero/día',
        'Cajero/mes'
    ]

    st.dataframe(tabla_transacciones, use_container_width=True, hide_index=True)

    # MÉTRICAS LADO A LADO
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Sucursal/mes",
            value=f"{int(transacciones_stats['Volumen_Transacciones_Sucursal'].sum()):,}"
        )

    with col2:
        st.metric(
            label="Total Cajero/mes",
            value=f"{int(transacciones_stats['Transacciones_Cajero_Mensuales'].sum()):,}"
        )

    with col3:
        total_general = (
            transacciones_stats['Volumen_Transacciones_Sucursal'].sum() + 
            transacciones_stats['Transacciones_Cajero_Mensuales'].sum()
        )
        st.metric(
            label="Total General/mes",
            value=f"{int(total_general):,}"
        )

    with col4:
        porcentaje_cajero = (
            transacciones_stats['Transacciones_Cajero_Mensuales'].sum() / total_general * 100
        )
        st.metric(
            label="% Transacciones Cajero",
            value=f"{porcentaje_cajero:.1f}%"
        )
    
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
    
    # Cargar datos consolidados
    datos_consolidados = obtener_datos_consolidados()
    
    # SECCIÓN 1: MAPA DE SEGMENTACIÓN
    crear_seccion_encabezado(
        titulo="Mapa de Segmentación: Clientes vs. Transacciones por Sucursal"
    )
    
    try:
        mapa_seg, datos_sucursales = crear_mapa_segmentacion_geografica(datos_consolidados)
        st.components.v1.html(mapa_seg._repr_html_(), height=700, width=None)
    except Exception as e:
        st.error(f"Error al generar el mapa: {e}")
    
    st.divider()
    
    # SECCIÓN 2: ANÁLISIS DE EFICIENCIA
    crear_seccion_encabezado(
        titulo="Análisis de Eficiencia Operativa por Sucursal"
    )
    
    # Tabla de estadísticas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Detalle de Sucursales**")
        
        # Crear tabla formateada
        tabla_eficiencia = datos_sucursales[[
            'Nombre', 
            'Tipo de Sucursal',
            'Numero_Clientes_Producto',
            'Volumen_Transacciones_Sucursal',
            'Número de Empleados',
            'Clientes_por_Empleado',
            'Transacciones_por_Cliente'
        ]].copy()
        
        tabla_eficiencia.columns = [
            'Sucursal',
            'Tipo',
            'Clientes',
            'Trans/mes',
            'Empleados',
            'Clientes/Emp',
            'Trans/Cliente'
        ]
        
        tabla_eficiencia = tabla_eficiencia.sort_values('Clientes', ascending=False)
        
        st.dataframe(tabla_eficiencia, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Métricas Globales**")
        
        total_clientes = datos_sucursales['Numero_Clientes_Producto'].sum()
        total_transacciones = datos_sucursales['Volumen_Transacciones_Sucursal'].sum()
        total_empleados = datos_sucursales['Número de Empleados'].sum()
        
        st.metric("Total Clientes", f"{int(total_clientes):,}")
        st.metric("Total Trans/mes", f"{int(total_transacciones):,}")
        st.metric("Total Empleados", f"{int(total_empleados)}")
        st.metric("Trans/Cliente Promedio", 
                 f"{(total_transacciones / total_clientes):.2f}")
    
    st.divider()
    
    # SECCIÓN 3: IDENTIFICACIÓN DE ZONAS ESTRATÉGICAS
    crear_seccion_encabezado(
        titulo="Identificación de Zonas Estratégicas"
    )
    
    # Calcular cuartiles para segmentación
    q_clientes_75 = datos_sucursales['Numero_Clientes_Producto'].quantile(0.75)
    q_transacciones_75 = datos_sucursales['Volumen_Transacciones_Sucursal'].quantile(0.75)
    q_clientes_25 = datos_sucursales['Numero_Clientes_Producto'].quantile(0.25)
    q_transacciones_25 = datos_sucursales['Volumen_Transacciones_Sucursal'].quantile(0.25)
    
    # Categorizar sucursales
    def categorizar_sucursal(row):
        clientes = row['Numero_Clientes_Producto']
        transacciones = row['Volumen_Transacciones_Sucursal']
        
        if clientes >= q_clientes_75 and transacciones >= q_transacciones_75:
            return "🔥 Alto Rendimiento"
        elif clientes >= q_clientes_75 and transacciones < q_transacciones_75:
            return "⚡ Alta Demanda - Baja Transaccionalidad"
        elif clientes < q_clientes_25 and transacciones < q_transacciones_25:
            return "❌ Baja Actividad"
        elif clientes >= q_clientes_25 and transacciones >= q_transacciones_25:
            return "✅ Rendimiento Normal"
        else:
            return "⚠️ Requiere Análisis"
    
    datos_sucursales['Categoría'] = datos_sucursales.apply(categorizar_sucursal, axis=1)
    
    tabs = st.tabs([cat for cat in datos_sucursales['Categoría'].unique()])
    
    for tab, categoria in zip(tabs, datos_sucursales['Categoría'].unique()):
        with tab:
            datos_categoria = datos_sucursales[datos_sucursales['Categoría'] == categoria]
            
            if len(datos_categoria) > 0:
                st.dataframe(
                    datos_categoria[[
                        'Nombre', 'Numero_Clientes_Producto',
                        'Volumen_Transacciones_Sucursal', 'Clientes_por_Empleado'
                    ]].sort_values('Numero_Clientes_Producto', ascending=False)
                    .rename(columns={
                        'Nombre': 'Sucursal',
                        'Numero_Clientes_Producto': 'Cantidad_Clientes',
                        'Volumen_Transacciones_Sucursal': 'Trans/mes',
                        'Clientes_por_Empleado': 'Clientes/Emp'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No hay sucursales en esta categoría")



# OPTIMIZACIÓN LOGÍSTICA

def pagina_optimizacion_logistica():

    crear_seccion_encabezado(
        titulo="Optimización Logística de Cajeros Automáticos",
        descripcion="Análisis de distribución y rutas óptimas para mantenimiento y reposición"
    )
    
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    
    # SECCIÓN 1: Resumen de Cajeros
    crear_seccion_encabezado(titulo="Resumen de Cajeros Automáticos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Cajeros", len(cajeros))
    
    with col2:
        transacciones_total = cajeros['Volumen de Transacciones Diarias'].sum()
        st.metric("Transacciones/día", int(transacciones_total))
    
    with col3:
        tipos_transacciones = cajeros['Tipo de Transacciones'].nunique()
        st.metric("Tipos de Transacciones", tipos_transacciones)
    
    # Mostrar detalle de tipos de transacciones
    col1, col2 = st.columns([1, 1])
    
    with col1:
        tipos_dist = cajeros['Tipo de Transacciones'].value_counts()
        st.markdown("**Distribución por Tipo:**")
        for tipo, cantidad in tipos_dist.items():
            st.write(f"• {tipo}: {cantidad} cajeros")
    
    with col2:
        st.markdown("**Volumen por Tipo:**")
        vol_por_tipo = cajeros.groupby('Tipo de Transacciones')['Volumen de Transacciones Diarias'].sum()
        for tipo, volumen in vol_por_tipo.items():
            st.write(f"• {tipo}: {int(volumen):,} transacciones/día")
    
    st.divider()
    
    # SECCIÓN 2: Matriz de Distancias entre Sucursales
    crear_seccion_encabezado(
        titulo="Matriz de Distancias entre Sucursales",
        descripcion="Distancias en kilómetros entre todas las ubicaciones"
    )
    
    matriz_dist = crear_matriz_distancias(sucursales)
    nombres_sucursales = sucursales['Nombre'].tolist()
    
    fig_matriz = crear_grafico_matriz_distancias(
        matriz_dist,
        etiquetas=nombres_sucursales
    )
    st.plotly_chart(fig_matriz, use_container_width=True)
    
    st.divider()
    
    # SECCIÓN 3: Rutas Óptimas de Mantenimiento
    crear_seccion_encabezado(
        titulo="Plan de Rutas Óptimas para Mantenimiento y Reposición",
        descripcion="Sucursales Principales atienden a Sucursales Secundarias más cercanas"
    )
    
    rutas_optimas = calcular_rutas_mantenimiento(sucursales)
    
    # Mapa de rutas
    st.markdown("### Mapa de Rutas de Mantenimiento")
    
    try:
        mapa_rutas = crear_mapa_rutas_mantenimiento(sucursales, rutas_optimas)
        st.components.v1.html(mapa_rutas._repr_html_(), height=600, width=None)
    except Exception as e:
        st.error(f"Error al generar el mapa: {e}")
    
    st.divider()
    
    # Tabla de Rutas
    st.markdown("### Detalle del Plan de Mantenimiento")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            rutas_optimas[['Sucursal_Origen', 'Tipo_Origen', 'Sucursal_Destino', 
                          'Tipo_Destino', 'Distancia_km', 'Tiempo_Estimado_min']],
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("**Métricas del Plan:**")
        
        distancia_total = rutas_optimas['Distancia_km'].sum()
        st.metric("Distancia Total", f"{distancia_total:.2f} km")
        
        tiempo_total = rutas_optimas['Tiempo_Estimado_min'].sum()
        st.metric("Tiempo Total", f"{int(tiempo_total)} min")
        
        num_rutas = len(rutas_optimas)
        st.metric("Número de Rutas", num_rutas)
        
        distancia_promedio = distancia_total / num_rutas if num_rutas > 0 else 0
        st.metric("Distancia Promedio", f"{distancia_promedio:.2f} km")
    
    st.divider()
    
    # SECCIÓN 4: Análisis por Sucursal Principal
    crear_seccion_encabezado(titulo="Análisis por Sucursal Principal")
    
    sucursales_principales = sucursales[sucursales['Tipo de Sucursal'] == 'Sucursal Principal']
    
    tabs = st.tabs([row['Nombre'] for _, row in sucursales_principales.iterrows()])
    
    for tab, (_, sucursal_principal) in zip(tabs, sucursales_principales.iterrows()):
        with tab:
            rutas_sucursal = rutas_optimas[
                rutas_optimas['Sucursal_Origen'] == sucursal_principal['Nombre']
            ]
            
            if len(rutas_sucursal) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sucursales Atendidas",
                        len(rutas_sucursal)
                    )
                
                with col2:
                    st.metric(
                        "Distancia Total",
                        f"{rutas_sucursal['Distancia_km'].sum():.2f} km"
                    )
                
                with col3:
                    st.metric(
                        "Tiempo Total",
                        f"{int(rutas_sucursal['Tiempo_Estimado_min'].sum())} min"
                    )
                
                st.markdown("**Rutas asignadas:**")
                st.dataframe(
                    rutas_sucursal[['Sucursal_Destino', 'Distancia_km', 'Tiempo_Estimado_min']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Esta sucursal principal no tiene sucursales secundarias asignadas.")
    


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