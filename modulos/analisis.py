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
    crear_mapa_rutas_mantenimiento,
    aplicar_tema,
    crear_heatmap_productos_sucursales,
    crear_analisis_productos_por_tipo_sucursal,
    crear_comparativa_clientes_por_tipo_sucursal
)

from modulos.predicciones import (
        generar_datos_historicos,
        entrenar_modelo_regresion,
        generar_predicciones_futuras,
        analizar_tendencias_por_sucursal,
        analizar_estacionalidad,
        calcular_indicadores_demanda,
        crear_grafico_series_temporal,
        crear_grafico_predicciones_vs_historico,
        crear_grafico_comparacion_productos,
        crear_grafico_tendencias_comparativas,
        crear_grafico_estacionalidad,
        crear_heatmap_sucursal_mes
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
        # Tabla sin columnas de tipo
        st.dataframe(
            rutas_optimas[['Sucursal_Origen', 'Sucursal_Destino', 
                          'Distancia_km', 'Tiempo_Estimado_min']],
            use_container_width=True,
            hide_index=True
        )
        
        # Gráfico de transacciones por tipo debajo de la tabla
        st.markdown("### Análisis de Demanda por Tipo de Transacción")
        
        datos_consolidados = obtener_datos_consolidados()
        
        try:
            fig_trans_tipo = crear_grafico_transacciones_cajeros_por_tipo(datos_consolidados)
            st.plotly_chart(fig_trans_tipo, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar gráfico: {e}")
    
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
        
        st.markdown("---")
        
        


# PÁGINA 4: MARKETING DIRIGIDO

def pagina_marketing_dirigido():
    
    # Cargar datos
    datos_consolidados = obtener_datos_consolidados()
    
    # SECCIÓN 1: Resumen de Productos por Sucursal
    crear_seccion_encabezado(titulo="Distribución de Productos por Sucursal")
    
    resumen_productos = datos_consolidados.groupby(['Nombre', 'Productos Financieros Adquiridos']).agg({
        'Numero_Clientes_Producto': 'sum',
        'Volumen_Ventas_Producto': 'sum'
    }).reset_index()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_clientes = resumen_productos['Numero_Clientes_Producto'].sum()
        st.metric("Total de Clientes", f"{int(total_clientes):,}")
    
    with col2:
        total_ventas = resumen_productos['Volumen_Ventas_Producto'].sum()
        st.metric("Volumen Total de Ventas", f"${int(total_ventas):,}")
    
    with col3:
        productos_unicos = resumen_productos['Productos Financieros Adquiridos'].nunique()
        st.metric("Productos Activos", productos_unicos)
    
    st.divider()
    
    # SECCIÓN 2: Mapa de Calor de Ventas
    crear_seccion_encabezado(
        titulo="Mapa de Calor: Productos por Ubicación",
        descripcion="Visualización del volumen de ventas de cada producto en cada sucursal específica"
    )
    
    try:
        fig_heatmap = crear_heatmap_productos_sucursales(datos_consolidados)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar mapa de calor: {e}")
    
    st.divider()
    
    # SECCIÓN 3: Análisis por Tipo de Sucursal
    crear_seccion_encabezado(
        titulo="Análisis por Tipo de Sucursal",
        descripcion="Comparación del desempeño de productos en Sucursales Principales vs Secundarias"
    )
    
    try:
        fig_tipo, analisis_tipo = crear_analisis_productos_por_tipo_sucursal(datos_consolidados)
        st.plotly_chart(fig_tipo, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar análisis por tipo: {e}")
        analisis_tipo = pd.DataFrame()
    
    st.divider()
    
    # SECCIÓN 4: Comparativa de Clientes
    crear_seccion_encabezado(
        titulo="Distribución de Clientes por Tipo de Sucursal"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            fig_clientes = crear_comparativa_clientes_por_tipo_sucursal(datos_consolidados)
            st.plotly_chart(fig_clientes, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar comparativa: {e}")
    
    with col2:
        if not analisis_tipo.empty:
            st.markdown("**Métricas por Tipo**")
            
            # Calcular métricas agregadas
            for tipo in analisis_tipo['Tipo de Sucursal'].unique():
                datos_tipo = analisis_tipo[analisis_tipo['Tipo de Sucursal'] == tipo]
                
                st.markdown(f"**{tipo}**")
                total_ventas_tipo = datos_tipo['Volumen_Ventas_Producto'].sum()
                total_clientes_tipo = datos_tipo['Numero_Clientes_Producto'].sum()
                
                st.metric(
                    "Ventas Totales",
                    f"${int(total_ventas_tipo):,}"
                )
                st.metric(
                    "Clientes Totales",
                    f"{int(total_clientes_tipo):,}"
                )
    
    st.divider()
    
    
    # SECCIÓN 6: Análisis Detallado por Producto
    crear_seccion_encabezado(
        titulo="Análisis Detallado por Producto",
        descripcion="Selecciona un producto para ver su desempeño detallado por ubicación"
    )
    
    productos_disponibles = sorted(datos_consolidados['Productos Financieros Adquiridos'].unique())
    producto_seleccionado = st.selectbox(
        "Seleccionar Producto Financiero",
        productos_disponibles
    )
    
    if producto_seleccionado:
        datos_producto = datos_consolidados[
            datos_consolidados['Productos Financieros Adquiridos'] == producto_seleccionado
        ].groupby('Nombre').agg({
            'Numero_Clientes_Producto': 'sum',
            'Volumen_Ventas_Producto': 'sum',
            'Saldo Promedio de Cuentas': 'mean'
        }).reset_index().sort_values('Volumen_Ventas_Producto', ascending=True)
        
        fig_producto = go.Figure()
        
        fig_producto.add_trace(go.Bar(
            y=datos_producto['Nombre'],
            x=datos_producto['Volumen_Ventas_Producto'],
            orientation='h',
            marker=dict(
                color=datos_producto['Volumen_Ventas_Producto'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Ventas")
            ),
            text=datos_producto['Volumen_Ventas_Producto'],
            texttemplate='$%{text:,.0f}',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Ventas: $%{x:,.0f}<br>Clientes: %{customdata}<extra></extra>',
            customdata=datos_producto['Numero_Clientes_Producto']
        ))
        
        fig_producto.update_layout(
            title=f'Volumen de Ventas de {producto_seleccionado} por Sucursal',
            xaxis_title='Volumen de Ventas ($)',
            yaxis_title='Sucursal',
            template='plotly_white',
            height=400
        )
        
        fig_producto = aplicar_tema(fig_producto)
        st.plotly_chart(fig_producto, use_container_width=True)
        
        # Métricas del producto seleccionado
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_clientes_producto = datos_producto['Numero_Clientes_Producto'].sum()
            st.metric(
                f"Clientes Totales ({producto_seleccionado})",
                f"{int(total_clientes_producto):,}"
            )
        
        with col2:
            total_ventas_producto = datos_producto['Volumen_Ventas_Producto'].sum()
            st.metric(
                "Volumen Total de Ventas",
                f"${int(total_ventas_producto):,}"
            )
        
        with col3:
            promedio_saldo = datos_producto['Saldo Promedio de Cuentas'].mean()
            st.metric(
                "Saldo Promedio",
                f"${int(promedio_saldo):,}"
            )

# PREDICCIÓN DE DEMANDA

def pagina_prediccion_demanda():
    
    # Generar datos
    with st.spinner("Generando datos y entrenando modelo..."):
        df_historico = generar_datos_historicos()
        modelo, scaler, r2, mae = entrenar_modelo_regresion(df_historico)
        predicciones = generar_predicciones_futuras(df_historico, modelo, scaler, meses_futuros=6)
        tendencias = analizar_tendencias_por_sucursal(df_historico)
        estacionalidad = analizar_estacionalidad(df_historico)
        kpis = calcular_indicadores_demanda(df_historico, predicciones)
    
    
    # SECCIÓN 2: MÉTRICAS DEL MODELO
    crear_seccion_encabezado(titulo="Desempeño del Modelo Predictivo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}", delta="Bondad de ajuste")
    with col2:
        st.metric("MAE", f"{mae:.0f}", delta="Error medio")
    with col3:
        st.metric("Trans. Actual", f"{kpis['Transacciones_Promedio_Actual']:,.0f}")
    with col4:
        tasa = kpis['Tasa_Crecimiento_Esperado_%']
        st.metric("Crecimiento", f"{tasa:+.2f}%")
    
    st.divider()
    
    # SECCIÓN 3: TENDENCIAS
    crear_seccion_encabezado(titulo="Análisis de Tendencias por Sucursal")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_tendencias = crear_grafico_tendencias_comparativas(tendencias)
        st.plotly_chart(fig_tendencias, use_container_width=True)
    
    with col2:
        st.markdown("**Resumen**")
        for idx, row in tendencias.iterrows():
            emoji = "📈" if row['Tipo_Tendencia'] == 'Crecimiento' else "📉" if row['Tipo_Tendencia'] == 'Decrecimiento' else "➡️"
            st.write(f"{emoji} {row['Sucursal']}: {row['Cambio_Transacciones_%']:+.1f}%")
    
    st.divider()
    
    # SECCIÓN 4: SERIES TEMPORALES
    crear_seccion_encabezado(titulo="Series Temporales y Pronósticos")
    
    sucursales_disponibles = sorted(df_historico['Sucursal'].unique())
    sucursal_seleccionada = st.selectbox("Selecciona sucursal", sucursales_disponibles, key="sucursal_pred")
    
    if sucursal_seleccionada:
        tab1, tab2, tab3 = st.tabs(["Series Temporal", "Histórico vs Predicción", "Análisis"])
        
        with tab1:
            fig_series = crear_grafico_series_temporal(df_historico, sucursal_seleccionada)
            st.plotly_chart(fig_series, use_container_width=True)
        
        with tab2:
            fig_pred = crear_grafico_predicciones_vs_historico(df_historico, predicciones, sucursal_seleccionada)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            datos_suc = df_historico[df_historico['Sucursal'] == sucursal_seleccionada]
            pred_suc = predicciones[predicciones['Sucursal'] == sucursal_seleccionada]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prom. Histórico", f"{datos_suc['Transacciones_Sucursal'].mean():,.0f}")
            with col2:
                st.metric("Prom. Predicción", f"{pred_suc['Transacciones_Predichas'].mean():,.0f}")
            with col3:
                cambio = ((pred_suc['Transacciones_Predichas'].mean() - datos_suc['Transacciones_Sucursal'].mean()) / datos_suc['Transacciones_Sucursal'].mean() * 100)
                st.metric("Cambio", f"{cambio:+.1f}%")
        
        with tab3:
            fig_comp = crear_grafico_comparacion_productos(df_historico, sucursal_seleccionada)
            st.plotly_chart(fig_comp, use_container_width=True)
    
    st.divider()
    
    # SECCIÓN 5: ESTACIONALIDAD
    crear_seccion_encabezado(titulo="Patrones de Estacionalidad")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_estac = crear_grafico_estacionalidad(estacionalidad)
        st.plotly_chart(fig_estac, use_container_width=True)
    
    with col2:
        st.markdown("**Análisis**")
        mes_max = estacionalidad.loc[estacionalidad['Transacciones_Sucursal'].idxmax()]
        mes_min = estacionalidad.loc[estacionalidad['Transacciones_Sucursal'].idxmin()]
        
        st.success(f"📈 Pico: {mes_max['Mes_Nombre']}")
        st.warning(f"📉 Bajo: {mes_min['Mes_Nombre']}")
        
        variacion = ((mes_max['Transacciones_Sucursal'] - mes_min['Transacciones_Sucursal']) / mes_min['Transacciones_Sucursal'] * 100)
        st.info(f"Variación: {variacion:.1f}%")
    
    st.divider()
    
    # SECCIÓN 6: MAPA DE CALOR
    crear_seccion_encabezado(titulo="Mapa de Calor: Actividad por Sucursal-Mes")
    
    fig_heatmap = crear_heatmap_sucursal_mes(df_historico)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.divider()
    
    # SECCIÓN 7: TABLA DE PREDICCIONES
    crear_seccion_encabezado(titulo="Tabla de Predicciones Futuras")
    
    pred_tabla = predicciones.groupby(['Fecha', 'Sucursal']).agg({
        'Transacciones_Predichas': 'sum',
        'Clientes_Proyectados': 'sum',
        'Confidence': 'first'
    }).reset_index().sort_values('Fecha')
    
    pred_tabla.columns = ['Fecha', 'Sucursal', 'Transacciones', 'Clientes', 'Confianza']
    pred_tabla['Confianza_%'] = (pred_tabla['Confianza'] * 100).round(1)
    
    st.dataframe(pred_tabla[['Fecha', 'Sucursal', 'Transacciones', 'Clientes', 'Confianza_%']], 
                 use_container_width=True, hide_index=True)
    

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