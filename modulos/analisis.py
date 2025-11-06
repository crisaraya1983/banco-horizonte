import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from modulos.carga_datos import (
    cargar_sucursales, cargar_cajeros, cargar_clientes, cargar_productos,
    obtener_datos_consolidados
)
from modulos.geoespacial import (
    crear_matriz_distancias,
    calcular_rutas_mantenimiento,
    identificar_riesgos_geoespaciales,
    identificar_ubicaciones_optimas_sucursales  
)

from modulos.componentes import (
    crear_seccion_encabezado
)

from modulos.visualizaciones import (
    crear_mapa_cobertura_con_radios,
    crear_grafico_concentracion_clientes,
    crear_grafico_transacciones_por_ubicacion,
    crear_grafico_comparativa_cobertura_clientes,
    crear_grafico_matriz_distancias,
    crear_grafico_transacciones_cajeros_por_tipo,
    crear_mapa_segmentacion_geografica,
    crear_mapa_rutas_mantenimiento,
    aplicar_tema,
    crear_heatmap_productos_sucursales,
    crear_analisis_productos_por_tipo_sucursal,
    crear_comparativa_clientes_por_tipo_sucursal,
    crear_mapa_riesgos_geoespaciales,
    crear_grafico_riesgo_score
)

from modulos.predicciones import (
    generar_datos_historicos_productos,
    entrenar_modelo_productos,
    predecir_demanda_productos,
    crear_grafico_demanda_productos_ubicacion,
    crear_grafico_evolucion_productos,
    crear_grafico_top_productos_area,
    crear_matriz_oportunidades_productos
)

# PÁGINA 1: ANÁLISIS DE COBERTURA GEOGRÁFICA

def pagina_analisis_cobertura():

    datos_consolidados = obtener_datos_consolidados()
    
    ubicaciones_sucursales = datos_consolidados[
        ['Nombre', 'Latitud', 'Longitud', 'Tipo de Sucursal', 
         'Número de Empleados', 'Volumen_Transacciones_Sucursal']
    ].drop_duplicates(subset=['Nombre']).reset_index(drop=True)
    
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
    
    crear_seccion_encabezado(
        titulo="Análisis de Concentración de Clientes"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_concentracion = crear_grafico_concentracion_clientes(datos_consolidados)
            st.plotly_chart(fig_concentracion, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de concentración: {e}")
    
    with col2:
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
        
        st.markdown("**Detalle por Sucursal**")
        st.dataframe(concentracion_stats, use_container_width=True, hide_index=True)
    
    st.divider()

    crear_seccion_encabezado(
        titulo="Volumen de Transacciones por Ubicación"
    )

    col1, col2 = st.columns(2)

    with col1:
        try:
            fig_transacciones_sucursal = crear_grafico_transacciones_por_ubicacion(datos_consolidados)
            st.plotly_chart(fig_transacciones_sucursal, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de transacciones de sucursal: {e}")

    with col2:
        try:
            fig_transacciones_cajeros = crear_grafico_transacciones_cajeros_por_tipo(datos_consolidados)
            st.plotly_chart(fig_transacciones_cajeros, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de transacciones de cajeros: {e}")

    st.divider()

    crear_seccion_encabezado(
        titulo="Detalle de Transacciones por Ubicación"
    )

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
    
    datos_consolidados = obtener_datos_consolidados()
    
    crear_seccion_encabezado(
        titulo="Mapa de Segmentación: Clientes vs. Transacciones por Sucursal"
    )
    
    try:
        mapa_seg, datos_sucursales = crear_mapa_segmentacion_geografica(datos_consolidados)
        st.components.v1.html(mapa_seg._repr_html_(), height=700, width=None)
    except Exception as e:
        st.error(f"Error al generar el mapa: {e}")
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Análisis de Eficiencia Operativa por Sucursal"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Detalle de Sucursales**")
        
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
    
    crear_seccion_encabezado(
        titulo="Identificación de Zonas Estratégicas"
    )
    
    q_clientes_75 = datos_sucursales['Numero_Clientes_Producto'].quantile(0.75)
    q_transacciones_75 = datos_sucursales['Volumen_Transacciones_Sucursal'].quantile(0.75)
    q_clientes_25 = datos_sucursales['Numero_Clientes_Producto'].quantile(0.25)
    q_transacciones_25 = datos_sucursales['Volumen_Transacciones_Sucursal'].quantile(0.25)
    
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
    
    crear_seccion_encabezado(
        titulo="Matriz de Distancias entre Sucursales"
    )
    
    matriz_dist = crear_matriz_distancias(sucursales)
    nombres_sucursales = sucursales['Nombre'].tolist()
    
    fig_matriz = crear_grafico_matriz_distancias(
        matriz_dist,
        etiquetas=nombres_sucursales
    )
    st.plotly_chart(fig_matriz, use_container_width=True)
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Plan de Rutas para Mantenimiento y Reposición de Efectivo"
    )
    
    rutas_optimas = calcular_rutas_mantenimiento(sucursales)
    
    try:
        mapa_rutas = crear_mapa_rutas_mantenimiento(sucursales, rutas_optimas)
        st.components.v1.html(mapa_rutas._repr_html_(), height=600, width=None)
    except Exception as e:
        st.error(f"Error al generar el mapa: {e}")
    
    st.divider()
    
    st.markdown("### Detalle del Plan de Mantenimiento")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            rutas_optimas[['Sucursal_Origen', 'Sucursal_Destino', 
                          'Distancia_km', 'Tiempo_Estimado_min']],
            use_container_width=True,
            hide_index=True
        )
        
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
    
    datos_consolidados = obtener_datos_consolidados()
    
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

    crear_seccion_encabezado(
        titulo="Volumen de ventas de cada Productos por Ubicación"
    )
    
    try:
        fig_heatmap = crear_heatmap_productos_sucursales(datos_consolidados)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar mapa de calor: {e}")
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Análisis por Tipo de Sucursal"
    )
    
    try:
        fig_tipo, analisis_tipo = crear_analisis_productos_por_tipo_sucursal(datos_consolidados)
        st.plotly_chart(fig_tipo, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar análisis por tipo: {e}")
        analisis_tipo = pd.DataFrame()
    
    st.divider()
    
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
    
    crear_seccion_encabezado(
        titulo="Ventas por Producto"
    )
    
    productos_disponibles = sorted(datos_consolidados['Productos Financieros Adquiridos'].unique())
    producto_seleccionado = st.selectbox(
        "Selecciona un Producto Financiero",
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
    
    with st.spinner("Generando datos históricos y entrenando modelos predictivos..."):
        df_historico_productos = generar_datos_historicos_productos()
        modelo_prod, scaler_prod, le_producto, le_sucursal, r2_prod, mae_prod = entrenar_modelo_productos(df_historico_productos)
        predicciones_productos = predecir_demanda_productos(
            modelo_prod, scaler_prod, le_producto, le_sucursal, 
            df_historico_productos, meses_futuros=6
        )
    
    crear_seccion_encabezado(titulo="Desempeño del Modelo Predictivo")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R² Score", 
            f"{r2_prod:.3f}",
            help="Indica qué tan bien el modelo explica la variabilidad de los datos. Valores cercanos a 1.0 son mejores."
        )

    with col2:
        st.metric(
            "Error Promedio (MAE)", 
            f"${mae_prod:,.0f}",
            help="Error absoluto medio: diferencia promedio entre valores reales y predichos"
        )

    with col3:
        ventas_actuales = df_historico_productos.groupby('Fecha')['Volumen_Ventas'].sum().tail(3).mean()
        st.metric(
            "Ventas Promedio Actual",
            f"${ventas_actuales:,.0f}",
            help="Promedio de ventas totales de los últimos 3 meses"
        )

    with col4:
        ventas_proyectadas_agregadas = predicciones_productos[
            predicciones_productos['Mes_Futuro'] <= 3
        ].groupby('Fecha')['Ventas_Proyectada'].sum()
        
        ventas_proyectada = ventas_proyectadas_agregadas.mean()
        cambio = ((ventas_proyectada - ventas_actuales) / ventas_actuales * 100)
        
        st.metric(
            "Crecimiento Esperado",
            f"{cambio:+.1f}%",
            help="Cambio esperado en ventas totales para los próximos 3 meses"
        )
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Demanda Proyectada por Producto y Área Geográfica",
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_demanda = crear_grafico_demanda_productos_ubicacion(predicciones_productos)
        st.plotly_chart(fig_demanda, use_container_width=True)
    
    with col2:
        st.markdown("**Resumen por Producto**")
        
        resumen_producto = predicciones_productos[predicciones_productos['Mes_Futuro'] == 1].groupby('Producto').agg({
            'Ventas_Proyectada': 'sum',
            'Clientes_Proyectados': 'sum'
        }).reset_index().sort_values('Ventas_Proyectada', ascending=False)
        
        for idx, row in resumen_producto.iterrows():
            st.metric(
                row['Producto'],
                f"${row['Ventas_Proyectada']:,.0f}",
                delta=f"{row['Clientes_Proyectados']} clientes"
            )
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Proyección de Ventas por Producto"
    )
    
    sucursales_disponibles = sorted(predicciones_productos['Sucursal'].unique())
    sucursal_seleccionada = st.selectbox(
        "Selecciona una sucursal para un análisis detallado",
        sucursales_disponibles,
        key="sucursal_evolucion"
    )
    
    if sucursal_seleccionada:
        fig_evolucion = crear_grafico_evolucion_productos(predicciones_productos, sucursal_seleccionada)
        st.plotly_chart(fig_evolucion, use_container_width=True)
        
        datos_sucursal = predicciones_productos[
            predicciones_productos['Sucursal'] == sucursal_seleccionada
        ][['Fecha', 'Producto', 'Ventas_Proyectada', 'Clientes_Proyectados']].sort_values('Fecha')

        st.markdown(f"**Proyección Detallada - {sucursal_seleccionada}**")
        st.dataframe(datos_sucursal, use_container_width=True, hide_index=True)
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Distribución de Demanda por Área"
    )
    
    fig_sunburst = crear_grafico_top_productos_area(predicciones_productos)
    st.plotly_chart(fig_sunburst, use_container_width=True)
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Matriz de Oportunidades de Crecimiento"
    )
    
    fig_oportunidades, df_oportunidades = crear_matriz_oportunidades_productos(
        predicciones_productos, df_historico_productos
    )
    st.plotly_chart(fig_oportunidades, use_container_width=True)
    
# ANÁLISIS DE RIESGOS

def pagina_analisis_riesgos():
    
    datos_consolidados = obtener_datos_consolidados()
    sucursales = cargar_sucursales()
    clientes = cargar_clientes()
    productos = cargar_productos()
    riesgos_df = identificar_riesgos_geoespaciales(sucursales, datos_consolidados, productos)
    ubicaciones_optimas = identificar_ubicaciones_optimas_sucursales(
        clientes, sucursales, datos_consolidados, n_clusters=3
    )
    
    crear_seccion_encabezado(
        titulo="Análisis de Riesgos Geoespaciales",
        descripcion="Evaluación de vulnerabilidades en las sucursales bancarias"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        muy_alto = len(riesgos_df[riesgos_df['Nivel_Riesgo'] == '🔴 Muy Alto'])
        st.metric("🔴 Muy Alto", muy_alto)
    
    with col2:
        alto = len(riesgos_df[riesgos_df['Nivel_Riesgo'] == '🟠 Alto'])
        st.metric("🟠 Alto", alto)
    
    with col3:
        total_clientes_en_riesgo = riesgos_df[
            riesgos_df['Nivel_Riesgo'].isin(['🔴 Muy Alto', '🟠 Alto'])
        ]['Clientes_Totales'].sum()
        st.metric("Clientes en Riesgo", f"{int(total_clientes_en_riesgo):,}")
    
    with col4:
        oportunidades = len(ubicaciones_optimas)
        st.metric("Ubicaciones Óptimas", oportunidades)
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Distribución Geoespacial de Riesgos"
    )
    
    try:
        mapa_riesgos = crear_mapa_riesgos_geoespaciales(riesgos_df, sucursales)
        st.components.v1.html(mapa_riesgos._repr_html_(), height=650, width=None)
    except Exception as e:
        st.error(f"Error al generar mapa: {e}")
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Indicador de Riesgo por Sucursal"
    )
    
    try:
        fig_score = crear_grafico_riesgo_score(riesgos_df)
        st.plotly_chart(fig_score, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.divider()
    
    crear_seccion_encabezado(
        titulo="Detalle de Sucursales por Nivel de Riesgo"
    )
    
    tabs = st.tabs(["🔴 Muy Alto", "🟠 Alto", "🟡 Medio", "🟢 Bajo"])
    niveles = ["🔴 Muy Alto", "🟠 Alto", "🟡 Medio", "🟢 Bajo"]
    
    for tab, nivel in zip(tabs, niveles):
        with tab:
            datos_nivel = riesgos_df[riesgos_df['Nivel_Riesgo'] == nivel]
            
            if len(datos_nivel) > 0:
                tabla_mostrar = datos_nivel[[
                    'Sucursal', 'Riesgo_Score', 'Clientes_Totales',
                    'Volumen_Ventas_Total', 'Volumen_Transacciones',
                    'Trans_Por_Empleado', 'Distancia_Sucursal_Mas_Cercana_km'
                ]].copy()
                
                tabla_mostrar.columns = [
                    'Sucursal', 'Score', 'Clientes', 'Ventas ($)', 
                    'Trans/mes', 'Trans/Emp', 'Dist. Cercana (km)'
                ]
                
                st.dataframe(
                    tabla_mostrar.sort_values('Score', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                if nivel == "🔴 Muy Alto":
                    st.warning(
                        """**Acciones Inmediatas:**
                        - Revisar viabilidad operativa
                        - Considerar fusión o relocalización
                        - Aumentar inversión en marketing digital"""
                    )
                elif nivel == "🟠 Alto":
                    st.info(
                        """**Acciones Recomendadas:**
                        - Diversificar cartera de productos
                        - Fortalecer atención al cliente
                        - Evaluar ofertas personalizadas"""
                    )
            else:
                st.info(f"No hay sucursales con nivel de riesgo {nivel}")
    
