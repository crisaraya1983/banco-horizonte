"""
Aplicación de Análisis Geoespacial - Banco Horizonte
=====================================================

Esta es la aplicación principal de Streamlit para el análisis geoespacial
de las sucursales, cajeros automáticos y clientes del Banco Horizonte.

La aplicación utiliza una navegación basada en sidebar que permite al usuario
seleccionar entre diferentes análisis y visualizaciones.

Estructura:
- Inicio: Presentación general del proyecto
- Análisis de Cobertura: Visualización de distribución de sucursales y cajeros
- Segmentación Geográfica: Análisis de clientes por región
- Optimización Logística: Rutas de mantenimiento de cajeros
- Marketing Dirigido: Análisis de patrones de compra por ubicación
- Predicción de Demanda: Modelos predictivos geoespaciales
- Análisis de Riesgos: Evaluación de riesgos por área geográfica
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Importamos nuestros módulos personalizados
from modulos.carga_datos import (
    cargar_sucursales, cargar_cajeros, cargar_clientes, 
    cargar_productos, cargar_todos_los_datos, validar_datos
)
from modulos.geoespacial import calcular_cobertura_geográfica
from modulos.analisis import (
    pagina_analisis_cobertura,
    pagina_segmentacion_geografica,
    pagina_optimizacion_logistica,
    pagina_marketing_dirigido,
    pagina_prediccion_demanda,
    pagina_analisis_riesgos
)


# ============================================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================================

def configurar_pagina():
    """
    Configura las propiedades generales de la página de Streamlit.
    
    Esto incluye el icono, el título que aparece en la pestaña del navegador,
    y la disposición del layout (wide es más aprovecha el espacio horizontal).
    """
    st.set_page_config(
        page_title="Banco Horizonte - Análisis Geoespacial",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def aplicar_estilos_personalizados():
    """
    Aplica estilos CSS personalizados para mejorar la apariencia visual.
    
    Streamlit permite inyectar CSS personalizado para modificar la apariencia
    más allá de los temas predefinidos.
    """
    st.markdown("""
    <style>
    /* Estilos personalizados para la aplicación */
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .subheader-custom {
        color: #555;
        font-size: 1.3em;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# PÁGINAS DE LA APLICACIÓN
# ============================================================================

def pagina_inicio():
    """
    Página de inicio: Presenta el proyecto y permite explorar datos.
    
    Esta página es el punto de entrada del usuario a la aplicación.
    Muestra un resumen del proyecto, el contexto del caso, y estadísticas
    generales de los datos disponibles.
    """
    st.markdown('<div class="main-header">🏦 Banco Horizonte: Análisis Geoespacial</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Bienvenido al Sistema de Análisis Geoespacial
    
    Este dashboard te permite explorar y analizar la red de sucursales, cajeros automáticos
    y clientes del Banco Horizonte utilizando información geográfica.
    
    #### 📋 Contexto del Proyecto
    Banco Horizonte enfrenta una creciente competencia en el mercado financiero. Para optimizar
    sus servicios, hemos implementado un sistema de análisis geoespacial que utiliza Sistemas de
    Información Geográfica (SIG) para:
    
    - 📍 Analizar la distribución actual de sucursales y cajeros automáticos
    - 👥 Entender patrones de comportamiento de clientes por ubicación geográfica
    - 🚚 Optimizar rutas de mantenimiento y logística
    - 📢 Diseñar campañas de marketing dirigidas por región
    - 🔮 Predecir demanda futura de productos financieros
    - ⚠️ Evaluar riesgos geográficos y tomar decisiones estratégicas
    
    ---
    """)
    
    # Cargamos los datos
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    productos = cargar_productos()
    
    # Mostramos métricas generales en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="📊 Sucursales Activas", value=len(sucursales))
    
    with col2:
        st.metric(label="🏧 Cajeros Automáticos", value=len(cajeros))
    
    with col3:
        st.metric(label="👥 Clientes en la Base", value=len(clientes))
    
    with col4:
        st.metric(label="💼 Productos Financieros", value=len(productos))
    
    st.markdown("---")
    
    # Análisis de cobertura rápida
    st.markdown('<div class="subheader-custom">📈 Resumen de Cobertura Geográfica</div>', 
                unsafe_allow_html=True)
    
    cobertura = calcular_cobertura_geográfica(
        clientes, cajeros, sucursales,
        umbral_sucursal=10.0, umbral_cajero=5.0
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
        <strong>Cobertura de Sucursales</strong><br>
        {cobertura['cobertura_sucursales_pct']:.1f}% de clientes<br>
        <small>(dentro de 10 km)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
        <strong>Cobertura de Cajeros</strong><br>
        {cobertura['cobertura_cajeros_pct']:.1f}% de clientes<br>
        <small>(dentro de 5 km)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
        <strong>Cobertura Completa</strong><br>
        {cobertura['cobertura_completa_pct']:.1f}% de clientes<br>
        <small>(ambas coberturas)</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Muestra datos disponibles
    st.markdown('<div class="subheader-custom">📊 Vista Previa de Datos</div>', 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sucursales", "Cajeros", "Clientes", "Productos"])
    
    with tab1:
        st.dataframe(sucursales, use_container_width=True)
    
    with tab2:
        st.dataframe(cajeros, use_container_width=True)
    
    with tab3:
        st.dataframe(clientes, use_container_width=True)
    
    with tab4:
        st.dataframe(productos, use_container_width=True)
    
    st.markdown("---")
    
    # Información de validación de datos
    with st.expander("🔍 Estado de los Datos (Validación)"):
        validaciones = validar_datos()
        for dataset, estado in validaciones.items():
            if estado['estado'] == 'OK':
                st.success(f"✅ {dataset.capitalize()}: {estado['registros']} registros cargados")
            else:
                st.error(f"❌ {dataset.capitalize()}: {estado.get('mensaje', 'Error desconocido')}")


def pagina_en_construccion(nombre_pagina):
    """
    Página placeholder para análisis que aún están en desarrollo.
    
    Esta función muestra un mensaje indicando que la página está en construcción,
    con un ícono atractivo.
    
    Parámetros:
        nombre_pagina (str): Nombre de la página en construcción
    """
    st.markdown(f"## 🚧 {nombre_pagina} - En Construcción")
    st.info(
        f"Esta sección está siendo desarrollada. "
        f"Pronto podrás acceder a: {nombre_pagina}"
    )
    st.markdown("---")
    st.markdown("**Vuelve pronto para esta funcionalidad.**")


# ============================================================================
# NAVEGACIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que contiene la lógica de navegación de la aplicación.
    
    Streamlit ejecuta esta función de arriba a abajo cada vez que el usuario
    interactúa con la aplicación. Aquí creamos el sidebar con opciones de menú
    y llamamos a la página correspondiente.
    """
    # Configuramos la página
    configurar_pagina()
    aplicar_estilos_personalizados()
    
    # Creamos el sidebar con opciones de navegación
    with st.sidebar:
        st.markdown("# 🗺️ Navegación")
        st.markdown("---")
        
        pagina_seleccionada = st.radio(
            "Selecciona un análisis:",
            options=[
                "🏠 Inicio",
                "📍 Análisis de Cobertura",
                "🎯 Segmentación Geográfica",
                "🚚 Optimización Logística",
                "📢 Marketing Dirigido",
                "🔮 Predicción de Demanda",
                "⚠️ Análisis de Riesgos"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Información en el sidebar
        st.markdown("### 📚 Información")
        st.markdown("""
        Esta aplicación utiliza análisis geoespacial para optimizar
        las operaciones bancarias del Banco Horizonte.
        
        **Tecnologías utilizadas:**
        - Streamlit
        - Folium (mapas)
        - Plotly (gráficos)
        - Pandas (datos)
        - Scikit-learn (ML)
        """)
    
    # Lógica de ruteo: mostrar la página seleccionada
    if pagina_seleccionada == "🏠 Inicio":
        pagina_inicio()

    elif pagina_seleccionada == "📍 Análisis de Cobertura":
        pagina_analisis_cobertura()

    elif pagina_seleccionada == "🎯 Segmentación Geográfica":
        pagina_segmentacion_geografica()

    elif pagina_seleccionada == "🚚 Optimización Logística":
        pagina_optimizacion_logistica()

    elif pagina_seleccionada == "📢 Marketing Dirigido":
        pagina_marketing_dirigido()

    elif pagina_seleccionada == "🔮 Predicción de Demanda":
        pagina_prediccion_demanda()

    elif pagina_seleccionada == "⚠️ Análisis de Riesgos":
        pagina_analisis_riesgos()


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()