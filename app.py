import streamlit as st
import pandas as pd
from pathlib import Path

from modulos.carga_datos import (
    cargar_sucursales, cargar_cajeros, cargar_clientes, 
    cargar_productos, obtener_productos_consolidados,
    obtener_datos_consolidados
)

from modulos.analisis import (
    pagina_analisis_cobertura,
    pagina_segmentacion_geografica,
    pagina_optimizacion_logistica,
    pagina_marketing_dirigido,
    pagina_prediccion_demanda,
    pagina_analisis_riesgos
)

from modulos.componentes import inicializar_componentes


# CONFIGURACIÓN DE STREAMLIT

def configurar_pagina():
    st.set_page_config(
        page_title="Banco Horizonte - Análisis Geoespacial",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def aplicar_estilos():
    st.markdown("""
    <style>
    /* Paleta de colores y variables */
    :root {
        --color-primary: #1a365d;
        --color-secondary: #2c5aa0;
        --color-accent: #f7fafc;
        --color-text: #2d3748;
        --color-border: #e2e8f0;
    }
    
    .main-header {
        color: #1a365d;
        font-size: 2em;
        font-weight: 600;
        margin-bottom: 0.5em;
        margin-top: 0em;
        letter-spacing: 0.5px;
    }
    
    .section-header {
        color: #2d3748;
        font-size: 0.85em;
        font-weight: 700;
        margin-top: 1.5em;
        margin-bottom: 1em;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #718096;
    }
    
    .metric-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #2c5aa0;
        margin-bottom: 0.5em;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85em;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2em;
        color: #1a365d;
        font-weight: 600;
    }
    
    [data-testid="stTabs"] [role="tab"] {
        font-weight: 600;
        font-size: 0.95em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    inicializar_componentes()


# PÁGINA PRINCIPAL

def pagina_inicio():

    st.markdown('<div class="main-header">Análisis Geoespacial</div>', 
                unsafe_allow_html=True)
    
    sucursales = cargar_sucursales()
    cajeros = cargar_cajeros()
    clientes = cargar_clientes()
    productos = cargar_productos()
    
    st.markdown('<div class="section-header">Resumen de Datos</div>', 
                unsafe_allow_html=True)
    
    productos_consolidados = obtener_productos_consolidados()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Sucursales Activas", value=len(sucursales))
    
    with col2:
        st.metric(label="Cajeros Automáticos", value=len(cajeros))
    
    with col3:
        st.metric(label="Clientes", value=int(productos_consolidados['Total Clientes'].sum()))
    
    with col4:
        st.metric(label="Productos", value=len(productos_consolidados))
    
    st.divider()
    
    st.markdown('<div class="section-header">Cobertura de Clientes por Sucursal</div>', 
                unsafe_allow_html=True)
    
    datos_consolidados = obtener_datos_consolidados()
    
    clientes_por_sucursal = datos_consolidados.groupby(['Ubicación', 'Nombre', 'Tipo de Sucursal']).agg({
        'Numero_Clientes_Producto': 'sum'
    }).reset_index()
    
    clientes_por_sucursal.columns = ['Ubicación', 'Sucursal', 'Tipo de Sucursal', 'Total_Clientes']
    
    total_clientes = clientes_por_sucursal['Total_Clientes'].sum()
    
    clientes_por_sucursal['Porcentaje'] = (clientes_por_sucursal['Total_Clientes'] / total_clientes * 100).round(2)
    
    clientes_por_sucursal_sorted = clientes_por_sucursal.sort_values('Total_Clientes', ascending=False).reset_index(drop=True)
    
    cols = st.columns(5)
    
    for idx, row in clientes_por_sucursal_sorted.iterrows():
        col_idx = idx % 5
        with cols[col_idx]:
            st.metric(
                label=row['Sucursal'],
                value=f"{row['Porcentaje']:.2f}%"
            )
  
    st.divider()
    
    st.markdown('<div class="section-header">Datos Disponibles</div>', 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sucursales", "Cajeros", "Clientes", "Productos"])
    
    with tab1:
        st.dataframe(sucursales, use_container_width=True, hide_index=True)
    
    with tab2:
        st.dataframe(cajeros, use_container_width=True, hide_index=True)
    
    with tab3:
        st.dataframe(clientes, use_container_width=True, hide_index=True)
    
    with tab4:
        st.dataframe(productos, use_container_width=True, hide_index=True)


def main():

    configurar_pagina()
    aplicar_estilos()
    
    with st.sidebar:
        st.title("Navegación")
        st.divider()
        
        pagina_seleccionada = st.radio(
            "Seleccionar análisis",
            options=[
                "Inicio",
                "Análisis de Cobertura",
                "Segmentación Geográfica",
                "Optimización Logística",
                "Marketing Dirigido",
                "Predicción de Demanda",
                "Análisis de Riesgos"
            ],
            label_visibility="collapsed"
        )
    
    if pagina_seleccionada == "Inicio":
        pagina_inicio()
    elif pagina_seleccionada == "Análisis de Cobertura":
        pagina_analisis_cobertura()
    elif pagina_seleccionada == "Segmentación Geográfica":
        pagina_segmentacion_geografica()
    elif pagina_seleccionada == "Optimización Logística":
        pagina_optimizacion_logistica()
    elif pagina_seleccionada == "Marketing Dirigido":
        pagina_marketing_dirigido()
    elif pagina_seleccionada == "Predicción de Demanda":
        pagina_prediccion_demanda()
    elif pagina_seleccionada == "Análisis de Riesgos":
        pagina_analisis_riesgos()


if __name__ == "__main__":
    main()