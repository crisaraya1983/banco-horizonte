import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

from modulos.carga_datos import (
    cargar_sucursales, cargar_cajeros, cargar_clientes, 
    cargar_productos, obtener_datos_consolidados
)
from modulos.geoespacial import (
    distancia_haversine
)


# =====================================================
# GENERACIÓN DE DATOS HISTÓRICOS FICTICIOS
# =====================================================

@st.cache_data
def generar_datos_historicos():
    """Genera 12 meses de datos históricos ficticios"""
    datos_consolidados = obtener_datos_consolidados()
    
    historico = []
    
    # Factores de estacionalidad por mes
    factores_estacionalidad = np.array([
        0.85, 0.87, 0.89, 0.91, 0.90, 0.92,
        0.94, 0.96, 0.95, 0.98, 1.02, 1.00
    ])
    
    # Variabilidad por sucursal
    ruido_sucursal = {}
    for sucursal in datos_consolidados['Nombre'].unique():
        ruido_sucursal[sucursal] = np.random.normal(1.0, 0.08, 12)
        ruido_sucursal[sucursal] = np.clip(ruido_sucursal[sucursal], 0.7, 1.3)
    
    # Generar para 12 meses
    for mes in range(12, -1, -1):
        fecha = datetime.now() - timedelta(days=30*mes)
        fecha_str = fecha.strftime('%Y-%m-%d')
        
        factor_estacional = factores_estacionalidad[abs(mes) % 12] if mes != 0 else 1.0
        
        for idx, row in datos_consolidados.iterrows():
            sucursal = row['Nombre']
            
            factor_ruido = ruido_sucursal[sucursal][abs(mes) % 12] if mes != 0 else 1.0
            factor_tendencia = 1.0 + (abs(mes) * 0.01)
            
            clientes = int(row['Numero_Clientes_Producto'] * factor_estacional * factor_ruido * factor_tendencia)
            transacciones = int(row['Volumen_Transacciones_Sucursal'] * factor_estacional * factor_ruido * factor_tendencia)
            
            clientes = int(clientes * np.random.normal(1.0, 0.05))
            transacciones = int(transacciones * np.random.normal(1.0, 0.05))
            
            historico.append({
                'Fecha': fecha_str,
                'Mes': mes,
                'Sucursal': sucursal,
                'Latitud': row['Latitud'],
                'Longitud': row['Longitud'],
                'Clientes': max(1, clientes),
                'Transacciones_Sucursal': max(1, transacciones),
            })
    
    df_historico = pd.DataFrame(historico)
    df_historico['Fecha'] = pd.to_datetime(df_historico['Fecha'])
    df_historico = df_historico.sort_values('Fecha').reset_index(drop=True)
    
    return df_historico


@st.cache_data
def entrenar_modelo_regresion(df_historico):
    """Entrena modelo de regresión lineal"""
    sucursales = cargar_sucursales()
    
    features = []
    
    for idx, row in df_historico.iterrows():
        mes_del_ano = row['Fecha'].month
        
        dist_sucursal = float('inf')
        for s_idx, s_row in sucursales.iterrows():
            dist = distancia_haversine(
                row['Latitud'], row['Longitud'],
                s_row['Latitud'], s_row['Longitud']
            )
            dist_sucursal = min(dist_sucursal, dist)
        
        features.append({
            'Distancia_Sucursal': dist_sucursal,
            'Clientes': row['Clientes'],
            'Mes_Ano': mes_del_ano,
            'Transacciones_Sucursal': row['Transacciones_Sucursal'],
        })
    
    df_features = pd.DataFrame(features)
    
    X = df_features[['Distancia_Sucursal', 'Clientes', 'Mes_Ano']].values
    y = df_features['Transacciones_Sucursal'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)
    
    y_pred = modelo.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    return modelo, scaler, r2, mae


@st.cache_data
def generar_predicciones_futuras(df_historico, _modelo, _scaler, meses_futuros=6):
    """Genera predicciones para N meses futuros"""
    sucursales = cargar_sucursales()
    datos_consolidados = obtener_datos_consolidados()
    
    datos_actuales = datos_consolidados.groupby('Nombre').agg({
        'Numero_Clientes_Producto': 'sum',
        'Latitud': 'first',
        'Longitud': 'first',
    }).reset_index()
    
    predicciones = []
    fecha_inicio = df_historico['Fecha'].max()
    
    for mes_futuro in range(1, meses_futuros + 1):
        fecha_prediccion = fecha_inicio + timedelta(days=30*mes_futuro)
        mes_del_ano = fecha_prediccion.month
        
        for idx, row in datos_actuales.iterrows():
            dist_sucursal = float('inf')
            for s_idx, s_row in sucursales.iterrows():
                dist = distancia_haversine(
                    row['Latitud'], row['Longitud'],
                    s_row['Latitud'], s_row['Longitud']
                )
                dist_sucursal = min(dist_sucursal, dist)
            
            X_pred = np.array([[
                dist_sucursal,
                row['Numero_Clientes_Producto'],
                mes_del_ano
            ]])
            
            X_pred_scaled = _scaler.transform(X_pred)  # ← CAMBIO AQUÍ
            prediccion = _modelo.predict(X_pred_scaled)[0]  # ← CAMBIO AQUÍ
            
            predicciones.append({
                'Fecha': fecha_prediccion.strftime('%Y-%m-%d'),
                'Sucursal': row['Nombre'],
                'Transacciones_Predichas': max(1, int(prediccion)),
                'Clientes_Proyectados': int(row['Numero_Clientes_Producto'] * (1 + 0.02 * mes_futuro)),
                'Confidence': 0.95 - (0.05 * mes_futuro)
            })
    
    return pd.DataFrame(predicciones)


@st.cache_data
def analizar_tendencias_por_sucursal(df_historico):
    """Calcula tendencias por sucursal"""
    tendencias = []
    
    for sucursal in df_historico['Sucursal'].unique():
        datos_sucursal = df_historico[df_historico['Sucursal'] == sucursal].sort_values('Fecha')
        
        datos_mensuales = datos_sucursal.groupby('Fecha').agg({
            'Clientes': 'sum',
            'Transacciones_Sucursal': 'sum',
        }).reset_index()
        
        if len(datos_mensuales) > 1:
            clientes_inicio = datos_mensuales['Clientes'].iloc[0]
            clientes_fin = datos_mensuales['Clientes'].iloc[-1]
            cambio_clientes = ((clientes_fin - clientes_inicio) / clientes_inicio * 100) if clientes_inicio > 0 else 0
            
            trans_inicio = datos_mensuales['Transacciones_Sucursal'].iloc[0]
            trans_fin = datos_mensuales['Transacciones_Sucursal'].iloc[-1]
            cambio_transacciones = ((trans_fin - trans_inicio) / trans_inicio * 100) if trans_inicio > 0 else 0
            
            tendencias.append({
                'Sucursal': sucursal,
                'Clientes_Promedio': datos_mensuales['Clientes'].mean(),
                'Transacciones_Promedio': datos_mensuales['Transacciones_Sucursal'].mean(),
                'Cambio_Clientes_%': cambio_clientes,
                'Cambio_Transacciones_%': cambio_transacciones,
                'Volatilidad': datos_mensuales['Transacciones_Sucursal'].std(),
                'Tipo_Tendencia': 'Crecimiento' if cambio_transacciones > 5 else 
                                 'Decrecimiento' if cambio_transacciones < -5 else 'Estable'
            })
    
    return pd.DataFrame(tendencias)


@st.cache_data
def analizar_estacionalidad(df_historico):
    """Analiza patrones de estacionalidad"""
    df_historico_copy = df_historico.copy()
    df_historico_copy['Mes_Numero'] = df_historico_copy['Fecha'].dt.month
    
    estacionalidad = df_historico_copy.groupby('Mes_Numero').agg({
        'Clientes': 'mean',
        'Transacciones_Sucursal': 'mean',
    }).reset_index()
    
    meses_nombres = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    estacionalidad['Mes_Nombre'] = estacionalidad['Mes_Numero'].map(meses_nombres)
    
    return estacionalidad


def calcular_indicadores_demanda(df_historico, predicciones):
    """Calcula KPIs"""
    ultimos_3_meses = df_historico.tail(int(len(df_historico)/4)).groupby('Fecha').agg({
        'Transacciones_Sucursal': 'sum'
    }).reset_index()
    
    proximos_3_meses = predicciones[predicciones['Confidence'] >= 0.80].groupby('Fecha').agg({
        'Transacciones_Predichas': 'sum'
    }).reset_index()
    
    if len(ultimos_3_meses) > 0 and len(proximos_3_meses) > 0:
        promedio_actual = ultimos_3_meses['Transacciones_Sucursal'].mean()
        promedio_futuro = proximos_3_meses['Transacciones_Predichas'].mean()
        tasa_crecimiento = ((promedio_futuro - promedio_actual) / promedio_actual * 100) if promedio_actual > 0 else 0
    else:
        promedio_actual = 0
        promedio_futuro = 0
        tasa_crecimiento = 0
    
    return {
        'Transacciones_Promedio_Actual': int(promedio_actual),
        'Transacciones_Promedio_Predichas': int(promedio_futuro),
        'Tasa_Crecimiento_Esperado_%': round(tasa_crecimiento, 2)
    }


# =====================================================
# VISUALIZACIONES
# =====================================================

def crear_grafico_series_temporal(df_historico, sucursal_seleccionada):
    """Series temporal interactivo"""
    datos = df_historico[df_historico['Sucursal'] == sucursal_seleccionada].groupby('Fecha').agg({
        'Transacciones_Sucursal': 'sum',
        'Clientes': 'sum',
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=datos['Fecha'],
        y=datos['Transacciones_Sucursal'],
        mode='lines+markers',
        name='Transacciones',
        line=dict(color='#2c5aa0', width=3),
        marker=dict(size=6),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=datos['Fecha'],
        y=datos['Clientes'],
        mode='lines',
        name='Clientes',
        line=dict(color='#27ae60', width=2, dash='dash'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f'Series Temporal: {sucursal_seleccionada}',
        xaxis_title='Fecha',
        yaxis=dict(
            title=dict(text='Transacciones', font=dict(color='#2c5aa0')),
            tickfont=dict(color='#2c5aa0')
        ),
        yaxis2=dict(
            title=dict(text='Clientes', font=dict(color='#27ae60')),
            tickfont=dict(color='#27ae60'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def crear_grafico_predicciones_vs_historico(df_historico, predicciones, sucursal_seleccionada):
    """Compara histórico vs predicción"""
    datos_hist = df_historico[df_historico['Sucursal'] == sucursal_seleccionada].groupby('Fecha').agg({
        'Transacciones_Sucursal': 'sum'
    }).reset_index()
    datos_hist['Tipo'] = 'Histórico'
    datos_hist.rename(columns={'Transacciones_Sucursal': 'Transacciones'}, inplace=True)
    
    datos_pred = predicciones[predicciones['Sucursal'] == sucursal_seleccionada][['Fecha', 'Transacciones_Predichas']].copy()
    datos_pred['Tipo'] = 'Predicción'
    datos_pred['Fecha'] = pd.to_datetime(datos_pred['Fecha'])
    datos_pred.rename(columns={'Transacciones_Predichas': 'Transacciones'}, inplace=True)
    
    datos_combinados = pd.concat([datos_hist, datos_pred], ignore_index=True)
    
    fig = px.line(
        datos_combinados,
        x='Fecha',
        y='Transacciones',
        color='Tipo',
        title=f'Histórico vs Predicciones: {sucursal_seleccionada}',
        markers=True,
        color_discrete_map={'Histórico': '#2c5aa0', 'Predicción': '#f39c12'},
        template='plotly_white',
        height=400
    )
    
    return fig


def crear_grafico_comparacion_productos(df_historico, sucursal_seleccionada):
    """Compara demanda por producto"""
    datos = df_historico[df_historico['Sucursal'] == sucursal_seleccionada].groupby('Fecha').agg({
        'Clientes': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=datos['Fecha'],
        y=datos['Clientes'],
        mode='lines+markers',
        name='Clientes',
        line=dict(color='#2c5aa0', width=2),
        marker=dict(size=5)
    ))
    
    fig.update_layout(
        title=f'Demanda: {sucursal_seleccionada}',
        xaxis_title='Fecha',
        yaxis_title='Clientes',
        template='plotly_white',
        height=400
    )
    
    return fig


def crear_grafico_tendencias_comparativas(tendencias_df):
    """Compara tendencias entre sucursales"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=tendencias_df['Sucursal'],
        y=tendencias_df['Cambio_Transacciones_%'],
        marker_color=['#27ae60' if x > 0 else '#e74c3c' for x in tendencias_df['Cambio_Transacciones_%']],
        text=tendencias_df['Cambio_Transacciones_%'].round(1),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Cambio en Transacciones por Sucursal',
        xaxis_title='Sucursal',
        yaxis_title='Cambio (%)',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def crear_grafico_estacionalidad(estacionalidad_df):
    """Visualiza estacionalidad"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=estacionalidad_df['Mes_Nombre'],
        y=estacionalidad_df['Transacciones_Sucursal'],
        marker_color='#2c5aa0',
        text=estacionalidad_df['Transacciones_Sucursal'].round(0),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Estacionalidad: Transacciones Promedio por Mes',
        xaxis_title='Mes',
        yaxis_title='Transacciones Promedio',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def crear_heatmap_sucursal_mes(df_historico):
    """Heatmap de actividad"""
    df_copy = df_historico.copy()
    df_copy['Mes_Numero'] = df_copy['Fecha'].dt.month
    
    pivot_data = df_copy.groupby(['Sucursal', 'Mes_Numero'])['Transacciones_Sucursal'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='Sucursal', columns='Mes_Numero', values='Transacciones_Sucursal')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=[f'Mes {i}' for i in pivot_table.columns],
        y=pivot_table.index,
        colorscale='Blues',
        text=np.round(pivot_table.values, 0),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title='Actividad Promedio por Sucursal y Mes',
        xaxis_title='Mes',
        yaxis_title='Sucursal',
        height=500,
        template='plotly_white'
    )
    
    return fig