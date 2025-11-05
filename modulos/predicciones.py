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
                'Transacciones_Proyectada': max(1, int(prediccion)),
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
        'Transacciones_Proyectada': 'sum'
    }).reset_index()
    
    if len(ultimos_3_meses) > 0 and len(proximos_3_meses) > 0:
        promedio_actual = ultimos_3_meses['Transacciones_Sucursal'].mean()
        promedio_futuro = proximos_3_meses['Transacciones_Proyectada'].mean()
        tasa_crecimiento = ((promedio_futuro - promedio_actual) / promedio_actual * 100) if promedio_actual > 0 else 0
    else:
        promedio_actual = 0
        promedio_futuro = 0
        tasa_crecimiento = 0
    
    return {
        'Transacciones_Promedio_Actual': int(promedio_actual),
        'Transacciones_Promedio_Proyectada': int(promedio_futuro),
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
    
    datos_pred = predicciones[predicciones['Sucursal'] == sucursal_seleccionada][['Fecha', 'Transacciones_Proyectada']].copy()
    datos_pred['Tipo'] = 'Predicción'
    datos_pred['Fecha'] = pd.to_datetime(datos_pred['Fecha'])
    datos_pred.rename(columns={'Transacciones_Proyectada': 'Transacciones'}, inplace=True)
    
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

# =====================================================
# PREDICCIÓN POR PRODUCTO FINANCIERO
# =====================================================

@st.cache_data
def generar_datos_historicos_productos():
    """Genera datos históricos de demanda por producto y ubicación"""
    datos_consolidados = obtener_datos_consolidados()
    
    historico_productos = []
    
    # Factores de estacionalidad por mes
    factores_estacionalidad = np.array([
        0.85, 0.87, 0.89, 0.91, 0.90, 0.92,
        0.94, 0.96, 0.95, 0.98, 1.02, 1.00
    ])
    
    # Generar para 12 meses
    for mes in range(12, -1, -1):
        fecha = datetime.now() - timedelta(days=30*mes)
        fecha_str = fecha.strftime('%Y-%m-%d')
        
        factor_estacional = factores_estacionalidad[abs(mes) % 12] if mes != 0 else 1.0
        
        for idx, row in datos_consolidados.iterrows():
            producto = row['Productos Financieros Adquiridos']
            sucursal = row['Nombre']
            
            # Variación por tipo de producto
            factor_producto = {
                'Préstamo': np.random.normal(1.0, 0.1),
                'Ahorro': np.random.normal(1.0, 0.08),
                'Inversión': np.random.normal(1.0, 0.12)
            }.get(producto, 1.0)
            
            factor_tendencia = 1.0 + (abs(mes) * 0.015)
            
            clientes = int(row['Numero_Clientes_Producto'] * factor_estacional * factor_producto * factor_tendencia)
            ventas = int(row['Volumen_Ventas_Producto'] * factor_estacional * factor_producto * factor_tendencia)
            
            historico_productos.append({
                'Fecha': fecha_str,
                'Mes': mes,
                'Sucursal': sucursal,
                'Producto': producto,
                'Latitud': row['Latitud'],
                'Longitud': row['Longitud'],
                'Clientes': max(1, clientes),
                'Volumen_Ventas': max(1, ventas),
            })
    
    df_historico = pd.DataFrame(historico_productos)
    df_historico['Fecha'] = pd.to_datetime(df_historico['Fecha'])
    df_historico = df_historico.sort_values('Fecha').reset_index(drop=True)
    
    return df_historico


@st.cache_data
def entrenar_modelo_productos(df_historico_productos):
    """Entrena modelo de predicción por producto"""
    from sklearn.preprocessing import LabelEncoder
    
    # Preparar features
    le_producto = LabelEncoder()
    le_sucursal = LabelEncoder()
    
    df = df_historico_productos.copy()
    df['Producto_Encoded'] = le_producto.fit_transform(df['Producto'])
    df['Sucursal_Encoded'] = le_sucursal.fit_transform(df['Sucursal'])
    df['Mes_Ano'] = df['Fecha'].dt.month
    
    X = df[['Producto_Encoded', 'Sucursal_Encoded', 'Mes_Ano', 'Clientes']].values
    y = df['Volumen_Ventas'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)
    
    y_pred = modelo.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    return modelo, scaler, le_producto, le_sucursal, r2, mae


@st.cache_data
def predecir_demanda_productos(_modelo, _scaler, _le_producto, _le_sucursal, 
                                df_historico_productos, meses_futuros=6):
    """Predice demanda futura por producto y ubicación"""
    datos_consolidados = obtener_datos_consolidados()
    
    predicciones = []
    fecha_inicio = df_historico_productos['Fecha'].max()
    
    for mes_futuro in range(1, meses_futuros + 1):
        fecha_prediccion = fecha_inicio + timedelta(days=30*mes_futuro)
        mes_del_ano = fecha_prediccion.month
        
        for idx, row in datos_consolidados.iterrows():
            producto = row['Productos Financieros Adquiridos']
            sucursal = row['Nombre']
            
            try:
                producto_encoded = _le_producto.transform([producto])[0]
                sucursal_encoded = _le_sucursal.transform([sucursal])[0]
            except:
                continue
            
            clientes_proyectados = int(row['Numero_Clientes_Producto'] * (1 + 0.03 * mes_futuro))
            
            X_pred = np.array([[
                producto_encoded,
                sucursal_encoded,
                mes_del_ano,
                clientes_proyectados
            ]])
            
            X_pred_scaled = _scaler.transform(X_pred)
            ventas_Proyectada = _modelo.predict(X_pred_scaled)[0]
            
            # Confianza: disminuye con el tiempo
            confianza = max(0.60, 0.95 - (0.06 * mes_futuro))
            
            predicciones.append({
                'Fecha': fecha_prediccion.strftime('%Y-%m-%d'),
                'Sucursal': sucursal,
                'Producto': producto,
                'Ventas_Proyectada': max(1, int(ventas_Proyectada)),
                'Clientes_Proyectados': clientes_proyectados,
                'Confianza_%': round(confianza * 100, 1),
                'Mes_Futuro': mes_futuro
            })
    
    return pd.DataFrame(predicciones)


# =====================================================
# VISUALIZACIONES MEJORADAS
# =====================================================

def crear_grafico_demanda_productos_ubicacion(predicciones_df):
    
    # Agrupar predicciones para el primer mes futuro
    datos = predicciones_df[predicciones_df['Mes_Futuro'] == 1].copy()
    
    fig = px.bar(
        datos,
        x='Sucursal',
        y='Ventas_Proyectada',
        color='Producto',
        title='Demanda Proyectada por Producto y Sucursal (Próximo Mes)',
        labels={'Ventas_Proyectada': 'Ventas Esperadas ($)'},
        text='Ventas_Proyectada',
        barmode='group',
        template='plotly_white',
        height=500
    )
    
    fig.update_traces(
        texttemplate='$%{text:,.0f}',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(
            title='Producto Financiero',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def crear_grafico_evolucion_productos(predicciones_df, sucursal_seleccionada):
    """Evolución temporal de demanda por producto en una sucursal"""
    
    datos = predicciones_df[predicciones_df['Sucursal'] == sucursal_seleccionada].copy()
    
    fig = go.Figure()
    
    colores_productos = {
        'Préstamo': '#2c5aa0',
        'Ahorro': '#27ae60',
        'Inversión': '#f39c12'
    }
    
    for producto in datos['Producto'].unique():
        datos_producto = datos[datos['Producto'] == producto].sort_values('Fecha')
        
        fig.add_trace(go.Scatter(
            x=datos_producto['Fecha'],
            y=datos_producto['Ventas_Proyectada'],
            mode='lines+markers',
            name=producto,
            line=dict(color=colores_productos.get(producto, '#333'), width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{fullData.name}</b><br>Fecha: %{x}<br>Ventas: $%{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Evolución de Demanda por Producto - {sucursal_seleccionada}',
        xaxis_title='Fecha',
        yaxis_title='Ventas Proyectada ($)',
        template='plotly_white',
        hovermode='x unified',
        height=400
    )
    
    return fig


def crear_grafico_top_productos_area(predicciones_df):
    """Identifica productos con mayor crecimiento por área"""
    
    # Calcular crecimiento promedio por producto y sucursal
    datos = predicciones_df.groupby(['Sucursal', 'Producto']).agg({
        'Ventas_Proyectada': 'mean'
    }).reset_index()
    
    # Top 3 productos por sucursal
    top_productos = datos.sort_values('Ventas_Proyectada', ascending=False).groupby('Sucursal').head(3)
    
    fig = px.sunburst(
        top_productos,
        path=['Sucursal', 'Producto'],
        values='Ventas_Proyectada',
        title='Productos con Mayor Demanda Esperada por Sucursal',
        color='Ventas_Proyectada',
        color_continuous_scale='Blues',
        height=600
    )
    
    fig.update_traces(
        textinfo='label+percent parent',
        hovertemplate='<b>%{label}</b><br>Ventas: $%{value:,.0f}<extra></extra>'
    )
    
    return fig


def crear_matriz_oportunidades_productos(predicciones_df, df_historico_productos):
    
    demanda_actual = df_historico_productos.groupby(['Sucursal', 'Producto']).agg({
        'Volumen_Ventas': 'mean'
    }).reset_index()
    demanda_actual.columns = ['Sucursal', 'Producto', 'Ventas_Actuales']
    
    demanda_futura = predicciones_df.groupby(['Sucursal', 'Producto']).agg({
        'Ventas_Proyectada': 'mean'
    }).reset_index()
    
    comparacion = demanda_actual.merge(demanda_futura, on=['Sucursal', 'Producto'])
    comparacion['Crecimiento_%'] = ((comparacion['Ventas_Proyectada'] - comparacion['Ventas_Actuales']) / 
                                      comparacion['Ventas_Actuales'] * 100).round(1)
    
    # CORRECCIÓN: Usar valor absoluto para el tamaño de las burbujas
    comparacion['Magnitud_Cambio'] = comparacion['Crecimiento_%'].abs()
    
    # Identificar oportunidades (crecimiento > 5%)
    comparacion['Oportunidad'] = comparacion['Crecimiento_%'].apply(
        lambda x: '🚀 Alto Potencial' if x > 10 else '📈 Crecimiento' if x > 5 else 
                  '📉 Decrecimiento' if x < -5 else '➡️ Estable'
    )
    
    fig = px.scatter(
        comparacion,
        x='Ventas_Actuales',
        y='Ventas_Proyectada',
        size='Magnitud_Cambio',
        color='Producto',
        hover_name='Sucursal',
        title='Matriz de Oportunidades: Demanda Actual vs Proyectada',
        labels={
            'Ventas_Actuales': 'Ventas Actuales ($)',
            'Ventas_Proyectada': 'Ventas Proyectada ($)'
        },
        size_max=30,
        template='plotly_white',
        height=500,
        # Agregar información adicional en el hover
        hover_data={
            'Crecimiento_%': ':.1f',
            'Magnitud_Cambio': False  # No mostrar esto en el hover
        }
    )
    
    # Línea diagonal (sin cambio)
    max_val = max(comparacion['Ventas_Actuales'].max(), comparacion['Ventas_Proyectada'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray', width=2),
        name='Sin cambio',
        showlegend=True
    ))
    
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='white'),
            opacity=0.7
        ),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig, comparacion