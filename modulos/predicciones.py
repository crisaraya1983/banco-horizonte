import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

from modulos.carga_datos import (
    obtener_datos_consolidados
)

@st.cache_data
def generar_datos_historicos_productos():
    """Genera datos históricos de demanda por producto y ubicación"""
    datos_consolidados = obtener_datos_consolidados()
    
    historico_productos = []
    
    factores_estacionalidad = np.array([
        0.85, 0.87, 0.89, 0.91, 0.90, 0.92,
        0.94, 0.96, 0.95, 0.98, 1.02, 1.00
    ])
    
    for mes in range(12, -1, -1):
        fecha = datetime.now() - timedelta(days=30*mes)
        fecha_str = fecha.strftime('%Y-%m-%d')
        
        factor_estacional = factores_estacionalidad[abs(mes) % 12] if mes != 0 else 1.0
        
        for idx, row in datos_consolidados.iterrows():
            producto = row['Productos Financieros Adquiridos']
            sucursal = row['Nombre']
            
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


def crear_grafico_demanda_productos_ubicacion(predicciones_df):
    
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
        yaxis_title='Ventas Proyectadas ($)', 
        template='plotly_white',
        hovermode='x unified',
        height=400
    )
    
    return fig


def crear_grafico_top_productos_area(predicciones_df):
    
    datos = predicciones_df.groupby(['Sucursal', 'Producto']).agg({
        'Ventas_Proyectada': 'mean'
    }).reset_index()
    
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
    
    comparacion['Magnitud_Cambio'] = comparacion['Crecimiento_%'].abs()
    
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
        hover_data={
            'Crecimiento_%': ':.1f',
            'Magnitud_Cambio': False
        }
    )
    
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