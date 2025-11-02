"""
Módulo de Cálculos Geoespaciales
=================================
Este módulo encapsula toda la lógica de cálculos geográficos.
La idea es que los módulos de análisis y visualización no necesiten preocuparse
por los detalles matemáticos; simplemente llaman a estas funciones.

Los cálculos geoespaciales son la base de nuestro análisis porque permiten:
1. Entender proximidad entre clientes y puntos de servicio
2. Identificar áreas desatendidas
3. Optimizar rutas de mantenimiento
4. Segmentar clientes por ubicación
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


def distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en kilómetros entre dos puntos usando la fórmula de Haversine.
    
    La fórmula de Haversine es la forma correcta de calcular distancias en la tierra
    considerando su forma esférica. Es más precisa que simplemente restar coordenadas.
    
    Parámetros:
        lat1, lon1: Latitud y longitud del primer punto
        lat2, lon2: Latitud y longitud del segundo punto
    
    Returns:
        float: Distancia en kilómetros
    """
    # Convertir grados a radianes
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Fórmula de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radio de la tierra en kilómetros
    
    return c * r


def calcular_distancia_a_punto_mas_cercano(lat_cliente, lon_cliente, puntos_servicio):
    """
    Encuentra el punto de servicio más cercano a un cliente y calcula su distancia.
    
    Esto es crítico para entender la cobertura. Un cliente que está a más de 5 km
    del punto de servicio más cercano podría estar en una zona desatendida.
    
    Parámetros:
        lat_cliente, lon_cliente: Coordenadas del cliente
        puntos_servicio: DataFrame con columnas 'Latitud' y 'Longitud'
    
    Returns:
        dict: {'distancia_km': float, 'índice_punto': int}
    """
    if len(puntos_servicio) == 0:
        return {"distancia_km": float('inf'), "índice_punto": None}
    
    distancias = []
    for idx, row in puntos_servicio.iterrows():
        dist = distancia_haversine(
            lat_cliente, lon_cliente,
            row['Latitud'], row['Longitud']
        )
        distancias.append(dist)
    
    distancias = np.array(distancias)
    índice_mínimo = np.argmin(distancias)
    
    return {
        "distancia_km": distancias[índice_mínimo],
        "índice_punto": puntos_servicio.index[índice_mínimo]
    }


def calcular_distancia_a_sucursal_mas_cercana(clientes_df, sucursales_df):
    """
    Para cada cliente, calcula su distancia a la sucursal más cercana.
    
    Esta función agrega una columna al dataframe de clientes indicando
    cuál es la sucursal más cercana y a qué distancia se encuentra.
    
    Parámetros:
        clientes_df: DataFrame de clientes
        sucursales_df: DataFrame de sucursales
    
    Returns:
        pd.DataFrame: Dataframe de clientes con nuevas columnas de proximidad
    """
    clientes = clientes_df.copy()
    clientes['Distancia_a_Sucursal_km'] = 0.0
    clientes['Índice_Sucursal_Cercana'] = 0
    
    for idx, cliente in clientes.iterrows():
        resultado = calcular_distancia_a_punto_mas_cercano(
            cliente['Latitud'],
            cliente['Longitud'],
            sucursales_df
        )
        clientes.at[idx, 'Distancia_a_Sucursal_km'] = resultado['distancia_km']
        clientes.at[idx, 'Índice_Sucursal_Cercana'] = resultado['índice_punto']
    
    return clientes


def calcular_distancia_a_cajero_mas_cercano(clientes_df, cajeros_df):
    """
    Similar a la función anterior pero para cajeros automáticos.
    
    Parámetros:
        clientes_df: DataFrame de clientes
        cajeros_df: DataFrame de cajeros
    
    Returns:
        pd.DataFrame: Dataframe de clientes con información de proximidad a cajeros
    """
    clientes = clientes_df.copy()
    clientes['Distancia_a_Cajero_km'] = 0.0
    clientes['Índice_Cajero_Cercano'] = 0
    
    for idx, cliente in clientes.iterrows():
        resultado = calcular_distancia_a_punto_mas_cercano(
            cliente['Latitud'],
            cliente['Longitud'],
            cajeros_df
        )
        clientes.at[idx, 'Distancia_a_Cajero_km'] = resultado['distancia_km']
        clientes.at[idx, 'Índice_Cajero_Cercano'] = resultado['índice_punto']
    
    return clientes


def identificar_zonas_desatendidas(clientes_df, sucursales_df, umbral_km=5.0):
    """
    Identifica clientes que están por encima del umbral de distancia.
    
    Los clientes a más de 5 km (o el umbral que especifiques) del punto de servicio
    más cercano se consideran en una zona desatendida. Esto es información valiosa
    para decisiones de expansión.
    
    Parámetros:
        clientes_df: DataFrame de clientes (debe incluir 'Distancia_a_Sucursal_km')
        sucursales_df: DataFrame de sucursales (solo se usa para contexto)
        umbral_km: Distancia máxima aceptable en km
    
    Returns:
        pd.DataFrame: Subset de clientes que están en zonas desatendidas
    """
    if 'Distancia_a_Sucursal_km' not in clientes_df.columns:
        clientes_df = calcular_distancia_a_sucursal_mas_cercana(clientes_df, sucursales_df)
    
    desatendidos = clientes_df[clientes_df['Distancia_a_Sucursal_km'] > umbral_km]
    return desatendidos


def crear_matriz_distancias(puntos_df):
    """
    Crea una matriz de distancias entre todos los puntos (cajeros o sucursales).
    
    Esta matriz es útil para optimización logística. Por ejemplo, para planificar
    rutas de mantenimiento eficientes de cajeros.
    
    Parámetros:
        puntos_df: DataFrame con columnas 'Latitud' y 'Longitud'
    
    Returns:
        np.ndarray: Matriz simétrica NxN donde cada elemento [i,j] es la distancia
                   entre el punto i y el punto j en km
    """
    n_puntos = len(puntos_df)
    matriz = np.zeros((n_puntos, n_puntos))
    
    for i in range(n_puntos):
        for j in range(n_puntos):
            if i != j:
                lat1, lon1 = puntos_df.iloc[i][['Latitud', 'Longitud']]
                lat2, lon2 = puntos_df.iloc[j][['Latitud', 'Longitud']]
                matriz[i, j] = distancia_haversine(lat1, lon1, lat2, lon2)
    
    return matriz


def calcular_centroide_geográfico(ubicaciones_df):
    """
    Calcula el punto central (centroide) de un conjunto de ubicaciones.
    
    Esto es útil para entender el "centro de gravedad" de clientes o sucursales
    en una región específica.
    
    Parámetros:
        ubicaciones_df: DataFrame con columnas 'Latitud' y 'Longitud'
    
    Returns:
        tuple: (latitud_promedio, longitud_promedio)
    """
    lat_prom = ubicaciones_df['Latitud'].mean()
    lon_prom = ubicaciones_df['Longitud'].mean()
    return lat_prom, lon_prom


def agrupar_clientes_por_proximidad(clientes_df, sucursales_df):
    """
    Agrupa clientes por su sucursal más cercana.
    
    Esto nos permite entender cuántos clientes "pertenecen" a cada sucursal
    desde una perspectiva geográfica, independientemente de dónde realicen
    transacciones actualmente.
    
    Parámetros:
        clientes_df: DataFrame de clientes
        sucursales_df: DataFrame de sucursales
    
    Returns:
        dict: Diccionario donde cada clave es el índice de sucursal y el valor
              es una lista de índices de clientes más cercanos a esa sucursal
    """
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes_df, sucursales_df)
    
    agrupaciones = {}
    for idx_sucursal in sucursales_df.index:
        clientes_cercanos = clientes[clientes['Índice_Sucursal_Cercana'] == idx_sucursal]
        agrupaciones[idx_sucursal] = clientes_cercanos.index.tolist()
    
    return agrupaciones


def calcular_densidad_clientes_por_sucursal(clientes_df, sucursales_df):
    """
    Calcula cuántos clientes (por unidad de área) hay cerca de cada sucursal.
    
    La densidad de clientes es un indicador de demanda potencial en cada área.
    Una sucursal con baja densidad podría estar en una zona poco poblada.
    
    Parámetros:
        clientes_df: DataFrame de clientes
        sucursales_df: DataFrame de sucursales
    
    Returns:
        pd.DataFrame: DataFrame con densidad de clientes por sucursal
    """
    agrupaciones = agrupar_clientes_por_proximidad(clientes_df, sucursales_df)
    
    densidades = []
    for idx_sucursal, clientes_índices in agrupaciones.items():
        densidades.append({
            'Sucursal_Índice': idx_sucursal,
            'Cantidad_Clientes': len(clientes_índices),
            'Densidad': len(clientes_índices) / max(1, len(clientes_df))  # Proporcional
        })
    
    return pd.DataFrame(densidades)


def calcular_cobertura_geográfica(clientes_df, cajeros_df, sucursales_df, 
                                   umbral_sucursal=10.0, umbral_cajero=5.0):
    """
    Realiza un análisis completo de cobertura geográfica.
    
    Determina qué porcentaje de clientes está cubierto (dentro del umbral)
    tanto por sucursales como por cajeros automáticos.
    
    Parámetros:
        clientes_df: DataFrame de clientes
        cajeros_df: DataFrame de cajeros
        sucursales_df: DataFrame de sucursales
        umbral_sucursal: Distancia máxima a sucursal (km)
        umbral_cajero: Distancia máxima a cajero (km)
    
    Returns:
        dict: Diccionario con métricas de cobertura
    """
    clientes = clientes_df.copy()
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales_df)
    clientes = calcular_distancia_a_cajero_mas_cercano(clientes, cajeros_df)
    
    cobertura_sucursales = (clientes['Distancia_a_Sucursal_km'] <= umbral_sucursal).sum() / len(clientes)
    cobertura_cajeros = (clientes['Distancia_a_Cajero_km'] <= umbral_cajero).sum() / len(clientes)
    cobertura_completa = ((clientes['Distancia_a_Sucursal_km'] <= umbral_sucursal) & 
                          (clientes['Distancia_a_Cajero_km'] <= umbral_cajero)).sum() / len(clientes)
    
    return {
        'cobertura_sucursales_pct': cobertura_sucursales * 100,
        'cobertura_cajeros_pct': cobertura_cajeros * 100,
        'cobertura_completa_pct': cobertura_completa * 100,
        'clientes_con_cobertura_completa': int(cobertura_completa * len(clientes)),
        'clientes_sin_cobertura_completa': int((1 - cobertura_completa) * len(clientes))
    }