import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


def distancia_haversine(lat1, lon1, lat2, lon2):

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
    if 'Distancia_a_Sucursal_km' not in clientes_df.columns:
        clientes_df = calcular_distancia_a_sucursal_mas_cercana(clientes_df, sucursales_df)
    
    desatendidos = clientes_df[clientes_df['Distancia_a_Sucursal_km'] > umbral_km]
    return desatendidos


def crear_matriz_distancias(puntos_df):
    """
    CMatriz de distancias entre todos los puntos (cajeros o sucursales)
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
    """
    lat_prom = ubicaciones_df['Latitud'].mean()
    lon_prom = ubicaciones_df['Longitud'].mean()
    return lat_prom, lon_prom


def agrupar_clientes_por_proximidad(clientes_df, sucursales_df):
    """
    Agrupa clientes por su sucursal más cercana.
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
    """
    agrupaciones = agrupar_clientes_por_proximidad(clientes_df, sucursales_df)
    
    densidades = []
    for idx_sucursal, clientes_índices in agrupaciones.items():
        densidades.append({
            'Sucursal_Índice': idx_sucursal,
            'Cantidad_Clientes': len(clientes_índices),
            'Densidad': len(clientes_índices) / max(1, len(clientes_df))
        })
    
    return pd.DataFrame(densidades)


def calcular_cobertura_geográfica(clientes_df, cajeros_df, sucursales_df, 
                                   umbral_sucursal=10.0, umbral_cajero=5.0):
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

def calcular_rutas_mantenimiento(sucursales_df):

    principales = sucursales_df[sucursales_df['Tipo de Sucursal'] == 'Sucursal Principal'].copy()
    secundarias = sucursales_df[sucursales_df['Tipo de Sucursal'] == 'Sucursal Secundaria'].copy()
    
    rutas = []
    
    # Para cada sucursal secundaria, encontrar la principal más cercana
    for idx_sec, secundaria in secundarias.iterrows():
        distancia_minima = float('inf')
        principal_mas_cercana = None
        
        for idx_prin, principal in principales.iterrows():
            distancia = distancia_haversine(
                secundaria['Latitud'], secundaria['Longitud'],
                principal['Latitud'], principal['Longitud']
            )
            
            if distancia < distancia_minima:
                distancia_minima = distancia
                principal_mas_cercana = principal
        
        if principal_mas_cercana is not None:
            # Estimar tiempo de viaje (asumiendo 40 km/h promedio)
            tiempo_estimado = (distancia_minima / 40) * 60  # en minutos
            
            rutas.append({
                'Sucursal_Origen': principal_mas_cercana['Nombre'],
                'Tipo_Origen': principal_mas_cercana['Tipo de Sucursal'],
                'Lat_Origen': principal_mas_cercana['Latitud'],
                'Lon_Origen': principal_mas_cercana['Longitud'],
                'Sucursal_Destino': secundaria['Nombre'],
                'Tipo_Destino': secundaria['Tipo de Sucursal'],
                'Lat_Destino': secundaria['Latitud'],
                'Lon_Destino': secundaria['Longitud'],
                'Distancia_km': round(distancia_minima, 2),
                'Tiempo_Estimado_min': int(tiempo_estimado)
            })
    
    return pd.DataFrame(rutas)