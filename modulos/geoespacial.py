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

def identificar_riesgos_geoespaciales(sucursales_df, datos_consolidados, productos_df):
    """
    Identifica riesgos geoespaciales usando datos consolidados reales
    """
    riesgos = []
    
    # Definir límites geográficos (Costa Rica: lat -11 a -6, lon -82 a -77)
    LAT_MIN, LAT_MAX = -11, -6
    LON_MIN, LON_MAX = -82, -77
    
    for idx, sucursal in sucursales_df.iterrows():
        lat, lon = sucursal['Latitud'], sucursal['Longitud']
        nombre = sucursal['Nombre']
        
        # RIESGO 1: Ubicación fuera del territorio
        fuera_territorio = not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX)
        
        # RIESGO 2: Diversificación de productos
        productos_sucursal = productos_df[
            productos_df['Sucursal Donde Se Ofrece'] == sucursal['Tipo de Sucursal']
        ]
        diversificacion = len(productos_sucursal) if len(productos_sucursal) > 0 else 1
        
        # RIESGO 3: Aislamiento geográfico
        distancias_a_otras = []
        for idx2, otra in sucursales_df.iterrows():
            if idx != idx2:
                dist = distancia_haversine(lat, lon, otra['Latitud'], otra['Longitud'])
                distancias_a_otras.append(dist)
        
        distancia_minima = min(distancias_a_otras) if distancias_a_otras else float('inf')
        aislamiento = distancia_minima > 25  # > 25 km es riesgo
        
        # RIESGO 4: Volumen de clientes y ventas en la sucursal
        datos_suc = datos_consolidados[datos_consolidados['Nombre'] == nombre]
        
        if len(datos_suc) > 0:
            num_clientes = datos_suc['Numero_Clientes_Producto'].sum()
            volumen_ventas = datos_suc['Volumen_Ventas_Producto'].sum()
            volumen_transacciones = datos_suc['Volumen_Transacciones_Sucursal'].iloc[0]
            empleados = datos_suc['Número de Empleados'].iloc[0]
        else:
            num_clientes = 0
            volumen_ventas = 0
            volumen_transacciones = 0
            empleados = 1
        
        # RIESGO 5: Baja actividad (volumen de clientes bajo o baja productividad)
        clientes_promedio = datos_consolidados['Numero_Clientes_Producto'].mean()
        ventas_promedio = datos_consolidados['Volumen_Ventas_Producto'].mean()
        
        baja_actividad_clientes = num_clientes < clientes_promedio * 0.5
        baja_actividad_ventas = volumen_ventas < ventas_promedio * 0.5
        baja_actividad = baja_actividad_clientes or baja_actividad_ventas
        
        # RIESGO 6: Eficiencia (transacciones por empleado)
        trans_por_empleado = volumen_transacciones / empleados if empleados > 0 else 0
        trans_promedio = datos_consolidados['Volumen_Transacciones_Sucursal'].sum() / datos_consolidados['Número de Empleados'].sum()
        baja_eficiencia = trans_por_empleado < trans_promedio * 0.6
        
        # Calcular nivel de riesgo general (escala 0-100)
        riesgo_score = sum([
            fuera_territorio * 45,           # Crítico
            (diversificacion == 1) * 20,     # Dependencia de producto
            aislamiento * 15,                # Aislamiento
            baja_actividad_clientes * 25,    # Pocos clientes
            baja_actividad_ventas * 25,      # Pocas ventas
            baja_eficiencia * 10             # Baja eficiencia
        ])
        
        riesgo_score = min(100, riesgo_score)  # Máximo 100
        
        # Clasificación
        nivel_riesgo = (
            "🔴 Muy Alto" if riesgo_score >= 80 else
            "🟠 Alto" if riesgo_score >= 50 else
            "🟡 Medio" if riesgo_score >= 30 else
            "🟢 Bajo"
        )
        
        riesgos.append({
            'Sucursal': nombre,
            'Latitud': lat,
            'Longitud': lon,
            'Tipo de Sucursal': sucursal['Tipo de Sucursal'],
            'Fuera_Territorio': fuera_territorio,
            'Productos_Oferecidos': diversificacion,
            'Distancia_Sucursal_Mas_Cercana_km': round(distancia_minima, 2),
            'Aislamiento': aislamiento,
            'Clientes_Totales': int(num_clientes),
            'Volumen_Ventas_Total': int(volumen_ventas),
            'Volumen_Transacciones': int(volumen_transacciones),
            'Empleados': int(empleados),
            'Trans_Por_Empleado': round(trans_por_empleado, 2),
            'Baja_Actividad': baja_actividad,
            'Baja_Eficiencia': baja_eficiencia,
            'Riesgo_Score': riesgo_score,
            'Nivel_Riesgo': nivel_riesgo
        })
    
    return pd.DataFrame(riesgos)


def identificar_ubicaciones_optimas_sucursales(clientes_df, sucursales_df, datos_consolidados, n_clusters=3):
    """
    Identifica ubicaciones óptimas para nuevas sucursales usando clustering de clientes sin cobertura
    """
    from sklearn.cluster import KMeans
    
    clientes = clientes_df.copy()
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales_df)
    
    # Filtrar clientes sin cobertura (> 15 km)
    clientes_sin_cobertura = clientes[clientes['Distancia_a_Sucursal_km'] > 15].copy()
    
    if len(clientes_sin_cobertura) < 3:
        return pd.DataFrame()
    
    # Ponderación por valor (saldo * frecuencia)
    clientes_sin_cobertura['Valor'] = (
        clientes_sin_cobertura['Saldo Promedio de Cuentas'] * 
        clientes_sin_cobertura['Frecuencia de Visitas']
    )
    
    # Clustering basado en ubicación, ponderado por valor
    coords = clientes_sin_cobertura[['Latitud', 'Longitud']].values
    pesos = clientes_sin_cobertura['Valor'].values / clientes_sin_cobertura['Valor'].sum()
    
    kmeans = KMeans(n_clusters=min(n_clusters, len(clientes_sin_cobertura)), 
                    random_state=42, n_init=10)
    clientes_sin_cobertura['Cluster'] = kmeans.fit_predict(coords)
    
    # Calcular centroide y valor de cada cluster
    ubicaciones_optimas = []
    
    for cluster_id in range(kmeans.n_clusters):
        datos_cluster = clientes_sin_cobertura[clientes_sin_cobertura['Cluster'] == cluster_id]
        
        # Centroide ponderado por valor
        lat_opt = (datos_cluster['Latitud'] * datos_cluster['Valor']).sum() / datos_cluster['Valor'].sum()
        lon_opt = (datos_cluster['Longitud'] * datos_cluster['Valor']).sum() / datos_cluster['Valor'].sum()
        
        num_clientes = len(datos_cluster)
        valor_total = datos_cluster['Valor'].sum()
        saldo_promedio = datos_cluster['Saldo Promedio de Cuentas'].mean()
        
        # Encontrar distancia a sucursal más cercana
        dist_minima = datos_cluster['Distancia_a_Sucursal_km'].min()
        
        # Demanda de productos
        productos_demandados = datos_cluster['Productos Financieros Adquiridos'].value_counts()
        producto_principal = productos_demandados.index[0] if len(productos_demandados) > 0 else "General"
        
        ubicaciones_optimas.append({
            'Cluster_ID': cluster_id + 1,
            'Latitud': lat_opt,
            'Longitud': lon_opt,
            'Clientes_Sin_Cobertura': num_clientes,
            'Valor_Total': int(valor_total),
            'Saldo_Promedio': int(saldo_promedio),
            'Distancia_Sucursal_Cercana_km': round(dist_minima, 2),
            'Producto_Principal': producto_principal,
            'Potencial': 'Alto' if valor_total > datos_consolidados['Volumen_Ventas_Producto'].median() * 5 else 'Medio'
        })
    
    return pd.DataFrame(ubicaciones_optimas)


def calcular_cobertura_vs_demanda(clientes_df, sucursales_df, umbral_km=15.0):
    """
    Identifica zonas con alta demanda pero cobertura insuficiente
    """
    clientes = clientes_df.copy()
    clientes = calcular_distancia_a_sucursal_mas_cercana(clientes, sucursales_df)
    
    # Segmentar clientes por cobertura
    bien_cubiertos = clientes[clientes['Distancia_a_Sucursal_km'] <= umbral_km]
    mal_cubiertos = clientes[clientes['Distancia_a_Sucursal_km'] > umbral_km]
    
    oportunidades = []
    
    # Identificar zonas de alto valor pero mal cubiertas
    for idx, cliente_mal in mal_cubiertos.iterrows():
        saldo = cliente_mal['Saldo Promedio de Cuentas']
        frecuencia = cliente_mal['Frecuencia de Visitas']
        
        # Alto valor + baja cobertura = oportunidad
        if saldo > clientes['Saldo Promedio de Cuentas'].median() and frecuencia > 2:
            oportunidades.append({
                'Latitud': cliente_mal['Latitud'],
                'Longitud': cliente_mal['Longitud'],
                'Saldo': saldo,
                'Frecuencia_Visitas': frecuencia,
                'Distancia_Sucursal': cliente_mal['Distancia_a_Sucursal_km'],
                'Tipo_Oportunidad': 'Alto Valor - Cobertura Insuficiente'
            })
    
    return pd.DataFrame(oportunidades), len(bien_cubiertos), len(mal_cubiertos)