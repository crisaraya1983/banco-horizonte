import streamlit as st
from typing import Optional, Dict, Any

def inyectar_estilos_globales():
    """
    Inyecta estilos CSS globales que se aplican a toda la aplicación.
    """
    st.markdown("""
    <style>
    /* Variables y colores */
    :root {
        --primary-color: #2c5aa0;
        --dark-primary: #1a365d;
        --light-primary: #e8f1f8;
        --accent-color: #3498db;
        --success-color: #27ae60;
        --warning-color: #f39c12;
        --error-color: #e74c3c;
        --info-color: #3498db;
        --text-primary: #2d3748;
        --text-secondary: #718096;
        --bg-light: #f8f9fa;
        --bg-white: #ffffff;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Tipografía base */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        color: var(--text-primary);
        background-color: var(--bg-white);
    }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-light);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    </style>
    """, unsafe_allow_html=True)


def crear_seccion_encabezado(titulo: str, descripcion: str = "", 
                             badge: Optional[str] = None, 
                             badge_color: str = "primary"):
    color_badges = {
        "primary": "#2c5aa0",
        "success": "#27ae60",
        "warning": "#f39c12",
        "error": "#e74c3c"
    }
    
    badge_bg = color_badges.get(badge_color, "#2c5aa0")
    
    html_lines = [
        '<div style="margin-bottom: 24px;">',
        '  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">',
        '    <h2 style="color: var(--dark-primary); font-size: 1.8em; font-weight: 600; margin: 0; letter-spacing: 0.3px;">',
        f'      {titulo}',
        '    </h2>',
    ]
    
    if badge:
        html_lines.append(f"""
    <span style="display: inline-block; background: {badge_bg}; color: white; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
      {badge}
    </span>""")
    
    html_lines.extend([
        '  </div>',
    ])
    
    if descripcion:
        html_lines.append(f'  <p style="color: var(--text-secondary); font-size: 14px; margin: 0; line-height: 1.5;">{descripcion}</p>')
    
    html_lines.append('</div>')
    
    html_content = '\n'.join(html_lines)
    st.markdown(html_content, unsafe_allow_html=True)


def crear_linea_separadora(estilo: str = "subtle"):
    if estilo == "subtle":
        st.markdown(
            '<div style="height: 1px; background: linear-gradient(to right, transparent, #e2e8f0, transparent); margin: 20px 0;"></div>',
            unsafe_allow_html=True
        )
    else:
        st.divider()


def crear_tarjeta_metrica(titulo: str, valor: str, subtitulo: str = "", 
                         icono: str = "", color_fondo: str = "light"):
    color_map = {
        "light": "var(--light-primary)",
        "primary": "var(--primary-color)",
        "success": "var(--success-color)",
        "warning": "var(--warning-color)",
        "error": "var(--error-color)"
    }
    
    bg_color = color_map.get(color_fondo, "var(--light-primary)")
    
    html_lines = [
        '<div style="background: var(--bg-white); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: var(--shadow-sm);">',
        '  <div style="display: flex; align-items: flex-start; gap: 15px;">',
    ]
    
    if icono:
        html_lines.append(f"""    <div style="background: {bg_color}; padding: 12px; border-radius: 8px; font-size: 24px; line-height: 1;">
      {icono}
    </div>""")
    
    html_lines.extend([
        '    <div style="flex: 1;">',
        f'      <div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">{titulo}</div>',
        f'      <div style="font-size: 28px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">{valor}</div>',
    ])
    
    if subtitulo:
        html_lines.append(f'      <div style="font-size: 13px; color: var(--text-secondary);">{subtitulo}</div>')
    
    html_lines.extend([
        '    </div>',
        '  </div>',
        '</div>'
    ])
    
    html_content = '\n'.join(html_lines)
    st.markdown(html_content, unsafe_allow_html=True)


def crear_tarjeta_informativa(titulo: str, contenido: str, tipo: str = "info", icono: str = "ℹ️"):
    color_map = {
        "info": {"bg": "#e8f1f8", "border": "#3498db", "icon_color": "#2c5aa0"},
        "success": {"bg": "#eafaf1", "border": "#27ae60", "icon_color": "#27ae60"},
        "warning": {"bg": "#fef5e7", "border": "#f39c12", "icon_color": "#f39c12"},
        "error": {"bg": "#fadbd8", "border": "#e74c3c", "icon_color": "#e74c3c"}
    }
    
    colores = color_map.get(tipo, color_map["info"])
    
    html_content = f"""
    <div style="background: {colores['bg']}; border-left: 4px solid {colores['border']}; border-radius: 8px; padding: 16px; margin: 12px 0; display: flex; gap: 12px;">
        <div style="font-size: 24px; line-height: 1.4; color: {colores['icon_color']};">
            {icono}
        </div>
        <div>
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
                {titulo}
            </div>
            <div style="font-size: 14px; color: var(--text-secondary); line-height: 1.5;">
                {contenido}
            </div>
        </div>
    </div>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)


def crear_indicador_estado(valor: float, minimo: float = 0, maximo: float = 100, 
                          etiqueta: str = "Progreso", mostrar_porcentaje: bool = True):
    rango = maximo - minimo
    porcentaje = ((valor - minimo) / rango) * 100
    porcentaje = max(0, min(100, porcentaje))
    
    if porcentaje >= 80:
        color = "var(--success-color)"
    elif porcentaje >= 50:
        color = "var(--accent-color)"
    else:
        color = "var(--warning-color)"
    
    html_content = f"""
    <div style="margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: var(--text-primary); font-size: 14px;">
                {etiqueta}
            </span>
            {f'<span style="color: {color}; font-weight: 700; font-size: 14px;">{porcentaje:.1f}%</span>' if mostrar_porcentaje else ''}
        </div>
        <div style="background: var(--bg-light); border-radius: 8px; height: 8px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);">
            <div style="background: {color}; height: 100%; width: {porcentaje}%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); animation: slideIn 0.6s ease-out;"></div>
        </div>
    </div>
    
    <style>
    @keyframes slideIn {{
        from {{
            width: 0%;
        }}
        to {{
            width: {porcentaje}%;
        }}
    }}
    </style>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)


def crear_panel_estadisticas(estadisticas: Dict[str, Any]):
    html_lines = [
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 16px 0;">',
    ]
    
    for label, data in estadisticas.items():
        valor = data.get("valor", "N/A")
        cambio = data.get("cambio", "")
        tipo = data.get("tipo", "info")
        
        color_tipo = {
            "success": "#27ae60",
            "warning": "#f39c12",
            "error": "#e74c3c",
            "info": "#3498db"
        }.get(tipo, "#3498db")
        
        html_lines.append(
            f'  <div style="background: var(--bg-white); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; text-align: center; box-shadow: var(--shadow-sm);">'
        )
        html_lines.append(
            f'    <div style="font-size: 13px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 8px;">{label}</div>'
        )
        html_lines.append(
            f'    <div style="font-size: 28px; font-weight: 700; color: var(--text-primary);">{valor}</div>'
        )
        
        if cambio:
            html_lines.append(
                f'    <div style="color: {color_tipo}; font-weight: 600; font-size: 12px; margin-top: 8px;">{cambio}</div>'
            )
        
        html_lines.append('  </div>')
    
    html_lines.append('</div>')
    
    html_content = '\n'.join(html_lines)
    st.markdown(html_content, unsafe_allow_html=True)

def inicializar_componentes():
    if 'componentes_inicializados' not in st.session_state:
        inyectar_estilos_globales()
        st.session_state.componentes_inicializados = True