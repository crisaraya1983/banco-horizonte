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


def inicializar_componentes():
    if 'componentes_inicializados' not in st.session_state:
        inyectar_estilos_globales()
        st.session_state.componentes_inicializados = True