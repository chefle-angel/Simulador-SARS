import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==============================================================================
# 1. MOTOR MATEMÁTICO (Traducción directa de tu RK4 y EDOs)
# ==============================================================================

def sistema_edo(t, y, p):
    """
    Sistema de Ecuaciones Diferenciales (SEIQRJ)
    y: [S, E, Q, I, J, R, C]
    """
    S, E, Q, I, J, R, C = y
    N = S + E + Q + I + J + R
    if N < 1: N = 1
    
    # Fuerza de infección
    force = (p['beta'] / N) * (I + p['epsE']*E + p['epsQ']*Q + p['epsJ']*J)
    new_inf = force * S
    
    # Ecuaciones
    dS = p['Pi'] - new_inf - p['mu']*S
    dE = p['p'] + new_inf - (p['u1'] + p['k1'] + p['mu'])*E
    dQ = p['u1']*E - (p['k2'] + p['mu'])*Q
    dI = p['k1']*E - (p['u2'] + p['d1'] + p['s1'] + p['mu'])*I
    dJ = p['u2']*I + p['k2']*Q - (p['d2'] + p['s2'] + p['mu'])*J
    dR = p['s1']*I + p['s2']*J - p['mu']*R
    dC = new_inf # Casos acumulados para estadísticas
    
    return np.array([dS, dE, dQ, dI, dJ, dR, dC])

def rk4_solver(fun, t_span, y0, p):
    """
    Resolutor Runge-Kutta 4 manual (Igual que en tu MATLAB)
    """
    n = len(t_span)
    num_vars = len(y0)
    Y = np.zeros((num_vars, n))
    Y[:, 0] = y0
    
    dt = t_span[1] - t_span[0]
    y = y0
    
    for i in range(n - 1):
        t = t_span[i]
        k1 = fun(t, y, p)
        k2 = fun(t + 0.5*dt, y + 0.5*dt*k1, p)
        k3 = fun(t + 0.5*dt, y + 0.5*dt*k2, p)
        k4 = fun(t + dt, y + dt*k3, p)
        
        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        Y[:, i + 1] = y
        
    return Y

# ==============================================================================
# 2. INTERFAZ GRÁFICA (Streamlit)
# ==============================================================================

# Configuración de la página (Título e icono)
st.set_page_config(page_title="Simulador SARS", layout="wide", page_icon="🦠")

st.title("🦠 Simulador Epidemiológico Avanzado")
st.markdown("Basado en modelo **SEIQRJ** con control de parámetros en tiempo real.")

# --- SIDEBAR: CONTROLES (Tus sliders) ---
st.sidebar.header("⚙️ Parámetros")

# Grupo 1: Población Inicial
with st.sidebar.expander("1. Población Inicial", expanded=True):
    S0 = st.number_input("Susceptibles (S0)", value=12000000, step=1000)
    col1, col2 = st.columns(2)
    E0 = col1.number_input("Expuestos (E0)", value=1565)
    I0 = col2.number_input("Infectados (I0)", value=695)
    Q0 = col1.number_input("Cuarentena (Q0)", value=292)
    J0 = col2.number_input("Aislados (J0)", value=326)
    R0 = st.number_input("Recuperados (R0)", value=20)

# Grupo 2: Parámetros Biológicos
with st.sidebar.expander("2. Biología del Virus"):
    beta = st.slider("Contagio (β)", 0.0, 2.0, 0.2, 0.01)
    mu = st.slider("Tasa Muerte Nat. (μ)", 0.0, 0.0001, 0.000034, format="%.6f")
    
    st.markdown("---")
    st.markdown("**Infecciosidad Relativa:**")
    epsE = st.slider("Expuestos (εE)", 0.0, 1.0, 0.3)
    epsQ = st.slider("Cuarentena (εQ)", 0.0, 1.0, 0.0)
    epsJ = st.slider("Aislados (εJ)", 0.0, 1.0, 0.1)
    
    st.markdown("---")
    st.markdown("**Tiempos:**")
    k1 = st.slider("Incubación (k1)", 0.01, 1.0, 0.1)
    k2 = st.slider("Síntomas Q (k2)", 0.01, 1.0, 0.125)

# Grupo 3: Intervención y Tasas
with st.sidebar.expander("3. Gestión Clínica"):
    days = st.slider("Días a simular", 100, 1000, 365)
    p_inm = st.slider("Inmigración (p)", 0, 12000, 0)
    
    col3, col4 = st.columns(2)
    d1 = col3.slider("Letalidad I (δ1)", 0.0, 0.1, 0.0079, format="%.4f")
    d2 = col4.slider("Letalidad J (δ2)", 0.0, 0.1, 0.0068, format="%.4f")
    
    s1 = col3.slider("Recup. I (σ1)", 0.0, 0.5, 0.0337, format="%.4f")
    s2 = col4.slider("Recup. J (σ2)", 0.0, 0.5, 0.0386, format="%.4f")
    
    u1 = st.slider("Rastreo (γ1)", 0.0, 1.0, 0.2)
    u2 = st.slider("Aislar (γ2)", 0.0, 1.0, 0.2)

# ==============================================================================
# 3. EJECUCIÓN DEL MODELO
# ==============================================================================

# Preparar parámetros en un diccionario
N_ini = S0 + E0 + I0 + Q0 + J0 + R0
params = {
    'beta': beta, 'mu': mu, 'Pi': mu * N_ini, 'p': p_inm,
    'epsE': epsE, 'epsQ': epsQ, 'epsJ': epsJ,
    'k1': k1, 'k2': k2,
    'd1': d1, 'd2': d2,
    's1': s1, 's2': s2,
    'u1': u1, 'u2': u2
}

# Vector inicial y tiempo
y0 = np.array([S0, E0, Q0, I0, J0, R0, 0.0]) # El último es C (Acumulado)
t_span = np.linspace(0, days, 600)

# Resolver
Y = rk4_solver(sistema_edo, t_span, y0, params)

# Extraer resultados
S, E, Q, I, J, R, C = Y
N_total = S + E + Q + I + J + R
C_total_real = C + (E0 + Q0 + I0 + J0) # Ajuste de acumulados iniciales

# ==============================================================================
# 4. VISUALIZACIÓN (Gráficas interactivas con Plotly)
# ==============================================================================

# Crear pestañas para las vistas (Lineal vs Logarítmica)
tab1, tab2, tab3 = st.tabs(["📈 Dinámica (Lineal)", "📊 Escala Logarítmica", "📋 Datos Detallados"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_span, y=S, name='Susceptibles', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=t_span, y=E, name='Expuestos', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=t_span, y=I, name='Infectados', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=t_span, y=Q, name='Cuarentena', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t_span, y=J, name='Aislados', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=t_span, y=R, name='Recuperados', line=dict(color='green', dash='dash')))
    
    fig.update_layout(title="Evolución de la Epidemia", xaxis_title="Días", yaxis_title="Población", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig_log = go.Figure()
    fig_log.add_trace(go.Scatter(x=t_span, y=E, name='Expuestos', line=dict(color='orange')))
    fig_log.add_trace(go.Scatter(x=t_span, y=I, name='Infectados', line=dict(color='red')))
    fig_log.add_trace(go.Scatter(x=t_span, y=Q, name='Cuarentena', line=dict(color='blue')))
    fig_log.add_trace(go.Scatter(x=t_span, y=J, name='Aislados', line=dict(color='purple')))
    
    fig_log.update_yaxes(type="log")
    fig_log.update_layout(title="Vista Logarítmica (Crecimiento exponencial)", xaxis_title="Días", hovermode="x unified")
    st.plotly_chart(fig_log, use_container_width=True)

with tab3:
    # Métricas clave (KPIs)
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Pico de Infectados", f"{int(max(I)):,}")
    col_kpi2.metric("Total Afectados", f"{int(C_total_real[-1]):,}")
    col_kpi3.metric("Muertes Totales", f"{int(N_ini - N_total[-1]):,}", delta_color="inverse")
    
    st.write("---")
    st.write("**Estado Final de la Simulación:**")
    final_data = {
        "Compartimento": ["Susceptibles", "Expuestos", "Cuarentena", "Infectados", "Aislados", "Recuperados"],
        "Población": [int(S[-1]), int(E[-1]), int(Q[-1]), int(I[-1]), int(J[-1]), int(R[-1])]
    }
    st.dataframe(pd.DataFrame(final_data))
