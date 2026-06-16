"""
PROYECTO DE CÁLCULO NUMÉRICO
Universidad Metropolitana
Simulación de Proyectil con Resistencia Aerodinámica

Modelo físico:
  F_d = (1/2) * rho * Cd * A * v^2
  
Ecuaciones de movimiento (sistema de EDOs):
  dx/dt = vx
  dy/dt = vy
  dvx/dt = -(k/m) * v * vx       donde k = (1/2)*rho*Cd*A
  dvy/dt = -g - (k/m) * v * vy   v = sqrt(vx^2 + vy^2)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ─────────────────────────────────────────────────────────────
# 1. PARÁMETROS FÍSICOS
# ─────────────────────────────────────────────────────────────
g    = 9.81          # gravedad (m/s^2)
m    = 0.145         # masa del proyectil (kg) – tipo pelota de béisbol
r    = 0.037         # radio (m)
A    = np.pi * r**2  # área transversal (m^2)
Cd   = 0.47          # coeficiente de arrastre (esfera)
v0   = 50.0          # velocidad inicial (m/s)
th0  = 45.0          # ángulo de elevación (grados)
dt   = 0.001         # paso temporal (s) – elegido para estabilidad RK4

# Densidades atmosféricas (3 escenarios para análisis de sensibilidad)
escenarios = {
    "Alta altitud (~3000 m)": 0.909,
    "Nivel del mar (estándar)": 1.225,
    "Atmósfera densa (~-500 m)": 1.500,
}

# ─────────────────────────────────────────────────────────────
# 2. IMPLEMENTACIÓN RK4
# ─────────────────────────────────────────────────────────────

def derivadas(estado, k_over_m):
    """
    Calcula las derivadas del vector de estado [x, y, vx, vy].
    
    Física:
      - La fuerza de arrastre actúa en dirección opuesta al vector velocidad.
      - F_drag = k * v^2 → aceleración = (k/m) * v, en dirección -v_hat
      - ax = -(k/m) * |v| * vx
      - ay = -g - (k/m) * |v| * vy
    """
    x, y, vx, vy = estado
    v_mag = np.sqrt(vx**2 + vy**2)   # magnitud de la velocidad escalar
    
    dxdt  = vx
    dydt  = vy
    dvxdt = -k_over_m * v_mag * vx
    dvydt = -g - k_over_m * v_mag * vy
    
    return np.array([dxdt, dydt, dvxdt, dvydt])


def rk4_paso(estado, dt, k_over_m):
    """
    Avanza un paso temporal dt usando Runge-Kutta de 4to orden.
    
    Fórmula:
      k1 = f(t, y)
      k2 = f(t + dt/2, y + dt/2 * k1)
      k3 = f(t + dt/2, y + dt/2 * k2)
      k4 = f(t + dt,   y + dt   * k3)
      y_next = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = derivadas(estado,           k_over_m)
    k2 = derivadas(estado + dt/2*k1, k_over_m)
    k3 = derivadas(estado + dt/2*k2, k_over_m)
    k4 = derivadas(estado + dt   *k3, k_over_m)
    
    return estado + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simular(rho, v0, theta_deg, dt=0.001):
    """
    Simula la trayectoria completa del proyectil.
    Retorna arrays de (t, x, y, vx, vy).
    """
    k         = 0.5 * rho * Cd * A      # constante de arrastre total
    k_over_m  = k / m
    
    theta = np.radians(theta_deg)
    vx0   = v0 * np.cos(theta)
    vy0   = v0 * np.sin(theta)
    
    # Estado inicial: [x, y, vx, vy]
    estado = np.array([0.0, 0.0, vx0, vy0])
    
    trayectoria = [estado.copy()]
    t = 0.0
    tiempos = [t]
    
    # Integración hasta que y < 0 (proyectil aterriza)
    while True:
        estado_nuevo = rk4_paso(estado, dt, k_over_m)
        t += dt
        trayectoria.append(estado_nuevo.copy())
        tiempos.append(t)
        
        # Condición de parada: altura negativa (ya pasó el suelo)
        if estado_nuevo[1] < 0 and t > 0.1:
            break
        
        estado = estado_nuevo
    
    trayectoria = np.array(trayectoria)
    tiempos     = np.array(tiempos)
    
    return tiempos, trayectoria[:, 0], trayectoria[:, 1], trayectoria[:, 2], trayectoria[:, 3]


# ─────────────────────────────────────────────────────────────
# 3. CÁLCULO DE RAÍCES – Bisección para encontrar tf y R exactos
# ─────────────────────────────────────────────────────────────

def encontrar_alcance(rho, v0, theta_deg, dt=0.001, tol=1e-8):
    """
    Usa el Método de Bisección para encontrar con precisión el tiempo
    de vuelo tf en el que y(tf) = 0 (excluyendo t=0).
    
    Una vez hallado tf, el alcance R = x(tf).
    
    Justificación del método de bisección:
      - La función y(t) es continua.
      - Existe un intervalo [ta, tb] donde y(ta) > 0 y y(tb) < 0.
      - Por el Teorema del Valor Intermedio, existe exactamente un cero en ese intervalo.
    """
    k        = 0.5 * rho * Cd * A
    k_over_m = k / m
    
    theta = np.radians(theta_deg)
    vx0   = v0 * np.cos(theta)
    vy0   = v0 * np.sin(theta)
    
    def y_en_t(t_target):
        """Integra hasta t_target y retorna y(t_target)."""
        estado = np.array([0.0, 0.0, vx0, vy0])
        t = 0.0
        while t < t_target:
            paso = min(dt, t_target - t)
            estado = rk4_paso(estado, paso, k_over_m)
            t += paso
        return estado[0], estado[1]  # x, y
    
    # Encontrar intervalo inicial donde y cambia de signo
    # Simulación rápida para encontrar [ta, tb]
    estado = np.array([0.0, 0.0, vx0, vy0])
    t = 0.0
    ta = tb = None
    y_prev = 0.0
    t_prev = 0.0
    
    while True:
        estado = rk4_paso(estado, dt, k_over_m)
        t += dt
        y_curr = estado[1]
        
        if y_curr < 0 and t > 0.1:
            ta = t_prev
            tb = t
            break
        
        y_prev = y_curr
        t_prev = t
    
    # Bisección en [ta, tb]
    iteraciones = 0
    while (tb - ta) > tol:
        tc = (ta + tb) / 2.0
        _, y_c = y_en_t(tc)
        _, y_a = y_en_t(ta)
        
        if y_a * y_c < 0:
            tb = tc
        else:
            ta = tc
        iteraciones += 1
    
    tf = (ta + tb) / 2.0
    x_final, _ = y_en_t(tf)
    
    return tf, x_final, iteraciones


# ─────────────────────────────────────────────────────────────
# 4. TRAYECTORIA IDEAL (sin arrastre, k=0)
# ─────────────────────────────────────────────────────────────

def trayectoria_ideal(v0, theta_deg, num_puntos=1000):
    """
    Solución analítica del proyectil en el vacío:
      x(t) = v0*cos(θ)*t
      y(t) = v0*sin(θ)*t - (1/2)*g*t²
      
    Tiempo de vuelo:  tf = 2*v0*sin(θ)/g
    Alcance máximo:   R  = v0²*sin(2θ)/g
    """
    theta = np.radians(theta_deg)
    tf    = 2 * v0 * np.sin(theta) / g
    t     = np.linspace(0, tf, num_puntos)
    x     = v0 * np.cos(theta) * t
    y     = v0 * np.sin(theta) * t - 0.5 * g * t**2
    R     = v0**2 * np.sin(2*theta) / g
    return t, x, y, tf, R


# ─────────────────────────────────────────────────────────────
# 5. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("  PROYECTO CÁLCULO NUMÉRICO – PROYECTIL CON ARRASTRE (RK4)")
print("=" * 65)
print(f"\n  Parámetros: m={m} kg | r={r} m | Cd={Cd} | v₀={v0} m/s | θ₀={th0}°")
print(f"  Paso temporal dt = {dt} s (garantiza estabilidad RK4)\n")

# Caso principal: nivel del mar
rho_std = 1.225
t_arr, x_arr, y_arr, vx_arr, vy_arr = simular(rho_std, v0, th0, dt)

# Búsqueda de raíces exacta
tf_exact, R_exact, n_iter = encontrar_alcance(rho_std, v0, th0, dt)

# Trayectoria ideal
t_id, x_id, y_id, tf_id, R_id = trayectoria_ideal(v0, th0)

print("─" * 65)
print("  RESULTADOS (nivel del mar, ρ = 1.225 kg/m³)")
print("─" * 65)
print(f"  Tiempo de vuelo  tf  = {tf_exact:.4f} s  (bisección, {n_iter} iteraciones)")
print(f"  Alcance máximo   R   = {R_exact:.2f} m")
print(f"  Alcance ideal    R₀  = {R_id:.2f} m  (sin arrastre)")
print(f"  Reducción        ΔR  = {R_id - R_exact:.2f} m  ({(R_id-R_exact)/R_id*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 6. ANÁLISIS DE SENSIBILIDAD (3 densidades atmosféricas)
# ─────────────────────────────────────────────────────────────

print("\n─" * 33)
print("  ANÁLISIS DE SENSIBILIDAD – Variación de densidad atmosférica")
print("─" * 65)
print(f"  {'Escenario':<35} {'ρ (kg/m³)':>10} {'R (m)':>10} {'tf (s)':>8} {'ΔR%':>7}")
print("  " + "-" * 63)

resultados_sens = {}
for nombre, rho_val in escenarios.items():
    tf_s, R_s, _ = encontrar_alcance(rho_val, v0, th0, dt)
    delta_R = (R_id - R_s) / R_id * 100
    print(f"  {nombre:<35} {rho_val:>10.3f} {R_s:>10.2f} {tf_s:>8.4f} {delta_R:>7.1f}%")
    resultados_sens[nombre] = (rho_val, R_s, tf_s)

print(f"\n  {'Vacío (ideal)':<35} {'0.000':>10} {R_id:>10.2f} {tf_id:>8.4f} {'0.0%':>7}")

# ─────────────────────────────────────────────────────────────
# 7. ANÁLISIS GRÁFICO
# ─────────────────────────────────────────────────────────────

# Estilo académico limpio
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#FAFAFA')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

colores = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E']

# ── GRÁFICA 1: Trayectorias comparativas ──────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#F8F9FA')

# Trayectoria ideal
ax1.plot(x_id, y_id, '--', color='#FF7F0E', lw=2.2, label=f'Vacío (ideal)  R = {R_id:.1f} m', zorder=3)

# Trayectorias con arrastre para cada densidad
for i, (nombre, (rho_v, R_v, tf_v)) in enumerate(resultados_sens.items()):
    t_s, x_s, y_s, _, _ = simular(rho_v, v0, th0, dt)
    mask = y_s >= 0
    label = f'{nombre}  R = {R_v:.1f} m'
    ax1.plot(x_s[mask], y_s[mask], color=colores[i], lw=2, label=label, zorder=4)

# Punto de alcance máximo (nivel del mar)
ax1.axvline(R_exact, color='#1F77B4', ls=':', alpha=0.6, lw=1.2)
ax1.axhline(0, color='#333333', lw=1.0)

ax1.set_xlabel('Alcance horizontal x (m)', fontsize=12)
ax1.set_ylabel('Altura y (m)', fontsize=12)
ax1.set_title('Trayectorias del Proyectil: Modelo Real vs. Ideal\n'
              f'v₀ = {v0} m/s  |  θ₀ = {th0}°  |  m = {m} kg  |  Cd = {Cd}',
              fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=-2)

# Anotación del alcance
ax1.annotate(f'R = {R_exact:.1f} m\n(nivel del mar)',
             xy=(R_exact, 0), xytext=(R_exact - 25, 22),
             fontsize=8.5, color='#1F77B4',
             arrowprops=dict(arrowstyle='->', color='#1F77B4', lw=1.2))

# ── GRÁFICA 2: Velocidad vs Tiempo ────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#F8F9FA')

mask0 = y_arr >= 0
v_mag = np.sqrt(vx_arr**2 + vy_arr**2)
ax2.plot(t_arr[mask0], v_mag[mask0], color='#D62728', lw=2, label='|v| total')
ax2.plot(t_arr[mask0], vx_arr[mask0], color='#1F77B4', lw=1.6, ls='--', label='vₓ')
ax2.plot(t_arr[mask0], vy_arr[mask0], color='#2CA02C', lw=1.6, ls='-.', label='vy')
ax2.axhline(0, color='#666', lw=0.8)
ax2.set_xlabel('Tiempo (s)', fontsize=11)
ax2.set_ylabel('Velocidad (m/s)', fontsize=11)
ax2.set_title('Componentes de Velocidad vs Tiempo\n(nivel del mar)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)

# ── GRÁFICA 3: Sensibilidad del Alcance vs Densidad ───────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#F8F9FA')

rhos_scan = np.linspace(0.1, 2.0, 40)
R_scan = []
for rho_v in rhos_scan:
    _, R_v, _ = encontrar_alcance(rho_v, v0, th0, dt)
    R_scan.append(R_v)

ax3.plot(rhos_scan, R_scan, color='#9467BD', lw=2.2)
ax3.axhline(R_id, color='#FF7F0E', ls='--', lw=1.6, label=f'Vacío R₀ = {R_id:.1f} m')

# Marcar los 3 escenarios
for i, (nombre, (rho_v, R_v, _)) in enumerate(resultados_sens.items()):
    ax3.scatter(rho_v, R_v, color=colores[i], s=70, zorder=5, label=f'ρ={rho_v} → R={R_v:.0f} m')

ax3.set_xlabel('Densidad atmosférica ρ (kg/m³)', fontsize=11)
ax3.set_ylabel('Alcance R (m)', fontsize=11)
ax3.set_title('Análisis de Sensibilidad\nAlcance vs. Densidad Atmosférica', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8, loc='upper right')

# Nota metodológica
fig.text(0.5, 0.01,
         'Método numérico: Runge-Kutta 4º Orden  |  Búsqueda de raíces: Bisección  |  dt = 0.001 s',
         ha='center', fontsize=9, color='#555555', style='italic')

plt.show()
plt.savefig('/mnt/user-data/outputs/proyecto_calculo_numerico.png',
            dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
print("\n  ✓ Figura guardada correctamente.")
plt.close()
print("\n" + "=" * 65)
print("  PROYECTO COMPLETADO")
print("=" * 65)