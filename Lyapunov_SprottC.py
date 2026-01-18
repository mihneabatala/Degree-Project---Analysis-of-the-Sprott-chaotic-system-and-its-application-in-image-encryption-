import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===== Sistem Thomas =====
def thomas_system(t, state):
    x, y, z = state
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x**2
    return [dxdt, dydt, dzdt]

# ===== Configurație simulare =====
T_final = 100
dt = 0.01
t_eval = np.arange(0, T_final, dt)
t_span = (0, T_final)

# Condiții inițiale apropiate
delta0 = 1e-8  # perturbație inițială
y0 = [0.21, 0.1, 0.2]
y0_perturbed = [y0[0] + delta0, y0[1], y0[2]]

# Integrarea ambelor traiectorii
sol_1 = solve_ivp(lambda t, y: thomas_system(t, y), t_span, y0, t_eval=t_eval, method='RK45')
sol_2 = solve_ivp(lambda t, y: thomas_system(t, y), t_span, y0_perturbed, t_eval=t_eval, method='RK45')

# Calculul distanței în timp
diff = sol_2.y - sol_1.y
distance = np.linalg.norm(diff, axis=0)

# Eliminare puncte zero sau negative (log invalid)
valid = distance > 0
log_distance = np.log(distance[valid])
time_valid = t_eval[valid]

# Estimare panta (lambda) prin regresie liniară
slope, _ = np.polyfit(time_valid, log_distance, 1)

# ===== Plot =====
plt.figure(figsize=(8, 5))
plt.plot(time_valid, log_distance, label='log(Δ(t))')
plt.plot(time_valid, slope * time_valid + log_distance[0], '--', label=f'λ ≈ {slope:.4f}')
plt.title('Estimare exponent Lyapunov pentru sistemul Thomas')
plt.xlabel('Timp (s)')
plt.ylabel('log(Δ(t))')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Afișare rezultat =====
print(f"Exponentul Lyapunov estimat: λ ≈ {slope:.4f}")
