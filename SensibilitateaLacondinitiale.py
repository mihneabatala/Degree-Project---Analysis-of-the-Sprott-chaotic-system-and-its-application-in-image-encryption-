import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Parametri inițiali
Fs = 100  # Hz
t_start = 0
t_end = 200
t_eval = np.linspace(t_start, t_end, int((t_end - t_start) * Fs))

# 2. Două condiții inițiale aproape identice
initial_condition1 = np.random.randn(3)
initial_condition2 = initial_condition1 + 1e-12

# 3. Sistemul Thomas (b = 0.18)
# b = 0.18
def thomas(t, state):
    x, y, z = state
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x**2
    return [dxdt, dydt, dzdt]

# 4. Integrarea sistemului
sol1 = solve_ivp(thomas, [t_start, t_end], initial_condition1, t_eval=t_eval)
sol2 = solve_ivp(thomas, [t_start, t_end], initial_condition2, t_eval=t_eval)

# 5. Calculul diferențelor între traiectorii
diff_x = sol1.y[0] - sol2.y[0]
diff_y = sol1.y[1] - sol2.y[1]
diff_z = sol1.y[2] - sol2.y[2]

# 6. Plot sensibilitate
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t_eval, diff_x)
plt.title("Sensibilitatea X(t) la condiții inițiale")
plt.xlabel("Timp (s)")
plt.ylabel("ΔX")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_eval, diff_y)
plt.title("Sensibilitatea Y(t) la condiții inițiale")
plt.xlabel("Timp (s)")
plt.ylabel("ΔY")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_eval, diff_z)
plt.title("Sensibilitatea Z(t) la condiții inițiale")
plt.xlabel("Timp (s)")
plt.ylabel("ΔZ")
plt.grid(True)

plt.tight_layout()
plt.show()
