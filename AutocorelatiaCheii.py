import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ====== 1. Sistemul Sprott C ======
def sprott_system(t, state):
    x, y, z = state
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x**2
    return [dxdt, dydt, dzdt]

# ====== 2. Integrarea sistemului ======
T_final = 2800
dt = 0.01
t_eval = np.arange(0, T_final, dt)
state_initial = np.random.randn(3)

sol = solve_ivp(lambda t, y: sprott_system(t, y), (0, T_final), state_initial, t_eval=t_eval)
x, y, z = sol.y

# Eliminăm faza tranzitorie
cut_index = int(100 / dt)
chaotic_seq = np.abs(x[cut_index:] + y[cut_index:] )
key_stream = (255 * chaotic_seq / np.max(chaotic_seq)).astype(np.uint8)

# ====== 3. Funcția pentru autocorelare ======
def autocorrelation(x, lag):
    n = len(x)
    x_mean = np.mean(x)
    numerator = np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean))
    denominator = np.sum((x - x_mean)**2)
    return numerator / denominator

# ====== 4. Calcul autocorelare pentru laguri 1–50 ======
lags = np.arange(1, 150)
autocorr_values = [autocorrelation(key_stream, lag) for lag in lags]

# ====== 5. Afișare grafic ======
plt.figure(figsize=(10, 4))
plt.plot(lags, autocorr_values, marker='o', color='orange')
plt.title("Autocorelarea cheii haotice (Sprott C)")
plt.xlabel("Lag")
plt.ylabel("Coeficient de autocorelare")
plt.grid(True)
plt.tight_layout()
plt.show()
