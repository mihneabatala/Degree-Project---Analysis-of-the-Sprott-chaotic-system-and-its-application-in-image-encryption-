import numpy as np
from scipy.integrate import solve_ivp
import cv2
import matplotlib.pyplot as plt
from scipy.stats import entropy, chisquare

# ====== 1. Sistemul Thomas ======
b = 0.19

def thomas_system(t, state):
    x, y, z = state
    dxdt = np.sin(y) - b * x
    dydt = np.sin(z) - b * y
    dzdt = np.sin(x) - b * z
    return [dxdt, dydt, dzdt]

# ====== 2. Integrarea și generarea cheii haotice (Thomas) ======
T_final = 3300
dt = 0.01
t_eval = np.arange(0, T_final, dt)
state_initial = np.random.randn(3)

# Integrează sistemul Thomas
sol = solve_ivp(thomas_system, (0, T_final), state_initial, t_eval=t_eval)
x, y, z = sol.y

# Eliminare tranzitor: primele 500 s
cut_index = int(700 / dt)  # 500 / 0.01 = 50 000  
x, y, z = x[cut_index:], y[cut_index:], z[cut_index:]

# ====== 3. Încărcare imagine și pregătire ======
image = cv2.imread('fish.jpg')
if image is None:
    raise ValueError("Imaginea nu a fost încărcată corect!")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = image_gray.shape
num_pixels = rows * cols
print(f"Imagine încărcată: {rows}x{cols} ({num_pixels} pixeli)")

# ====== 4. Generarea cheii din cos(x), cos(y), sin(z) ======
#    (folosim valorile soluției Thomas pentru x, y, z)
key_raw = x
key_scaled = (255 * key_raw / np.max(key_raw)).astype(np.uint8)

# Ajustăm lungimea cheii la numărul de pixeli
flat_key = np.resize(key_scaled, num_pixels).reshape((rows, cols))

# ====== 5. Criptare și decriptare cu XOR ======
encrypted_image = np.bitwise_xor(image_gray, flat_key)
decrypted_image = np.bitwise_xor(encrypted_image, flat_key)

# ====== 6. Afișare rezultate ======
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

axs[0].imshow(image_gray, cmap='gray')
axs[0].set_title('Imagine originală')
axs[0].axis('off')

axs[1].imshow(encrypted_image, cmap='gray')
axs[1].set_title('Imagine criptată (XOR cu cheia Thomas)')
axs[1].axis('off')

axs[2].imshow(decrypted_image, cmap='gray')
axs[2].set_title('Imagine decriptată')
axs[2].axis('off')

plt.tight_layout()
plt.show()

hist, _ = np.histogram(encrypted_image, bins=256, range=(0, 256), density=True)
entropy_value = entropy(hist, base=2)
print(f"Entropia informațională a cheii: {entropy_value:.4f} biți")

plt.figure(figsize=(10, 5))

plt.hist(encrypted_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Histogramă a valorilor imaginii criptate')
plt.xlabel('Valoare pixel')
plt.ylabel('Frecvență')
plt.grid(True)
plt.tight_layout()
plt.show()



