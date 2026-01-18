import numpy as np
from scipy.integrate import solve_ivp
import cv2
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ====== 1. Sistemul Thomas ====== pentru Sprott doar se modifica sistemul
b = 0.2

def thomas_system(t, state):
    x, y, z = state
    dxdt = np.sin(y) - b * x
    dydt = np.sin(z) - b * y
    dzdt = np.sin(x) - b * z
    return [dxdt, dydt, dzdt]

# ====== 2. Parametri integrare ======
T_final = 3300
dt = 0.01
t_eval = np.arange(0, T_final, dt)
cut_index = int(500 / dt)  # eliminăm primii 500s
state_initial = np.random.randn(3)


sol = solve_ivp(lambda t, y: thomas_system(t, y), (0, T_final), state_initial, t_eval=t_eval)
x, y, z = sol.y  
x, y, z = x[cut_index:], y[cut_index:], z[cut_index:]

# ====== 3. Încărcare imagine și pregătire ======
image = cv2.imread('photo.jpg')
if image is None:
    print("Imaginea nu a fost încărcată corect!")
    exit()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = image_gray.shape
num_pixels = rows * cols
print(f"Imagine încărcată: {rows}x{cols} ({num_pixels} pixeli)")

# ====== 4. Cheia compusa ======
key_raw = np.abs(np.cos(x) + np.cos(y) + np.sin(z))  # noua expresie
key_scaled = (255 * key_raw / np.max(key_raw)).astype(np.uint8)
key_stream = np.resize(key_scaled, num_pixels).reshape((rows, cols))


# ====== 5. Criptare și decriptare cu XOR ======
encrypted_image = np.bitwise_xor(image_gray, key_stream)
decrypted_image = np.bitwise_xor(encrypted_image, key_stream)

# ====== 6. Afișare rezultate ======
fig, axs = plt.subplots(1, 3, figsize=(24, 6))
axs[0].imshow(image_gray, cmap='gray')
axs[0].set_title('Imagine originală')
axs[0].axis('off')

axs[1].imshow(encrypted_image, cmap='gray')
axs[1].set_title('Imagine criptată fără difuzie (Thomas)')
axs[1].axis('off')

axs[2].imshow(decrypted_image, cmap='gray')
axs[2].set_title('Imagine decriptată')
axs[2].axis('off')


hist, _ = np.histogram(key_stream, bins=256, range=(0, 256), density=True)
entropy_value = entropy(hist, base=2)
print(f"Entropia informațională a cheii compuse: {entropy_value:.4f} biți")


plt.figure(figsize=(10, 5))
plt.hist(encrypted_image.flatten(), bins=256, range=(0, 256), color='gray', edgecolor='black')
plt.title('Histogramă a pixelilor din imaginea criptată')
plt.xlabel('Valori pixel')
plt.ylabel('Frecvență')
plt.grid(True)
plt.tight_layout()
plt.show()
