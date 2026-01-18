import numpy as np
from scipy.integrate import solve_ivp
import cv2
import matplotlib.pyplot as plt

# ====== 1. Sistemul Thomas ======
def thomas_system(t, state):
   x, y, z = state
   dxdt = y * z
   dydt = x - y
   dzdt = 1 - x**2
   return [dxdt, dydt, dzdt]

# ====== 2. Integrarea sistemului și generarea cheii ======
T_final = 200
dt = 0.01
t_eval = np.arange(0, T_final, dt)
state_initial = np.random.randn(3)
b = 0.18

sol = solve_ivp(lambda t, y: thomas_system(t, y), (0, T_final), state_initial, t_eval=t_eval)
x, y, z = sol.y

# Eliminăm faza tranzitorie (primele 40%)
cut_index = int(0.4 * len(x))
chaotic_seq = np.abs(x[cut_index:] + y[cut_index:] + z[cut_index:])

# ====== 3. Încărcare imagine .jpg și conversie în gri ======
image = cv2.imread("photo.jpg")
if image is None:
    print("Imaginea nu a fost încărcată corect!")
    exit()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = image_gray.shape
num_pixels = rows * cols

# ====== 4. Generare cheie scalată (0–255) ======
key_stream = (255 * chaotic_seq / np.max(chaotic_seq)).astype(np.uint8)
key_stream = np.resize(key_stream, num_pixels)

# ====== 5. Permutare haotică a pixelilor ======
permutare = np.argsort(key_stream)  # permutăm după valorile cheii
inverse_permutare = np.argsort(permutare)

# Transformăm imaginea în vector 1D
image_flat = image_gray.flatten()

# Aplicăm permutarea pozițiilor pixelilor
image_perm = image_flat[permutare]

# ====== 6. Criptare XOR ======
encrypted_flat = np.bitwise_xor(image_perm, key_stream)

# ====== 7. Decriptare ======
# Aplicăm din nou XOR
decrypted_perm = np.bitwise_xor(encrypted_flat, key_stream)
# Inversăm permutarea pozițiilor
decrypted_flat = decrypted_perm[inverse_permutare]
# Refacem imaginea 2D
decrypted_image = decrypted_flat.reshape((rows, cols))
encrypted_image = encrypted_flat.reshape((rows, cols))

# ====== 8. Afișare rezultate ======
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

axs[0].imshow(image_gray, cmap='gray')
axs[0].set_title('Imagine originală')
axs[0].axis('off')

axs[1].imshow(encrypted_image, cmap='gray')
axs[1].set_title('Imagine criptată (SprottC + permutare)')
axs[1].axis('off')

axs[2].imshow(decrypted_image, cmap='gray')
axs[2].set_title('Imagine decriptată')
axs[2].axis('off')

plt.tight_layout()
plt.show()

