import numpy as np
from scipy.integrate import solve_ivp
import cv2
import matplotlib.pyplot as plt

# ====== Sistemul Sprott C ======
def sprott_system(t, state):
    x, y, z = state
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x**2
    return [dxdt, dydt, dzdt]

# ====== Parametrii integrare ======
T_final = 200
dt = 0.01
t_eval = np.arange(0, T_final, dt)
cut_index = int(100 / dt)

# ====== Încărcare imagine și criptare  ======
image = cv2.imread("dog.jpg")
if image is None:
    print("Imaginea nu a fost încărcată corect!")
    exit()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = image_gray.shape
num_pixels = rows * cols
image_flat = image_gray.flatten()

# ====== Cheia reală (secretă) ======
state_real = np.array([0.06, 0.03, 0.01])
sol_real = solve_ivp(lambda t, y: sprott_system(t, y), (0, T_final), state_real, t_eval=t_eval)
x_real = np.abs(np.sin(sol_real.y[0][cut_index:]) + np.sin(sol_real.y[1][cut_index:]) + np.sin(sol_real.y[2][cut_index:]))
key_real = (255 * x_real / np.max(x_real)).astype(np.uint8)
key_real = np.resize(key_real, num_pixels)

perm_real = np.argsort(key_real)
image_perm = image_flat[perm_real]
encrypted_flat = np.bitwise_xor(image_perm, key_real)
encrypted_image = encrypted_flat.reshape((rows, cols))

# ====== Atac brute-force ======
best_psnr = 0
best_image = None
best_key = None

print("Pornim atacul brute-force...")

for i in range(1000):
    x0 = np.random.uniform(0, 0.1)
    y0 = np.random.uniform(0, 0.1)
    z0 = np.random.uniform(0, 0.1)
    state_guess = np.array([x0, y0, z0])
    
    try:
        sol = solve_ivp(lambda t, y: sprott_system(t, y), (0, T_final), state_guess, t_eval=t_eval, rtol=1e-6)
        x_chaos = np.abs(np.sin(sol.y[0][cut_index:]) + np.sin(sol.y[1][cut_index:]) + np.sin(sol.y[2][cut_index:]))
        key_guess = (255 * x_chaos / np.max(x_chaos)).astype(np.uint8)
        key_guess = np.resize(key_guess, num_pixels)

        perm = np.argsort(key_guess)
        inv_perm = np.argsort(perm)

        decrypted_perm = np.bitwise_xor(encrypted_flat, key_guess)
        decrypted_flat = decrypted_perm[inv_perm]
        decrypted_image = decrypted_flat.reshape((rows, cols))

        # PSNR folosind OpenCV
        psnr_val = cv2.PSNR(image_gray, decrypted_image)

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_image = decrypted_image.copy()
            best_key = state_guess

        if (i+1) % 100 == 0:
            print(f"[{i+1}/100] PSNR maxim până acum: {best_psnr:.2f} dB")

    except Exception as e:
        continue


# ====== Afișare rezultat final ======
print(f"\nCea mai bună decriptare a avut PSNR: {best_psnr:.4f} dB")
print(f"Cheie aproximativă ghicită: {best_key}")

plt.figure(figsize=(12, 6))

# Imagine originală
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title("Imagine originală")
plt.axis('off')

# Imagine decriptată cea mai bună
plt.subplot(1, 2, 2)
plt.imshow(best_image, cmap='gray')
plt.title(f"Imagine decriptată — PSNR {best_psnr:.2f} dB")
plt.axis('off')

plt.tight_layout()
plt.show()

