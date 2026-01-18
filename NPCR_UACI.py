#NPCR SI UACI 
import numpy as np
from scipy.integrate import solve_ivp
import cv2
import matplotlib.pyplot as plt

# ====== 1. Sistemul Sprott C ======
def sprott_system(t, state):
    x, y, z = state
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x**2
    return [dxdt, dydt, dzdt]

# ====== 2. Parametri integrare ======
T_final = 2800
dt = 0.01
t_eval = np.arange(0, T_final, dt)
cut_index = int(100 / dt)

# ====== 3. Imagine și preprocesare ======
image = cv2.imread("fish.jpg")
if image is None:
    print("Imaginea nu a fost încărcată corect!")
    exit()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = image_gray.shape
num_pixels = rows * cols
image_flat = image_gray.flatten()

# ====== 4. Imagine modificată într-un singur pixel ======
image_gray_mod = image_gray.copy()
image_gray_mod[0, 0] = (int(image_gray_mod[0, 0]) + 1) % 256
image_flat_mod = image_gray_mod.flatten()

# ====== 5. Calcul hash_val pentru fiecare imagine ======

def compute_simple_hash(image):
    mean_val = np.mean(image)       # media pixelilor
    std_val  = np.std(image)        # deviația standard a pixelilor
    hash_val = ((mean_val + std_val) * 1e-2) % 1
    return hash_val

hash_val_A  = compute_simple_hash(image_gray)
hash_val_B = compute_simple_hash(image_gray_mod)

# ====== 6. Condiții inițiale dependente de imagine ======
state_base = np.array([1.3, 1.2, 0.01])
state_initial_A  = state_base + np.array([hash_val_A,  hash_val_A / 2,  hash_val_A / 3])
state_initial_B = state_base + np.array([hash_val_B, hash_val_B / 2, hash_val_B / 3])


# ====== 7. Integrare haotică ======
sol_A = solve_ivp(lambda t, y: sprott_system(t, y), (0, T_final), state_initial_A, t_eval=t_eval)
sol_B = solve_ivp(lambda t, y: sprott_system(t, y), (0, T_final), state_initial_B, t_eval=t_eval)

x_A, y_A, z_A = sol_A.y[:, cut_index:]
x_B, y_B, z_B = sol_B.y[:, cut_index:]

# ====== 8. Cheie din sin(x) + sin(y) + sin(z) ======
seq_A  = np.abs(np.sin(x_A) + np.sin(y_A) + np.sin(z_A))
seq_B = np.abs(np.sin(x_B) + np.sin(y_B) + np.sin(z_B))

key_A  = (255 * seq_A / np.max(seq_A)).astype(np.uint8)
key_B = (255 * seq_B / np.max(seq_B)).astype(np.uint8)

key_A  = np.resize(key_A, num_pixels)
key_B = np.resize(key_B, num_pixels)


# ====== 8. Permutări haotice ======
perm_A  = np.argsort(key_A)
perm_B = np.argsort(key_B)

# ====== 9. Criptare ambele imagini ======
image_perm_A  = image_flat[perm_A]
image_perm_B = image_flat_mod[perm_B]

encrypted_A  = np.bitwise_xor(image_perm_A, key_A)
encrypted_B = np.bitwise_xor(image_perm_B, key_B)

encrypted_img   = encrypted_A.reshape((rows, cols))
encrypted_img_B = encrypted_B.reshape((rows, cols))

# ====== 10. Afișare imagini criptate ======
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

axs[0].imshow(image_gray, cmap='gray')
axs[0].set_title('Imagine originală')
axs[0].axis('off')

axs[1].imshow(encrypted_img, cmap='gray')
axs[1].set_title('Imagine criptată originală')
axs[1].axis('off')

axs[2].imshow(encrypted_img_B, cmap='gray')
axs[2].set_title('Imagine criptată (1 pixel modificat)')
axs[2].axis('off')

plt.tight_layout()
plt.show()


# ====== 9. Calcul NPCR și UACI ======
enc = encrypted_img.astype(np.int16)
enc_p = encrypted_img_B.astype(np.int16)

diff_pixels = np.sum(enc != enc_p)
total_pixels = enc.size
NPCR = (diff_pixels / total_pixels) * 100

UACI = (np.sum(np.abs(enc - enc_p)) / (total_pixels * 255)) * 100

print(f"NPCR: {NPCR:.4f}%")
print(f"UACI: {UACI:.4f}%")
