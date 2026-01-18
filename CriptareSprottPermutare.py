import numpy as np
from scipy.integrate import solve_ivp
import cv2
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import chisquare

# ====== 1. Sistemul Sprott C ======
def sprott_system(t, state):
    x, y, z = state
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x**2
    return [dxdt, dydt, dzdt]

# ====== 2. Integrarea și generarea secvenței haotice ======
T_final = 2800
dt = 0.01
t_eval = np.arange(0, T_final, dt)

state_initial = np.random.randn(3)

sol = solve_ivp(lambda t, y: sprott_system(t, y), (0, T_final), state_initial, t_eval=t_eval)
x, y, z = sol.y

cut_index = int(100 / dt)  # t > 100s

x = x[cut_index:]
y = y[cut_index:]
z = z[cut_index:]

# Formula cheia haotică
chaotic_seq = np.abs(np.sin(x)+np.sin(y)+np.sin(z))

# ====== 3. Încărcare imagine și conversie în gri ======
image = cv2.imread("fish.jpg")
if image is None:
    print("Imaginea nu a fost încărcată corect!")
    exit()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = image_gray.shape
num_pixels = rows * cols
print(f"Imagine încărcată: {rows}x{cols} ({num_pixels} pixeli)")

# ====== 4. Generare cheie scalată (0–255) ======
key_stream = (255 * chaotic_seq / np.max(chaotic_seq)).astype(np.uint8)
key_stream = np.resize(key_stream, num_pixels)

# ====== 5. Permutare haotică a pozițiilor pixelilor ======
permutare = np.argsort(key_stream)
inverse_permutare = np.argsort(permutare)

image_flat = image_gray.flatten()
image_perm = image_flat[permutare]

# ====== 6. Criptare XOR ======
encrypted_flat = np.bitwise_xor(image_perm, key_stream)

# Reconstruiește imaginea criptată 2D
encrypted_image = encrypted_flat.reshape((rows, cols))

# ====== 7. Decriptare ======
decrypted_perm = np.bitwise_xor(encrypted_flat, key_stream)
decrypted_flat = decrypted_perm[inverse_permutare]
decrypted_image = decrypted_flat.reshape((rows, cols))

# ====== 8. Afișare rezultate ======
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

axs[0].imshow(image_gray, cmap='gray')
axs[0].set_title('Imagine originală')
axs[0].axis('off')

axs[1].imshow(encrypted_image, cmap='gray')
axs[1].set_title('Imagine criptată + permutare (Sprott C)')
axs[1].axis('off')

axs[2].imshow(decrypted_image, cmap='gray')
axs[2].set_title('Imagine decriptată')
axs[2].axis('off')

plt.tight_layout()
plt.show()

# ====== 9. Histogramă pentru imaginea criptată ======
plt.figure(figsize=(10, 5))
plt.hist(key_stream, bins=256, range=(0, 256), color='gray', edgecolor='black')
plt.title('Histogramă a cheii compuse')
plt.xlabel('Valori pixel')
plt.ylabel('Frecvență')
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== 10. Entropia cheii ======


# Afișează rezultatul
hist, _ = np.histogram(encrypted_flat, bins=256, range=(0, 256), density=True)
entropy_value = entropy(hist, base=2)
print(f"Entropia informațională a imaginii criptate: {entropy_value:.4f} biți")

# ====== 10.5. Autocorelarea cheii haotice (lag 1–50) ======
def autocorrelation_key(stream, max_lag):
    n = len(stream)
    mean = np.mean(stream)
    var  = np.var(stream)
    acs = []
    for lag in range(1, max_lag+1):
        num = np.sum((stream[:-lag] - mean) * (stream[lag:] - mean))
        acs.append(num / ((n - lag) * var))
    return acs

lags = np.arange(1, 141)
ac_values = autocorrelation_key(key_stream, 140)

plt.figure(figsize=(8, 4))
markerline, stemlines, baseline = plt.stem(lags, ac_values)  # fără use_line_collection
plt.setp(markerline, 'markerfacecolor', 'C0')
plt.setp(stemlines, 'color', 'C0')
plt.title("Autocorelarea cheii haotice (lag 140)")
plt.xlabel("Lag")
plt.ylabel("Autocorelatie")
plt.grid(True)
plt.tight_layout()
plt.show()


def pearson_horizontal(image):
    # Flatten pe linii: X = toţi pixelii din fiecare rând, Y = pixelii din dreapta lor
    image = image.astype(np.float32)
    rows, cols = image.shape
    X = image[:, :-1].flatten()   # toți pixelii cu excepția ultimei coloane
    Y = image[:, 1:].flatten()    # pixelii de sub fiecare, cu excepția primei coloane
    
    # Media și deviația standard
    mu_X = np.mean(X)
    mu_Y = np.mean(Y)
    sigma_X = np.std(X)
    sigma_Y = np.std(Y)
    
    # Covarianța
    cov = np.mean((X - mu_X) * (Y - mu_Y))
    
    # Coeficientul Pearson
    if sigma_X * sigma_Y == 0:
        return 0.0
    return cov / (sigma_X * sigma_Y)

rho_horiz = pearson_horizontal(encrypted_image)
print(f"Coeficientul Pearson orizontal între pixeli adiacenți: {rho_horiz:.4f}")

# ====== 10.1. Autocorelație verticală (coeficient Pearson) ======
def pearson_vertical(image):
    # Flatten pe coloane: X = toți pixelii din fiecare coloană, Y = pixelii de sub ei
    image = image.astype(np.float32)
    rows, cols = image.shape
    X = image[:-1, :].flatten()   # toți pixelii cu excepția ultimei linii
    Y = image[1:, :].flatten()    # pixelii de sub fiecare, cu excepția primei linii

    # Media și deviația standard
    mu_X = np.mean(X)
    mu_Y = np.mean(Y)
    sigma_X = np.std(X)
    sigma_Y = np.std(Y)

    # Covarianța
    cov = np.mean((X - mu_X) * (Y - mu_Y))

    # Coeficientul Pearson
    if sigma_X * sigma_Y == 0:
        return 0.0
    return cov / (sigma_X * sigma_Y)

rho_vert = pearson_vertical(encrypted_image)
print(f"Coeficientul Pearson vertical între pixeli adiacenți: {rho_vert:.4f}")





