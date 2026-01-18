from scipy.stats import entropy
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import cv2

b=0.15
def thomas_system(t, state):
    x, y, z = state
    dxdt = np.sin(y) - b * x
    dydt = np.sin(z) - b * y
    dzdt = np.sin(x) - b * z
    return [dxdt, dydt, dzdt]

# Integrarea sistemului
T_final = 1000
dt = 0.01
t_span = (0, T_final)
t_eval = np.arange(0, T_final, dt)
state_initial = np.random.randn(3)


sol_thomas = solve_ivp(lambda t, y: thomas_system(t, y), t_span, state_initial, t_eval=t_eval)

x, y, z = sol_thomas.y
cut = int(0.3 * len(x))
x, y, z = x[cut:], y[cut:], z[cut:]

# Normalizare în [0, 1]
x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))

#  Discretizare și calcul entropie (folosim histogramă)
def calculate_entropy_discretized(sequence, bins=256):
    hist, _ = np.histogram(sequence, bins=bins, range=(0, 1))
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]  # evităm log2(0)
    return entropy(probs, base=2)

# Entropii pentru valorile normalizate
entropy_x = calculate_entropy_discretized(x_normalized)
entropy_y = calculate_entropy_discretized(y_normalized)
entropy_z = calculate_entropy_discretized(z_normalized)

print(f"Entropia (discretizată) pentru x: {entropy_x}")
print(f"Entropia (discretizată) pentru y: {entropy_y}")
print(f"Entropia (discretizată) pentru z: {entropy_z}")

# In urma testarii mai multor valori ale lui b cu un set de 5 valori initiale am ales sa iau pe b=0.18.





# # Funcție pentru calcularea autocorelației
def calculate_autocorrelation(sequence, max_lag=250):
    """
    Calculează autocorelația normalizată pentru lag-uri de la 0 la max_lag.
    """
    mean_seq = np.mean(sequence)
    autocorr = np.correlate(sequence - mean_seq, sequence - mean_seq, mode='full')
    autocorr = autocorr[len(sequence)-1:]  # partea pozitivă
    autocorr = autocorr / autocorr[0]  # normalizare
    return autocorr[:max_lag]

# Calcul autocorelații
auto_corr_x = calculate_autocorrelation(x)
auto_corr_y = calculate_autocorrelation(y)
auto_corr_z = calculate_autocorrelation(z)

mean_autocorr_x = np.mean(np.abs(auto_corr_x[1:]))
mean_autocorr_y = np.mean(np.abs(auto_corr_y[1:]))
mean_autocorr_z = np.mean(np.abs(auto_corr_z[1:]))

print(f"Media autocorelației (x): {mean_autocorr_x:.4f}")
print(f"Media autocorelației (y): {mean_autocorr_y:.4f}")
print(f"Media autocorelației (z): {mean_autocorr_z:.4f}")

# ====== Afișare rezultate ======
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(auto_corr_x)
plt.title("Autocorelația normalizată - x")
plt.xlabel("Lag")
plt.ylabel("Autocorelație")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(auto_corr_y)
plt.title("Autocorelația normalizată - y")
plt.xlabel("Lag")
plt.ylabel("Autocorelație")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(auto_corr_z)
plt.title("Autocorelația normalizată - z")
plt.xlabel("Lag")
plt.ylabel("Autocorelație")
plt.grid(True)


plt.tight_layout()
plt.show()

# def normalize_sequence(sequence):
#     # Obținem valorile minime și maxime
#     min_val = np.min(sequence)
#     max_val = np.max(sequence)
    
#     # Normalizăm secvența între 0 și 1
#     normalized_sequence = (sequence - min_val) / (max_val - min_val)
    
#     return normalized_sequence

# normalized_x = normalize_sequence(x)
# normalized_y = normalize_sequence(y)
# normalized_z = normalize_sequence(z)

# # Aplicarea testului Kolmogorov-Smirnov pentru secvențele normalizate
# p_value_x = kstest(normalized_x, 'norm')
# p_value_y = kstest(normalized_y, 'norm')
# p_value_z = kstest(normalized_z, 'norm')

# # Afișarea rezultatelor testului Kolmogorov-Smirnov
# print(f"Testul Kolmogorov-Smirnov pentru x (normalizat): p-value = {p_value_x.pvalue}")
# print(f"Testul Kolmogorov-Smirnov pentru y (normalizat): p-value = {p_value_y.pvalue}")
# print(f"Testul Kolmogorov-Smirnov pentru z (normalizat): p-value = {p_value_z.pvalue}")


# Citirea imaginii și conversia în gri

# image = cv2.imread('lena.png')
# if image is None:
#     print("Imaginea nu a fost încărcată corect!")
#     exit()

# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# rows, cols = image_gray.shape

# # Crearea cheii pentru criptare pe baza sistemului haotic
# key_stream = x+y+z
# key_stream = np.abs(key_stream)  # Ne asigurăm că valorile cheii sunt pozitive

# # Scăderea valorilor pentru a se încadra în intervalul 0-255
# key_stream = (key_stream * 1e5).astype(np.uint8)

# # Numărul total de pixeli din imagine
# num_pixels = rows * cols

# # Extinderea key_stream pentru a se potrivi cu dimensiunea imaginii
# key_stream = np.resize(key_stream, num_pixels)

# # Reshape la dimensiunea imaginii (rows x cols)
# key_stream = key_stream.reshape((rows, cols))

# # Criptarea și decriptarea imaginii
# encrypted_image = np.bitwise_xor(image_gray, key_stream)
# decrypted_image = np.bitwise_xor(encrypted_image, key_stream)


# # Afisarea imaginilor
# fig, axs = plt.subplots(1, 3, figsize=(24, 6))
# axs[0].imshow(image_gray, cmap='gray')
# axs[0].set_title('Imagine originală')
# axs[0].axis('off')

# axs[1].imshow(encrypted_image, cmap='gray')
# axs[1].set_title('Imagine criptată')
# axs[1].axis('off')

# axs[2].imshow(decrypted_image, cmap='gray')
# axs[2].set_title('Imagine decriptată')
# axs[2].axis('off')

# plt.show()

# Crearea secvenței binare pe baza semnalului combinat
# combined_signal = x + y + z
# combined_median = np.median(combined_signal)
# binary_sequence = np.where(combined_signal > combined_median, 1, 0)  # Secvență binară


