import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern):
        output = input_pattern.copy()
        for _ in range(10):  # Iterations
            output = np.sign(np.dot(output, self.weights))
            output[output == 0] = 1  # Handle zeros
        return output

# Crear un vector que represente un aro
def create_ring_vector(size=40, inner_radius=10, outer_radius=15):
    ring_vector = np.full(size * size, -1)  # Initialize with -1
    center = size // 2  # Calculate the center

    for i in range(size):
        for j in range(size):
            distance_squared = (i - center) ** 2 + (j - center) ** 2
            if inner_radius**2 < distance_squared <= outer_radius**2:
                ring_vector[i * size + j] = 1  # Fill with 1 where the ring is

    
    ring_vector[center * size + center] = 2

    return ring_vector

# Create the vector for the ring
x = create_ring_vector(size=40, inner_radius=10, outer_radius=15)

# Inicializar la red
hopfield_net = HopfieldNetwork(size=len(x))
hopfield_net.train([x])

# Imagen con ruido
noisy_image = np.full(len(x), -1)  # Initialize noisy image with -1


offset_x = 5  # Shift in x
offset_y = 3  # Shift in y

for i in range(40):
    for j in range(40):
        distance_squared = (i - (20 + offset_y)) ** 2 + (j - (20 + offset_x)) ** 2
        if 10**2 < distance_squared <= 15**2:  # Use the same inner and outer radii as above
            noisy_image[i * 40 + j] = 1

# Marcar el centro del aro modificado
noisy_image[(20 + offset_y) * 40 + (20 + offset_x)] = 2

# Predicción
predicted_image = hopfield_net.predict(noisy_image)

# Mostrar resultados
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.title("Imagen con Ruido")
plt.imshow(noisy_image.reshape(40,-40), cmap='gray')

plt.subplot(132)
plt.title("Patrón Original")
plt.imshow(x.reshape(40,-40), cmap='gray')

plt.subplot(133)
plt.title("Predicción")
plt.imshow(predicted_image.reshape(40,-40), cmap='gray')

plt.tight_layout()
plt.show()

