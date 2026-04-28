from sklearn.neural_network import MLPClassifier
import numpy as np

# --- DEEP LEARNING: FINDING COMPLEX, HIDDEN PATTERNS ---

# 1. The Complex Travel Data 
# Format: [Latitude, Longitude, Tourist Density Score (0-100)]
X_train_dl = np.array([
    [35.6, 139.6, 95], # Tokyo Center (Tourist Trap)
    [34.9, 135.7, 88], # Kyoto Center (Tourist Trap)
    [36.2, 137.9, 12], # Matsumoto mountains (Hidden Gem)
    [33.8, 135.3, 15], # Kumano Kodo trail (Hidden Gem)
    [35.1, 139.0, 85]  # Hakone main strip (Tourist Trap)
])

# Labels: 0 = Tourist Trap, 1 = Hidden Gem
y_train_dl = np.array([0, 0, 1, 1, 0])

# 2. The Neural Network (Deep Learning)
# We create "hidden layers" (hidden_layer_sizes) to learn the deep patterns
dl_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
dl_model.fit(X_train_dl, y_train_dl)

# --- LIVE TEST: The Unknown Location ---
# A small coastal village in Shikoku with moderate foot traffic
test_lat = 33.5
test_lon = 133.5
test_density = 20

print("=====================================================")
print(f"  DL LIVE TEST: Coordinates {test_lat}, {test_lon} | Density: {test_density}")
print("=====================================================\n")

prediction = dl_model.predict([[test_lat, test_lon, test_density]])
result = "Hidden Gem!" if prediction[0] == 1 else "Tourist Trap"

print(f"Deep Learning Classification: {result}")
print("\nWhy DL is used here: A simple ML model might just look at 'Density < 50'. The Neural Network maps the geometric relationship between the coordinates AND the density across its hidden layers to understand the actual geography.")
print("=====================================================")