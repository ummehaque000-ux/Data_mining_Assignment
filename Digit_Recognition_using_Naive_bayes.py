import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
print("Loading dataset...")
mnist = fetch_openml('mnist_784', as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

# Step 2: Normalize data
X = X / 255.0

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Apply PCA (improves Naive Bayes performance)
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Step 5: Train Naive Bayes model
print("Training model...")
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Predict
y_pred_single = model.predict(X_test[index].reshape(1, -1))

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

# Step 8: Show a random test image with prediction
index = random.randint(0, len(X_test) - 1)

# Step 9: Predict single sample
y_pred_single = model.predict(X_test[index].reshape(1, -1))

print("Predicted:", y_pred_single[0])
print("Actual:", y_test[index])

# Reconstruct image from PCA (approximate)
reconstructed_image = pca.inverse_transform(X_test[index])

plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {y_pred[index]}, Actual: {y_test[index]}")
plt.axis('off')
plt.show()