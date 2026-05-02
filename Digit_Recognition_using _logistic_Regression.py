import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
print("Loading dataset...")
mnist = fetch_openml('mnist_784', as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

# Step 2: Normalize
X = X / 255.0

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

# Step 7: Show result
index = random.randint(0, len(X_test) - 1)
 
plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {y_pred[index]}, Actual: {y_test[index]}")
plt.show()