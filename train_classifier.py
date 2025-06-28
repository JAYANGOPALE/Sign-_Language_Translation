import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Load hand landmark data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = []
labels = []

# Use 2D landmarks: 21 × (x, y) = 42
for d, l in zip(data_dict['data'], data_dict['labels']):
    if len(d) == 42:
        data.append(d)
        labels.append(l)

data = np.array(data)
labels = np.array(labels)

print(f"✅ Samples loaded: {len(data)}")
print(f"Label distribution: {Counter(labels)}")

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc*100:.2f}%")

# Save model
with open('rf_model.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)

print("📦 Model saved as rf_model.pickle")
