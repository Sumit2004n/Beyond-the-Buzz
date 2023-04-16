import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("train.csv")

# Split the data into features and labels
X = data.drop("VERDICT", axis=1)
y = data["VERDICT"]

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=X_train.shape[1:]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Make predictions on the test set
test_data = pd.read_csv("test.csv")
X_test = scaler.transform(test_data)
y_pred = model.predict(X_test)

# Save the predictions to a file
np.savetxt("predictions.csv", y_pred, delimiter=",")
