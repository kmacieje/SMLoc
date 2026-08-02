import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

df = pd.read_csv('office_dataset_mean_std.csv', sep=';')

X = df.drop(columns=['x', 'y']).values
y = df[['x', 'y']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(600, activation='relu'),
    Dense(600, activation='relu'),
    Dense(2, activation='relu')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train_scaled, y_train,
    epochs=500,
    batch_size=100,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)

predictions = model.predict(X_test_scaled)