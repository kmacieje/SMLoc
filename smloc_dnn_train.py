import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

df = pd.read_csv('office_features.csv', sep=';')

X = df.drop(columns=['x', 'y']).values
y = df[['x', 'y']].values
feature_names = df.drop(columns=['x', 'y']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(208, activation='relu'),
    Dropout(0.20057185640801906),
    Dense(176, activation='relu'),
    Dropout(0.20057185640801906),
    Dense(2, activation='linear')
])

optimizer = Adam()
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=600,
    batch_size=8,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping],
    verbose=1
)

stopped_epoch = early_stopping.stopped_epoch
if stopped_epoch > 0:
    print(f"\nStopped early at epoch {stopped_epoch}, restored weights from {stopped_epoch - early_stopping.patience}")
else:
    print("\nFinished all epochs")

model.save("ftm_xy_dnn.keras")
joblib.dump(scaler, "ftm_xy_scaler.pkl")

loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
predictions = model.predict(X_test_scaled)
distance_errors = np.sqrt(np.sum((predictions - y_test)**2, axis=1))
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mean_distance_error = np.mean(distance_errors)
error_68th = np.percentile(distance_errors, 68)
error_95th = np.percentile(distance_errors, 95)
