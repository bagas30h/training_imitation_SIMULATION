import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Flatten, Dense, BatchNormalization, Activation, InputLayer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import ntpath
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
columns = ['right', 'center', 'left', 'linear_velocity', 'angular_velocity', 'Commands']
data = pd.read_csv(os.path.join('velocity_data.csv'), names=columns)

# Process image paths
data['right'] = data['right'].apply(lambda x: ntpath.basename(x))
data['center'] = data['center'].apply(lambda x: ntpath.basename(x))
data['left'] = data['left'].apply(lambda x: ntpath.basename(x))

# Clean angular velocity
data['angular_velocity'] = pd.to_numeric(data['angular_velocity'], errors='coerce')
data.dropna(subset=['angular_velocity'], inplace=True)

# Plot histogram of angular velocities
num_bins = 25
samples_per_bin = 2800
hist, bins = np.histogram(data['angular_velocity'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5

# Visualize the distribution of angular velocities
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['angular_velocity']), np.max(data['angular_velocity'])), 
         (samples_per_bin, samples_per_bin), label="Threshold line")

plt.title('Distribution of Angular Velocities')
plt.xlabel('Angular Velocity')
plt.ylabel('Number of Samples')
plt.legend()
plt.show()

# Load images and angular velocity with focus on the center camera or adjusted side cameras
def load_img_angular_velocity(datadir, df):
    image_path = []
    angular_velocity = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        # Use center camera by default, or side cameras for turning adjustments
        if random.random() < 0.8:
            camera_choice = 'center'
            correction = 0.0
        else:
            camera_choice = random.choice(['left', 'right'])
            correction = 0.20 if camera_choice == 'left' else -0.20
        
        # Append image path and adjusted angular velocity
        image_file = indexed_data[camera_choice]
        image_path.append(os.path.join(datadir, image_file.strip()))
        angular_velocity.append(float(indexed_data['angular_velocity']) + correction)
    
    return np.asarray(image_path), np.asarray(angular_velocity)

image_path, angular_velocities = load_img_angular_velocity('D:/Github/dataset_ros2/data', data)

# Split dataset
x_train, x_valid, y_train, y_valid = train_test_split(image_path, angular_velocities, test_size=0.2, random_state=6)

# Preprocess images with augmentation
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (0.5 - np.random.rand()) * 0.5
    hsv[:,:,2] = hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_random_shadow(image):
    top_y = 320 * np.random.rand()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.rand()
    shadow_mask = 0 * image[:,:,1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((Y_m - top_y)*(bot_x - top_x) - (bot_y - top_y)*(X_m - top_x) >= 0)] = 1
    if np.random.rand() > 0.5:
        random_bright = .5
        cond1 = shadow_mask == 1
        image[:,:,0][cond1] = image[:,:,0][cond1] * random_bright
        image[:,:,1][cond1] = image[:,:,1][cond1] * random_bright
        image[:,:,2][cond1] = image[:,:,2][cond1] * random_bright
    return image

def img_preprocess(img):
    img = mpimg.imread(img)
    height = img.shape[0]
    img = img[height//2:, :, :]  # Crop from the bottom to middle part
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = random_brightness(img)
    img = add_random_shadow(img)
    img = cv2.GaussianBlur(img, (1, 1), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# Preprocess and visualize the image processing step
def visualize_img_preprocessing(image_path):
    processed_images = [img_preprocess(img) for img in image_path[:5]]
    
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(processed_images):
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f'Processed Image {i+1}')
        plt.axis('off')
    plt.show()

# Show processed images
visualize_img_preprocessing(x_train)

x_train = np.array([img_preprocess(img) for img in x_train])
x_valid = np.array([img_preprocess(img) for img in x_valid])

# Histogram for training data
hist_train, bins_train = np.histogram(y_train, num_bins)
center_train = (bins_train[:-1] + bins_train[1:]) * 0.5

plt.figure(figsize=(10, 5))
plt.bar(center_train, hist_train, width=0.05, color='blue', alpha=0.6, label='Training Data')
plt.title('Distribution of Angular Velocities (Train Set)')
plt.xlabel('Angular Velocity')
plt.ylabel('Number of Samples')
plt.legend()
plt.show()

# Histogram for validation data
hist_valid, bins_valid = np.histogram(y_valid, num_bins)
center_valid = (bins_valid[:-1] + bins_valid[1:]) * 0.5

plt.figure(figsize=(10, 5))
plt.bar(center_valid, hist_valid, width=0.05, color='orange', alpha=0.6, label='Validation Data')
plt.title('Distribution of Angular Velocities (Validation Set)')
plt.xlabel('Angular Velocity')
plt.ylabel('Number of Samples')
plt.legend()
plt.show()

# Model CIL
def cil_model_improved():
    model = Sequential()

    # Input Layer
    model.add(InputLayer(input_shape=(66, 200, 3)))
    
    # CNN layers
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())

    for _ in range(5):
        kernel_size = (3, 3)
        stride = (2, 2) if (_ % 2 == 0) else (1, 1)
        model.add(Conv2D(64, kernel_size=kernel_size, strides=stride, padding='same', activation='relu'))
        model.add(BatchNormalization())

    model.add(Flatten())
    
    # FCN layers with dropout
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))  # Dropout to prevent overfitting
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Dense(1))  # Output layer for angular velocity
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

model = cil_model_improved()
print(model.summary())

# Train the model
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid), batch_size=2, verbose=1, shuffle=True)

# Save the model
model.save('model_mobil_bes.h5')
print('Model saved')

# Prediction on validation data
y_pred = model.predict(x_valid)

# Evaluation using MSE and MAE
mse = mean_squared_error(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)

print(f"Mean Squared Error (MSE) on validation set: {mse}")
print(f"Mean Absolute Error (MAE) on validation set: {mae}")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Visualize predictions vs actuals
def visualize_predictions(y_true, y_pred):
    plt.figure(figsize=(15, 5))
    plt.plot(y_true, label='True Angular Velocity', color='blue')
    plt.plot(y_pred, label='Predicted Angular Velocity', color='red')
    plt.title('True vs Predicted Angular Velocity')
    plt.xlabel('Sample Index')
    plt.ylabel('Angular Velocity')
    plt.legend()
    plt.show()

visualize_predictions(y_valid, y_pred)
