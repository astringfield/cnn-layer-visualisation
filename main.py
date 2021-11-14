import datetime
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from utils import plot_to_image, image_grid


def main():
    # Label names
    label_names = ["top", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    batch_size = 32
    epochs = 20
    learning_rate = 1e-3

    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = \
        keras.datasets.fashion_mnist.load_data()

    # Normalise input images (0, 1)
    max_val = train_images.max()
    train_images = train_images.astype("float32") / max_val
    test_images = test_images.astype("float32") / max_val

    # Ensure images have only one colour channel (i.e. height, width, 1)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    val_dataset.batch(batch_size=batch_size)

    input_shape = train_images[0].shape
    classifier = keras.Sequential()
    optimiser = keras.optimizers.SGD(learning_rate=learning_rate)
    loss_function = keras.losses.CategoricalCrossentropy()

    # Step 1 - Convolution
    classifier.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    classifier.add(layers.Conv2D(32, (3, 3), activation='relu'))
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
    classifier.add(layers.Dropout(0.5))  # antes era 0.25

    # Adding a second convolutional layer
    classifier.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
    classifier.add(layers.Dropout(0.5))  # antes era 0.25

    # Adding a third convolutional layer
    classifier.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
    classifier.add(layers.Dropout(0.5))  # antes era 0.25

    # Step 3 - Flattening
    classifier.add(layers.Flatten())

    # Step 4 - Full connection
    classifier.add(layers.Dense(units=512, activation='relu'))
    classifier.add(layers.Dropout(0.5))
    classifier.add(layers.Dense(units=10, activation='softmax'))

    classifier.summary()

    for epoch in range(epochs):
        print('\nRunning epoch %d', epoch)

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                logits = classifier(x_batch_train, training=True)
                loss_value = loss_function(y_batch_train, logits)
                a = 2

            gradients = tape.gradient(loss_value, classifier.trainable_weights)
            optimiser.apply_gradients(zip(gradients, classifier.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

    a = 2


if __name__ == '__main__':
    main()
