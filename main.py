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
                a=2

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


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

"""
Let's train it using mini-batch gradient with a custom training loop.
First, we're going to need an optimizer, a loss function, and a dataset:
"""

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

"""
Here's our training loop:
- We open a `for` loop that iterates over epochs
- For each epoch, we open a `for` loop that iterates over the dataset, in batches
- For each batch, we open a `GradientTape()` scope
- Inside this scope, we call the model (forward pass) and compute the loss
- Outside the scope, we retrieve the gradients of the weights
of the model with regard to the loss
- Finally, we use the optimizer to update the weights of the model based on the
gradients
"""

epochs = 2
# Delete this line
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))