"""
Resources
- https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c

"""

import datetime
import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from utils import plot_to_image, image_grid


def tds_model(image_width, image_height, image_depth, classes):
    classifier = keras.Sequential()
    input_shape = (image_width, image_height, image_depth)

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
    return classifier


def main():
    # Label names
    label_names = ["top", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    classes = len(label_names)
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

    train_labels = keras.utils.to_categorical(train_labels, 10)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size)

    test_labels = keras.utils.to_categorical(test_labels, 10)
    val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    val_dataset = val_dataset.batch(batch_size=batch_size)

    input_shape = train_images[0].shape
    optimiser = keras.optimizers.SGD(learning_rate=learning_rate)
    loss_function = keras.losses.CategoricalCrossentropy()

    # Prepare the metrics.
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric = keras.metrics.CategoricalAccuracy()

    classifier = tds_model(input_shape[0], input_shape[1], input_shape[2], classes)

    layer_outputs = [layer.output for layer in classifier.layers[1:]]
    classifier_visualisation = tf.keras.models.Model(inputs=classifier.input, outputs=layer_outputs)

    # # classifier.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"], learning_rate=learning_rate)
    # classifier.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])
    # classifier.fit(train_images, train_labels, batch_size=batch_size,
    #                epochs=epochs, validation_split=0.2)  # ,callbacks=[tensorboard_callback])
    # return

    for epoch in range(epochs):
        print('\nRunning epoch', epoch)
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                logits = classifier(x_batch_train, training=True)
                loss_value = loss_function(y_batch_train, logits)
                a = 2

            gradients = tape.gradient(loss_value, classifier.trainable_weights)
            optimiser.apply_gradients(zip(gradients, classifier.trainable_weights))

            # Update training metric.
            # y_batch_train_decoded = tf.argmax(y_batch_train, axis=1)
            # logits_decoded = tf.argmax(logits, axis=1)
            train_acc_metric.update_state(y_batch_train, logits)

            conv2d_first = classifier.layers[0]
            weights, biases = conv2d_first.get_weights()
            weights_r = weights.reshape(32, 3, 3, 1)
            first = weights[:, :, :, 0]
            first_r = weights_r[0, :, :, :]

            # num_filters = weights.shape[3]
            # for i in range(num_filters):
            #     # Normalise the weights
            #     filter = weights[:, :, :, i]
            #     filter_norm = (filter - filter.min()) / (filter.max() - filter.min())
            #
            #     # Plot each filter
            #     ax = plt.subplot(weights.shape[3], 1, i+1)
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     plt.imshow(filter_norm)
            #     a=2

            visualisation_image = test_images[0:32, :, :, :]
            successive_feature_maps = classifier_visualisation.predict(visualisation_image)

            layer_names = [layer.name for layer in classifier_visualisation.layers]
            for layer_name, feature_map in zip(layer_names, successive_feature_maps):
                print(feature_map.shape)

                if 'conv2d' in layer_name:
                    idx = 0
                    for images in feature_map:
                        # Set of feature maps for the first in the batch
                        # Plot input image
                        plt.figure(f'figure_image_{layer_name}_{idx}')
                        # plt.imshow(visualisation_image[0, :, :, 0])

                        '''Can these images be saved to tensorboard??'''

                        for idx in range(0, 32):
                            # Plot feature map
                            map = images[:, :, idx]
                            ax = plt.subplot(4, 8, idx+1)
                            plt.imshow(map, cmap='gray')
                            # plt.figure('figure_feature' + str(idx))
                            # # plt.imshow(map)
                            a=2
                        plt.show()
                        idx += 1
                        assert True
            assert False

            # Log every 200 batches.
            if step % 200 == 0:
                train_acc = train_acc_metric.result()
                print('training - step: ' + str(step) + ' - accuracy: ' +
                      str(format(float(train_acc), '.5f') + ' - loss: ' + str(format(float(loss_value), '.5f'))))
                # print("Seen so far: %s samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print('training - accuracy: ' + str(format(float(train_acc), '.5f')))
        # print("Training acc over epoch: %.5f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = classifier(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print('validation - accuracy: ' + str(format(float(val_acc), '.5f')))
        # print("Validation acc: %.4f" % (float(val_acc),))
        print("time taken: %.2fs" % (time.time() - start_time))

    a = 2
    print('Done')


if __name__ == '__main__':
    main()
