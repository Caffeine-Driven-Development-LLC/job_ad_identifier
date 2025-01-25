import util.file_utils as file_utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_data, train_labels = file_utils.get_data("train")
validation_data, validation_labels = file_utils.get_data("validation")

callback_list = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='out/cp.ckpt',
        monitor='val_accuracy',
        save_best_only=True,
    ),
]

url_bag_size = len(train_data[0]['url'])
inner_text_bag_size = len(train_data[0]['text'])

inner_text = layers.Input(shape=(inner_text_bag_size,), dtype='int32', name='inner_text_input')
url = layers.Input(shape=(url_bag_size,), dtype='int32', name='url_input')
features = layers.concatenate([inner_text, url], name='concatenate')
features = layers.Dense(16, activation='relu', name='Dense_1')(features)
features = layers.Dense(8, activation='relu', name='Dense_2')(features)
outputs = layers.Dense(1, activation='sigmoid', name='Output')(features)

model = keras.Model(inputs=[inner_text, url], outputs=outputs)

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

model.summary()

inner_text_train = tf.constant([row['text'] for row in train_data])
url_train = tf.constant([row['url'] for row in train_data])
y_train = tf.constant(train_labels)

inner_text_validation = tf.constant([row['text'] for row in validation_data])
url_validation = tf.constant([row['url'] for row in validation_data])
y_validation = tf.constant(validation_labels)

history = model.fit(
    [inner_text_train, url_train],
    y_train,
    epochs=50,
    validation_data=([inner_text_validation, url_validation], y_validation),
    callbacks=callback_list,
)

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, 'out/')

import numpy as np
np.object = np.object_

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.legend()
plt.show()

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.legend()
plt.show()

# print the val_accuracy and val_loss of the best model
print('Best model val_accuracy:', max(history.history['val_accuracy']))
print('Best model val_loss:', min(history.history['val_loss']))

