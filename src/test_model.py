import tensorflow as tf
from tensorflow import keras
import util.file_utils as file_utils


test_data, test_labels = file_utils.get_data("test")
model = keras.models.load_model('out/model.keras')

inner_text_test = tf.constant([row['text'] for row in test_data])
url_test = tf.constant([row['url'] for row in test_data])
y_test = tf.constant(test_labels)

results = model.evaluate([inner_text_test, url_test], y_test)

# print the model's accuracy
print(f"test loss: {results[0]}, test accuracy: {results[1]}")