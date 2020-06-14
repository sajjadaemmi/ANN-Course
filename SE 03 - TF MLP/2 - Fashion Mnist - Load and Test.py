import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = data.load_data()

x_train = x_train / 255.0 
x_test = x_test / 255.0

model = tf.keras.models.load_model('sajjad.h5')

model.evaluate(x_test, y_test)
