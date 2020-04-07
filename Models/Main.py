
# import tensorflow
# import keras
# from keras.datasets import fashion_mnist
#
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#

i)mport mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k'