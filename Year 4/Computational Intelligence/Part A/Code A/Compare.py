import numpy
import matplotlib.pyplot as plt

train_data = numpy.loadtxt("D:\\University\\Year 4\\EE40098\\MNIST\\fashion_mnist_train.csv", delimiter=",")
test_data = numpy.loadtxt("D:\\University\\Year 4\\EE40098\\MNIST\\fashion_mnist_test.csv", delimiter=",") 

trainn_data = numpy.loadtxt("D:\\University\\Year 4\\EE40098\\MNIST\\mnist_train.csv", delimiter=",")
testn_data = numpy.loadtxt("D:\\University\\Year 4\\EE40098\\MNIST\\mnist_test.csv", delimiter=",") 

fac = 0.99 / 255

train_imgs = numpy.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = numpy.asfarray(test_data[:, 1:]) * fac + 0.01

trainn_imgs = numpy.asfarray(trainn_data[:, 1:]) * fac + 0.01
testn_imgs = numpy.asfarray(testn_data[:, 1:]) * fac + 0.01

for i in range(10):
    img = train_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()

for i in range(10):
    img = trainn_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()