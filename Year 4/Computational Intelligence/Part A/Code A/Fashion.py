# Import numpy for arrays and matplotlib for drawing the numbers
import numpy
import matplotlib.pyplot as plt

image_size = 28 # width and length
image_pixels = image_size * image_size
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9

train_data = numpy.loadtxt("D:\\University\\Year 4\\EE40098\\MNIST\\fashion_mnist_train.csv", delimiter=",")
test_data = numpy.loadtxt("D:\\University\\Year 4\\EE40098\\MNIST\\fashion_mnist_test.csv", delimiter=",") 
test_data[:10]

test_data[test_data==255]
print(test_data.shape)

fac = 0.99 / 255
train_imgs = numpy.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = numpy.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = numpy.asfarray(train_data[:, :1])
test_labels = numpy.asfarray(test_data[:, :1])

lr = numpy.arange(10)

for label in range(10):
    one_hot = (lr==label).astype(int)
    #print("label: ", label, " in one-hot representation: ", one_hot)

lr = numpy.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(float)
test_labels_one_hot = (lr==test_labels).astype(float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

for i in range(10):
    img = train_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()

@numpy.vectorize
def sigmoid(x):
    return 1 / (1 + numpy.e ** -x)
activation_function = sigmoid

from scipy.stats import truncnorm

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)


class NeuralNetwork:
    
    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        rad = 1 / numpy.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        rad = 1 / numpy.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))
        
    
    def train_single(self, input_vector, target_vector):
        output_vectors = []
        input_vector = numpy.array(input_vector, ndmin=2).T
        target_vector = numpy.array(target_vector, ndmin=2).T

        output_vector1 = numpy.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)
        
        output_vector2 = numpy.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network * \
              (1.0 - output_network)     
        tmp = self.learning_rate  * numpy.dot(tmp, output_hidden.T)
        self.who += tmp

        # calculate hidden errors:
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate * numpy.dot(tmp, input_vector.T)
        
    def train(self, data_array, labels_one_hot_array, epochs=1, intermediate_results=False):
        intermediate_weights = []
        for epoch in range(epochs):  
            print("*", end="")
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append((self.wih.copy(), self.who.copy()))
        return intermediate_weights        
            
    def confusion_matrix(self, data_array, labels):
        cm = {}
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            if (target, res_max) in cm:
                cm[(target, res_max)] += 1
            else:
                cm[(target, res_max)] = 1
        return cm
        
    
    def run(self, input_vector):
        """ input_vector can be tuple, list or ndarray """
        
        input_vector = numpy.array(input_vector, ndmin=2).T

        output_vector = numpy.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = numpy.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

epochs = 50

ANN = NeuralNetwork(no_of_in_nodes = image_pixels, no_of_out_nodes = 10,  no_of_hidden_nodes = 228, learning_rate = 0.01)
    
weights = ANN.train(train_imgs, train_labels_one_hot, epochs=epochs, intermediate_results=True)

cm = ANN.confusion_matrix(train_imgs, train_labels)
#print(ANN.run(train_imgs[i]))
cm = list(cm.items())
#print(sorted(cm))

for i in range(epochs):  
    print("epoch: ", i)
    ANN.wih = weights[i][0]
    ANN.who = weights[i][1]
   
    corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
    print("Accuracy Train: ", corrects / ( corrects + wrongs))
    corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
    print("Accuracy Test: ", corrects / ( corrects + wrongs))
