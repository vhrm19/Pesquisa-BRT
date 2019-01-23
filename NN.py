import numpy as np
import tensorflow.keras as kr
import csv

class NeuralNetwork():
    def __init__(self, Input, Output, activation = 'tanh', optimizer = 'adam', epochs = 10000):

        """ 
        Activations: softmax, elu, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, exponential, linear.

        Optimizers: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam.

        Loss: MSE, MAE, MSPE, MSLE, squared_hinge, hinge, categorical_hinge, logcosh, categorical_crossentropy,
              parse_categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity.
        """
        self.Model = kr.Sequential([kr.layers.Dense(3, input_shape = (2,), activation = activation),
                                    kr.layers.Dense(2, activation = activation),
                                    kr.layers.Dense(1, activation = 'linear')])
        self.Model.compile(loss = 'MSE', optimizer= optimizer, metrics = ['acc', 'mae'])
        self.Model.fit(Input, Output, epochs = epochs, verbose = 0)
        evaluate =  self.Model.evaluate(Input, Output, verbose = 0)
        print('Loss:', evaluate[0], 'Accuracy:', evaluate[1], 'MAE:', evaluate[2])

        h = np.dot(np.dot(Input, np.linalg.inv(np.dot(Input.T, Input))), Input.T)
        cost = sum(sum(((self.Model.predict(np.array(Input)) - Output) / (1 - h))**2) / len(Input)) / len(Input)
        print("LOOCV:", cost)
   
    def Predict(self, X, Y):
        print("Result:", float(self.Model.predict(np.array([[X,Y]]))))

if __name__ == "__main__":

    np.random.seed(0)
    dataset = np.array(list(csv.reader(open("csv embarque.csv","r"), delimiter=';'))).astype("float")
    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    NN = NeuralNetwork(Entrada, Tempo_por_Passageiro)
    NN.Predict(1,2)
