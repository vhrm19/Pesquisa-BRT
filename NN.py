import numpy as np
import csv
import tensorflow.keras as kr

class NeuralNetwork():
    def __init__(self, Input, Output, activation = 'relu', optimizer = 'adam', epochs = 10000):
        self.Input = Input
        self.Output = Output
        self.activation_function = activation
        self.optimizer = optimizer
        self.epochs = epochs

        """ 
        Activations: softmax, elu, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, exponential, linear.

        Optimizers: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam.

        Loss: MSE, MAE, MSPE, MSLE, squared_hinge, hinge, categorical_hinge, logcosh, categorical_crossentropy,
                sparse_categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity.
        """

        self.Model = kr.Sequential([ kr.layers.Dense(3, input_dim = self.Input[0].T.shape[0], activation = self.activation_function),
                                kr.layers.Dense(2, activation = self.activation_function),
                                kr.layers.Dense(1, activation = 'linear')])
        
        self.Model.compile(loss='MSE', optimizer= self.optimizer, metrics=['accuracy'])
        self.Model.fit(Entrada, Tempo_por_Passageiro, epochs = self.epochs, shuffle = False, verbose = 0)
        scores = self.Model.evaluate(Entrada, Tempo_por_Passageiro, verbose = 0)
        print("Loss: %f, Accuracy: %f%%" % (scores[0], scores[1]*100))
    
    def Predict(self, X, Y):
        print("Result:", float(self.Model.predict(np.array([[X,Y]]))))

if __name__ == "__main__":

    dataset = np.array(list(csv.reader(open("csv embarque.csv","r"), delimiter=';'))).astype("float")
    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    NN = NeuralNetwork(Entrada, Tempo_por_Passageiro)
    NN.Predict(1,2)
