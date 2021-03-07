import numpy as np
import numpy.matlib 

class NeuralNetwork:
    """Implementation of a Neural Network

    Attributes
    -----------
        sizes (list of int): Size of each layer currently only supports 2 or 3 layers
        eta (double): Learning rate eta
        epochs (int): number of epochs
    
    Raises
    ------
        ValueError: if sizes has less than 2 layers
    """

    def __init__(self, sizes, eta, epochs, batch_size=1, batches=1):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.batches = batches

        if len(sizes) < 2:
            raise ValueError("Network must have at least 2 layers")
        
        self.sizes = [(a,b) for a,b in zip(sizes[1:],sizes[:-1])]
        
        self.initialise_weights()
        self.initialise_bias()
        

    def initialise_weights(self, method="none"):
        """Initialise weights

        Parameters
        ----------
        method (string): Initialisation method to use must be either 'none' or 'Xavier'
        """
        if method == "none":
            self.weights = [np.random.uniform(0,1,s) for s in self.sizes]
        elif method.lower() == "xavier":
            self.weights = [np.random.uniform(0,1,s)/np.sqrt(1/s[1]) for s in self.sizes]
        
        # Normalise weights
        for i in range(len(self.sizes)):
            self.weights[i] = \
            np.divide(self.weights[i],np.matlib.repmat(np.sum(self.weights[i],1) \
            [:,None],1,self.sizes[i][1]))


    def initialise_bias(self):
        """Initialise bias
        """
        self.bias = [np.zeros(s[0],) for s in self.sizes]

    @staticmethod
    def activation(x, act_type="sigmoid", derivative=False):
        """Define Activation Function Used

        Parameters
        ----------
            x (list of double): The data point
            act_type (string): The activation function
            derivative (bool): Whether or not to use the derivative

        Raises
        -------
        ValueError: If activation type is not sigmoid or relu
        """
        if act_type == "sigmoid":
            def sigmoid(x):
                    return 1.0/(1.0+np.exp(-x))
            if derivative:
                return x*(1.0-x) #sigmoid(x)*(1.0-sigmoid(x)) #
            return sigmoid(x)
        elif act_type == "relu":
            if derivative:
                return np.ones_like(x)
            else:
               return np.maximum(x, 0)
        elif act_type == "tanh":
            if derivative:
                return 1-np.tanh(x)**2
            return np.tanh(x)
        raise ValueError("Act_type must be either 'sigmoid', 'relu' or 'tanh'")


    def forward(self, x):
        """Predict/Calculate the forward pass

        Parameters
        ----------
        x (list of double): The data point
        """
        outputs = []
        outputs.append(x)
        for w,b in zip(self.weights,self.bias):
            x = self.activation(np.matmul(w,x) + b, act_type="sigmoid")
            outputs.append(x)
        return outputs
    

    def backward(self, y_train, output, changes):
        """Back-propogation step

        Parameters
        ----------
        y_train (list of double): The training datapoint
        output (list of double): The NN output
        changes (dict of list of double): The current weights and biases to update
        """
        error = y_train - output[-1]
        # Initialise gradients
        if changes:
            dW = changes['dW']
            dB = changes['dB']
        else:
            dW = [np.zeros(w.shape) for w in self.weights]
            dB = [np.zeros(b.shape) for b in self.bias]

        #print(output)
        for i in range(len(self.sizes),0,-1):
            delta = error*self.activation(output[i],act_type="sigmoid", derivative=True) # error * h'
            #print(delta,output[i])
            #print(output[0])
            dW[i-1] += np.outer(delta,output[i-1])
            dB[i-1] += delta

            error = np.dot(self.weights[i-1].T, delta)
        
        return {'dW':dW, 'dB':dB}


    def update(self, changes):
        """Update weights and bias

        Parameters
        ----------
        changes (list of dict of double): The weights and bias to update
        """
        for i in range(len(changes['dW'])):
            self.weights[i] += self.eta * changes['dW'][i]
            self.bias[i] += self.eta * changes['dB'][i]

    @staticmethod
    def update_changes(old,up):
        """Update dW and dB

        Parameters
        ----------
        old (dict of list of double): the old parameters
        up (dict of list of double): the new parameters
        """
        if not old:
            return up
        for i in range(len(old['dW'])):
            old['dW'][i] += up['dW'][i]
            old['dB'][i] += up['dB'][i] 
        return old

    def train(self, x_train, y_train):
        """Train the model (fit)

        Parameters
        ----------
        x_train (list of double): The training input data
        y_train (list of double): The training output data
        """
        n_samples = x_train.shape[0]
        errors = np.zeros((self.epochs,))
        for i in range(0,self.epochs):
            
            # We will shuffle the order of the samples each epoch
            shuffled_idxs = np.random.permutation(n_samples)

            for batch in range(self.batches):
                changes = {}
                # Loop over all the samples in the batch
                for j in range(0,self.batch_size):
                    
                    # # Input (random element from the dataset)
                    idx = shuffled_idxs[batch*self.batch_size + j]
                    x = x_train[idx]
                    
                    outputs = self.forward(x)

                    
                    # Form the desired output, the correct neuron should have 1 the rest 0
                    desired_output = y_train[idx]
                    
                    # Compute the error signal
                    changes = self.update_changes(changes,self.backward(desired_output, outputs, changes))
                    
                    errors[i] = errors[i] + 0.5*np.sum(np.square(desired_output-outputs[-1]))/n_samples

                self.update(changes)
                    # Store the error per epoch
                    #errors[i] = errors[i] + 0.5*np.sum(np.square(e_n))/n_samples
            print( "Epoch ", i+1, ": error = ", errors[i])
                
        
            

            #print( "Epoch ", i+1, ": error = ", errors[i])


