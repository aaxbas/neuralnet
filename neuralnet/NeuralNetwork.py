import numpy as np
import numpy.matlib 
import time

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
        # ANN Hyperparameters
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.batches = batches
        
        # Regularisation Hyperparameters
        self.tau = 0.01
        self.l1 = 0.001

        if len(sizes) < 2:
            raise ValueError("Network must have at least 2 layers")
        
        self.sizes = [(a,b) for a,b in zip(sizes[1:],sizes[:-1])]
        
        self.initialise_weights()
        self.initialise_bias()
        

    def initialise_weights(self, method="none", w=None):
        """Initialise weights

        Parameters
        ----------
        method (string): Initialisation method to use must be either 'none', 'he' or 'Xavier'
        w (list of :numpy.ndarray:): Initialise using pre-defined weights
        """

        if w is not None:  # Can initialise using pre-define weights
            self.weights = w
        elif method.lower() == "none":         # Initialisation method
            self.weights = [np.random.uniform(0,1,s) for s in self.sizes]
        elif method.lower() == "xavier":
            self.weights = [np.random.randn(*s) * np.sqrt(1 / (s[1])) for s in self.sizes]
            return
        elif method.lower() == "he":
            self.weights = [np.random.randn(*s) * np.sqrt(2 / (s[1])) for s in self.sizes]
            return

        # Normalise weights
        for i in range(len(self.sizes)):
            self.weights[i] = \
            np.divide(self.weights[i],np.matlib.repmat(np.sum(self.weights[i],1) \
            [:,None],1,self.sizes[i][1]))


    def initialise_bias(self, b=None):
        """Initialise bias
        """
        if b is not None:
            self.bias = b
        else:
            self.bias = [np.zeros(s[0],) for s in self.sizes] # initialise to zero

    @staticmethod
    def activation(x, act_type="relu", derivative=False):
        """Define Activation Function Used

        Parameters
        ----------
            x (list of double): The data point
            act_type (string): The activation function
            derivative (bool): Whether or not to use the derivative

        Returns
        --------
        x (:numpy.ndarray:): f(x) - The calculated activation

        Raises
        -------
        ValueError: If activation type is not sigmoid, relu, tanh or abs

        """

        if act_type == "sigmoid":  # Sigmoid Activation
            def sigmoid(x):        # Initialisation method
                return 1.0/(1.0+np.exp(-x))
            if derivative:
                return x*(1.0-x) # Alternate: sigmoid(x)*(1.0-sigmoid(x)) 
            return sigmoid(x)

        elif act_type == "relu": # ReLU Activation
            if derivative:
                new_x = np.copy(x)
                new_x[new_x<=0] = 0
                new_x[new_x>0] = 1
                return new_x
            return np.maximum(x, 0)

        elif act_type == "tanh": # tanh Activation
            if derivative:
                return 1-np.tanh(x)**2
            return np.tanh(x)
        
        elif act_type == "abs": # absolute value and derivative (here for convenience)
            if derivative:
                new_x = np.copy(x)
                new_x[new_x<0] = -1
                new_x[new_x>0] = 1
                return new_x
            return np.abs(x)
        raise ValueError("Act_type must be either 'sigmoid', 'relu', 'abs' or 'tanh'")


    def forward(self, x):
        """Predict/Calculate the forward pass

        Parameters
        ----------
        x (list of double): The data point

        Returns
        --------
        outputs (list of :numpy.ndarray:): The output of each layer
        """
        outputs = []
        outputs.append(x)

        # Loop through each layer and store the outputs
        for w,b in zip(self.weights,self.bias):
            x = self.activation(np.matmul(w,x) + b, act_type="relu")
            outputs.append(x)
        return outputs
    

    def backward(self, y_train, output, changes):
        """Back-propogation step

        Parameters
        ----------
        y_train (list of double): The training datapoint
        output (list of double): The NN output
        changes (dict of list of double): The current weights and biases to update

        Returns
        -------
        changes (dict): Changes to the weights and biases at each layer
        """

        # Calculate error
        error = y_train - output[-1]
        
        # Initialise gradients
        dW = changes['dW']
        dB = changes['dB']

        # Back-propagation
        for i in range(len(self.sizes),0,-1):
            delta = error*self.activation(output[i], act_type="relu", derivative=True)  # error * h'
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
            if self.l1:  # Use L1 Regularisation
                self.weights[i] -= self.eta* self.l1 * self.activation(self.weights[i], act_type="abs", derivative=True)
            
            # Update weights and bias
            self.weights[i] += self.eta * changes['dW'][i]/self.batch_size
            self.bias[i] += self.eta * changes['dB'][i]/self.batch_size
    
    @staticmethod
    def update_changes(old,up):
        """Update dW and dB

        Parameters
        ----------
        old (dict of list of double): the old parameters
        up (dict of list of double): the new parameters

        Returns
        -------
        old (dict): Updated parameters
        """
        for i in range(len(old['dW'])):
            old['dW'][i] += up['dW'][i]
            old['dB'][i] += up['dB'][i] 
        return old

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """Train the model (fit)

        Parameters
        ----------
        x_train (list of double): The training input data
        y_train (list of double): The training output data
        x_val (optional list of double): The validation input data
        y_val (optional list of double): The validation output data

        Returns
        -------
        An (list of :numpy.ndarray:): Average weight per epoch
        """

        # Get number of samples and initialise
        n_samples = x_train.shape[0]
        errors = np.zeros((self.epochs,))
        An = []


        for i in range(0,self.epochs):

            t1 = time.time() # time each epoch
            
            # Hold gradients
            changes = {}
            changes['dW'] = [np.zeros(w.shape) for w in self.weights]
            changes['dB'] = [np.zeros(b.shape) for b in self.bias]
            
            # We will shuffle the order of the samples each epoch
            shuffled_idxs = np.random.permutation(n_samples)
            AdW = np.zeros(self.weights[0].shape)

            for batch in range(self.batches):
                # Initialise gradients
                changes['dW'] = [np.zeros(w.shape) for w in self.weights]
                changes['dB'] = [np.zeros(b.shape) for b in self.bias]
                
                # Loop over all the samples in the batch
                for j in range(0,self.batch_size):
                    
                    # # Input (random element from the dataset)
                    idx = shuffled_idxs[batch*self.batch_size + j]
                    x = x_train[idx]
                    
                    outputs = self.forward(x)

                    # Form the desired output, the correct neuron should have 1 the rest 0
                    desired_output = y_train[idx]
                    
                    # Compute the error signal
                    changes = self.backward(desired_output, outputs, changes)
                    
                    # Calculate Error (MSE)
                    errors[i] = errors[i] + 0.5*np.sum(np.square(desired_output-outputs[-1]))/n_samples
                
                # Get Accumulated gradients
                AdW += changes['dW'][0]
                self.update(changes)
            
            # Calculate EMA of accumulated gradients
            if i == 0:
                An.append(AdW/(self.batch_size*self.batches))
            else:
                An.append(An[i-1]*(1-self.tau) + self.tau*AdW/(self.batch_size*self.batches))

            # Print validation results
            pred_str = ""
            if x_val is not None and y_val is not None:
                pred_val = self.calculate_accuracy(x_val, y_val)
                pred_str = f"Validation Average Accuracy = {pred_val['MeanAccuracy']} "
                pred_str += f"Validation Average Error = {pred_val['MeanError']}"
            
            # Add EL1 if L1 regularisation is used
            el1 = 0
            if self.l1:
                for w in self.weights:
                    el1 += np.sum(np.abs(w)).sum()
                el1 = el1*self.l1
            errors[i] = errors[i] + el1
            
            t2 = time.time()
            print("====================================================================")
            print( "Epoch ", i+1, ": error = ", errors[i],"Time Taken: ", t2-t1)
            print(pred_str)

        return An
                
    def calculate_accuracy(self, x_val,y_val):
        """Calculate Model Accuracy

        Parameters
        ----------
        x_val (list of double): validation input data
        y_val (list of double): validation output data

        Returns
        -------
        predictions (dict): Accuracy and Error per sample and their averages
        """

        # Initialise arrays
        predictions = {"Error":np.zeros((x_val.shape[0], 1)),
        "Accuracy":np.zeros((x_val.shape[0], 1)),
        "MSE":np.zeros((x_val.shape[0], 1)),
        "EL1":np.zeros((x_val.shape[0], 1))}


        for mu, (x, y) in enumerate(zip(x_val, y_val)):
            output = self.forward(x)  # Forward Path
            
            pred = np.argmax(output[-1]) # Get the predicted value

            error = 0.5*np.sum(np.square(output[-1] - y))
            
            # Add EL1 if L1 regularisation is used
            el1 = 0
            if self.l1:
                for w in self.weights:
                    el1 += np.sum(np.abs(w)).sum()
                el1 = el1*self.l1
            
            predictions['MSE'][mu] = error
            predictions['EL1'][mu] = el1
            predictions['Accuracy'][mu] = pred == np.argmax(y)
            
            error += el1
            predictions['Error'][mu] = error
        
        # Calculate Error and Accuracy
        predictions['MeanAccuracy'] = 100*np.sum(predictions['Accuracy'])/x_val.shape[0]
        predictions['MeanError'] = np.sum(predictions['Error'])/x_val.shape[0]
        predictions['MeanMSE'] = np.sum(predictions['Error'])/x_val.shape[0]
        predictions['MeanEL1'] = np.sum(predictions['Error'])/x_val.shape[0]


        return predictions

    def calculate_test_accuracy(self, x_test, y_test):
        """Calculate Model Accuracy

        Parameters
        ----------
        x_test (list of double): test input data
        y_test (list of double): test output data

        Returns
        --------
        results_test (:numpy.ndarray:): Sample Accuracy and Error
        """
        results_test = np.zeros((x_test.shape[0], 2))
        error_test = 0
        
        for mu in range(x_test.shape[0]):
            x3 = self.forward(x_test[mu])[-1]

            # Here calculate the error and accuracy per sample
            error_test = np.sum(x3-y_test[mu])**2
            accuracy_test = int(np.argmax(x3) == np.argmax(y_test[mu]))
            results_test[mu] = [error_test, accuracy_test]
        
        # print("Total accuracy = ", np.sum(results_test[:,1])/x_test.shape[0])
        return results_test