import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w,x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        num_incorrect = 1
        while num_incorrect != 0:
            num_incorrect = 0
            for x, y in dataset.iterate_once(1):
                # print(self.get_prediction(x), int(nn.as_scalar(y)))
                # print("multiplier: ", nn.as_scalar(self.get_prediction(x))*nn.as_scalar(y))
                if self.get_prediction(x) != nn.as_scalar(y):
                    num_incorrect += 1
                    self.w.update(x,nn.as_scalar(y))








class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.layers = 200
        self.m1 = nn.Parameter(1, self.layers)
        self.m2 = nn.Parameter(self.layers, 1)

        self.b1 = nn.Parameter(1, self.layers)
        self.b2 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        lay1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.m1), self.b1))
        f_x = nn.AddBias(nn.Linear(lay1, self.m2), self.b2)
        return f_x


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """

        for x, y in dataset.iterate_forever(self.layers):
            if nn.as_scalar(self.get_loss(x,y)) < .02:
                break

            loss = self.get_loss(x, y)
            grad_wrt_m1, grad_wrt_b1, grad_wrt_m2, grad_wrt_b2 = nn.gradients(loss, [self.m1, self.b1, self.m2, self.b2])
            mult = -.06
            self.m1.update(grad_wrt_m1, mult)
            self.m2.update(grad_wrt_m2, mult)
            self.b1.update(grad_wrt_b1, mult)
            self.b2.update(grad_wrt_b2, mult)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.layers = 784
        self.m1 = nn.Parameter(self.layers, 200)
        self.b1 = nn.Parameter(1,200)

        self.m2 = nn.Parameter(200, 150)
        self.b2 = nn.Parameter(1, 150)

        self.m3 = nn.Parameter(150, 10)
        self.b3 = nn.Parameter(1, 10)



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """

        lin1 = nn.Linear(x, self.m1)
        bias1 = nn.AddBias(lin1, self.b1)
        relu1 = nn.ReLU(bias1)

        lin2 = nn.Linear(relu1, self.m2)
        bias2 = nn.AddBias(lin2, self.b2)
        relu2 = nn.ReLU(bias2)

        lin3 = nn.Linear(relu2, self.m3)
        bias3 = nn.AddBias(lin3, self.b3)

        return bias3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = -.05
        while dataset.get_validation_accuracy() < .974:
            print(dataset.get_validation_accuracy())
            if (dataset.get_validation_accuracy() >= .89):
                learning_rate = -.005
            for x,y in dataset.iterate_once(10):
                loss = self.get_loss(x, y)
                grad_wrt_m1, grad_wrt_b1, grad_wrt_m2, grad_wrt_b2, grad_wrt_m3, grad_wrt_b3 = nn.gradients(loss,
                                                                                  [self.m1, self.b1, self.m2, self.b2, self.m3, self.b3])
                self.m1.update(grad_wrt_m1, learning_rate)
                self.m2.update(grad_wrt_m2, learning_rate)
                self.m3.update(grad_wrt_m3, learning_rate)

                self.b1.update(grad_wrt_b1, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)
                self.b3.update(grad_wrt_b3, learning_rate)




class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.initialW = nn.Parameter(self.num_chars, 50)

        self.m1 = nn.Parameter(50, 50)
        self.m2 = nn.Parameter(50, 50)

        self.b1 = nn.Parameter(1, 50)
        self.b2 = nn.Parameter(1, 50)

        self.final = nn.Parameter(50,5)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """

        for i in range(len(xs)):
            if i == 0:
                lin1 = nn.Linear(xs[i], self.initialW)
                relu1 = nn.ReLU(nn.AddBias(lin1, self.b1))
            else:
                z = nn.Add(nn.Linear(xs[i], self.initialW), nn.Linear(relu1, self.m1))
                relu1 = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(z, self.b1)), self.m2), self.b2)

        return nn.Linear(relu1, self.final)


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = -.05
        while dataset.get_validation_accuracy() < .84:
            if (dataset.get_validation_accuracy() >= .78):
                learning_rate = -.005
            for x, y in dataset.iterate_once(10):
                loss = self.get_loss(x, y)
                grad_wrt_m1, grad_wrt_b1, grad_wrt_m2, grad_wrt_b2, initialW, finalW = nn.gradients(loss,
                                                                                                            [self.m1,
                                                                                                             self.b1,
                                                                                                             self.m2,
                                                                                                             self.b2,
                                                                                                             self.initialW, self.final])
                self.m1.update(grad_wrt_m1, learning_rate)
                self.m2.update(grad_wrt_m2, learning_rate)

                self.b1.update(grad_wrt_b1, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)

                self.initialW.update(initialW, learning_rate)
                self.final.update(finalW, learning_rate)
