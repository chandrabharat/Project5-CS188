import nn
import backend as B

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
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        ran = self.run(x)
        if (nn.as_scalar(ran) >= 0):
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        stop = 0
        while stop == 0:
            stop = 1
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                scalar = nn.as_scalar(y)
                if scalar != prediction:
                    stop = 0
                    self.w.update(x, scalar)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w = nn.Parameter(1, 300)
        self.w2 = nn.Parameter(300,1)
        self.b = nn.Parameter(1,300)
        self.b2 = nn.Parameter(1,1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        lin = nn.Linear(x, self.w)
        biased = nn.AddBias(lin, self.b)
        toplayer = nn.ReLU(biased)
        lin2 = nn.Linear(toplayer, self.w2)
        retval = nn.AddBias(lin2, self.b2)
        return retval

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        yhat = self.run(x)
        return nn.SquareLoss(yhat, y) 

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:

            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if nn.as_scalar(loss) < 0.02:
                break
            else:

                for x, y in dataset.iterate_once(4):
                    new_loss = self.get_loss(x, y)
                    gradient = nn.gradients(new_loss, [self.w, self.w2, self.b, self.b2])
                    self.w.update(gradient[0], -0.0075)
                    self.w2.update(gradient[1], -0.0075)
                    self.b.update(gradient[2], -0.0075)
                    self.b2.update(gradient[3], -0.0075)

        


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
        #x = w, y = b

        x1_val = nn.Parameter(784,100)
        x2_val = nn.Parameter(100,10)
        y1_val = nn.Parameter(1,100)
        y2_val = nn.Parameter(1,10)

        self.lr = -.1

        self.x1 = x1_val
        self.x2 = x2_val
        self.y1 = y1_val
        self.y2 = y2_val
        


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
        "*** YOUR CODE HERE ***"
        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x,self.x1), self.y1)), self.x2), self.y2)

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
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.98:
            num_iter = 1
            for alpha,beta in dataset.iterate_once(num_iter):
                lst = [self.x1, self.x2, self.y1, self.y2]
                curr = self.get_loss(alpha,beta)
                grad = nn.gradients(curr, lst)
                self.x1.update(grad[0], self.lr)
                self.y1.update(grad[2], self.lr)
                self.x2.update(grad[1], self.lr)
                self.y2.update(grad[3], self.lr)

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
        "*** YOUR CODE HERE ***"
        # x = w and y = b

        x0_val = nn.Parameter(47, 110)
        x1_val = nn.Parameter(110, 110)
        x2_val = nn.Parameter(110, 5)
        y0_val = nn.Parameter(1,110)
        y1_val = nn.Parameter(1,5)

        self.x0 = x0_val
        self.x1 = x1_val
        self.x2 = x2_val
        self.y0 = y0_val
        self.y1 = y1_val

        self.lr = -.1

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
        "*** YOUR CODE HERE ***"
        init_node_lst = nn.Linear(xs[0], self.x0)
        nodes = init_node_lst
        for val in xs[1:]:
            to_add = nn.Linear(val, self.x0), nn.Linear(nodes, self.x1)
            nodes = nn.Add(to_add[0], to_add[1])
        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nodes, self.y0)), self.x2), self.y1)

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
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.85:
            iter_val = 30
            for alpha,beta in dataset.iterate_once(iter_val):
                curr = self.get_loss(alpha,beta)
                lst = [self.x0, self.x1, self.x2, self.y0, self.y1]
                grad = nn.gradients(curr, lst)
                self.x0.update(grad[0], self.lr)
                self.y0.update(grad[3], self.lr)
                self.x1.update(grad[1], self.lr)
                self.y1.update(grad[4], self.lr)
                self.x2.update(grad[2], self.lr)
