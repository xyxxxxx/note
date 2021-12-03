# Structure

A linear model is represented as a graph:

![Three blue circles in a row connected by arrows to a green circle above them](https://developers.google.com/machine-learning/crash-course/images/linear_net.svg)

Each blue circle represents an input feature, and the green circle represents the weighted sum of the inputs.



In the model represented by the following graph, we've added a "hidden layer" of intermediary values. Each yellow node in the hidden layer is a weighted sum of the blue input node values. The output is a weighted sum of the yellow nodes.

![Three blue circles in a row labeled "Input" connected by arrows to a row of yellow circles labeled "Hidden Layer" above them, which are in turn connected to a green circle labeled "Output" at the top.](https://developers.google.com/machine-learning/crash-course/images/1hidden.svg)

However, this model is still linear—its output is still a linear combination of its inputs.



In the model represented by the following graph, the value of each node in Hidden Layer 1 is transformed by a nonlinear function before being passed on to the weighted sums of the next layer. This nonlinear function is called the **activation function**.

![The same as the previous figure, except that a row of pink circles labeled 'Non-Linear Transformation Layer' has been added in between the two hidden layers.](https://developers.google.com/machine-learning/crash-course/images/activation.svg)

Now that we've added an activation function, adding layers has more impact. Stacking nonlinearities on nonlinearities lets us model very complicated relationships between the inputs and the predicted outputs. In brief, each layer is effectively learning a more complex, higher-level function over the raw inputs.



## Common Activation Functions

The following **sigmoid** activation function converts the weighted sum to a value between 0 and 1.
$$
F(x)=\frac{1}{1+e^{−x}}
$$
Here's a plot:

![Sigmoid function](https://developers.google.com/machine-learning/crash-course/images/sigmoid.svg)



The following **rectified linear unit** activation function (or **ReLU**, for short) often works a little better than a smooth function like the sigmoid, while also being significantly easier to compute.
$$
F(x)=\max\{0,x\}
$$
The superiority of ReLU is based on empirical findings, probably driven by ReLU having a more useful range of responsiveness. A sigmoid's responsiveness falls off relatively quickly on both sides.

![ReLU activation function](https://developers.google.com/machine-learning/crash-course/images/relu.svg)

In fact, any mathematical function can serve as an activation function. TensorFlow provides out-of-the-box support for many activation functions. You can find these activation functions within TensorFlow's [list of wrappers for primitive neural network operations](https://www.tensorflow.org/api_docs/python/tf/nn).

> neural networks aren't necessarily always better than feature crosses, but neural networks do offer a flexible alternative that works well in many cases.





# Training

## Failure Cases

There are a number of common ways for backpropagation to go wrong.

### Vanishing Gradients

The gradients for the lower layers (closer to the input) can become very small. In deep networks, computing these gradients can involve taking the product of many small terms.

When the gradients vanish toward 0 for the lower layers, these layers train very slowly, or not at all.

The ReLU activation function can help prevent vanishing gradients.

### Exploding Gradients

If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms. In this case you can have exploding gradients: gradients that get too large to converge.

Batch normalization can help prevent exploding gradients, as can lowering the learning rate.

### Dead ReLU Units

Once the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck. It outputs 0 activation, contributing nothing to the network's output, and gradients can no longer flow through it during backpropagation. With a source of gradients cut off, the input to the ReLU may not ever change enough to bring the weighted sum back above 0.

Lowering the learning rate can help keep ReLU units from dying.



## Dropout Regularization

Another form of regularization, called **Dropout**, is useful for neural networks. It works by randomly "dropping out" unit activations in a network for a single gradient step. The more you drop out, the stronger the regularization:

+ 0.0 = No dropout regularization.
+ 1.0 = Drop out everything. The model learns nothing.
+ Values between 0.0 and 1.0 = More useful.





# Multi-Class Neural Networks

**One vs. all** provides a way to leverage binary classification. Given a classification problem with N possible solutions, a one-vs.-all solution consists of N separate binary classifiers—one binary classifier for each possible outcome. This approach is fairly reasonable when the total number of classes is small, but becomes increasingly inefficient as the number of classes rises.

We can create a significantly more efficient one-vs.-all model with a deep neural network in which each output node represents a different class. The following figure suggests this approach:

![A neural network with five hidden layers and five output layers.](https://developers.google.com/machine-learning/crash-course/images/OneVsAll.svg)



## Softmax

**Softmax** assigns decimal probabilities to each class in a multi-class problem. Those decimal probabilities must add up to 1.0. This additional constraint helps training converge more quickly than it otherwise would.

For example, returning to the image analysis, Softmax might produce the following likelihoods of an image belonging to a particular class:

| Class | Probability |
| :---- | :---------- |
| apple | 0.001       |
| bear  | 0.04        |
| candy | 0.008       |
| dog   | 0.95        |
| egg   | 0.001       |

Softmax is implemented through a neural network layer just before the output layer. The Softmax layer must have the same number of nodes as the output layer.

![A deep neural net with an input layer, two nondescript hidden layers, then a Softmax layer, and finally an output layer with the same number of nodes as the Softmax layer.](https://developers.google.com/machine-learning/crash-course/images/SoftmaxLayer.svg)



 Click the plus icon to see the Softmax equation.



## Softmax Options

Consider the following variants of Softmax:

+ **Full Softmax** is the Softmax we've been discussing; that is, Softmax calculates a probability for every possible class.
+ **Candidate sampling** means that Softmax calculates a probability for all the positive labels but only for a random sample of negative labels. For example, if we are interested in determining whether an input image is a beagle or a bloodhound, we don't have to provide probabilities for every non-doggy example.

Full Softmax is fairly cheap when the number of classes is small but becomes prohibitively expensive when the number of classes climbs. Candidate sampling can improve efficiency in problems having a large number of classes.



## One Label vs. Many Labels

Softmax assumes that each example is a member of exactly one class. Some examples, however, can simultaneously be a member of multiple classes. For such examples:

+ You may not use Softmax.
+ You must rely on multiple logistic regressions.

For example, suppose your examples are images containing exactly one item—a piece of fruit. Softmax can determine the likelihood of that one item being a pear, an orange, an apple, and so on. If your examples are images containing all sorts of things—bowls of different kinds of fruit—then you'll have to use multiple logistic regressions instead.