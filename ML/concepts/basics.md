# framing

## Supervised machine learning

ML systems learn how to combine input to produce useful predictions on never-before-seen data.



## labels

A label is the thing we're predicting—the `y` variable in simple linear regression.



## features

A feature is the thing we're offering—the `x` variable in simple linear regression. A single machine learning project might use a single feature, while a more sophisticated machine learning project could use millions of features, specified as:
$$
x_1,x_2,\cdots,x_n
$$


## examples

An example is a particular instance of data, $$\boldsymbol x$$. Examples are divided into labeled and unlabeled examples.

A **labeled example** includes both feature(s) and the label: 

```
labeled examples: {features, label}: (x, y)
```

Labeled examples are used to **train** the model.

An unlabeled example contains features but not the label:

```
unlabeled examples: {features, ?}: (x, ?)
```

Once we've trained our model with labeled examples, we use that model to predict the label on unlabeled examples.



## models

A model defines the relationship between features and label. There are two phases of a model's life: 

+ **Training** means creating or learning the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.
+ **Inference** means applying the trained model to unlabeled examples. That is, you use the trained model to make useful predictions.

There are two types of model: 

+ A **regression** model predicts continuous value.
+ A **classification** model predicts discrete values.





# training

Training a model simply means determining good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

Loss is the penalty for a bad prediction. That is, **loss** is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. The goal of training a model is to find a set of weights and biases that have *low* loss, on average, across all examples.

The linear regression models use a loss function called **squared loss**. The squared loss for a single example is $$(y-y')^2$$. **Mean square error** (**MSE**) is the average squared loss per example over the whole dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples:
$$
MSE=\frac{1}{N}\sum_{(x,y)\in D}(y-{\rm prediction}(x))^2
$$

+ $$(x,y)$$ is an example, $$x$$ is the set of features, $$y$$ is the example's label.
+ $${\rm prediction}(x)$$ is a function that predicts label with the set of features.
+ $D$ is a data set containing many labeled examples
+ $$N$$ is the number of examples in D





# reducing loss

## iterative approach

The iterative approach is used throughout machine learning, as following figure suggests. Iterative strategies are prevalent in machine learning, primarily because they scale so well to large data sets.

![The cycle of moving from features and labels to models and predictions.](https://developers.google.com/machine-learning/crash-course/images/GradientDescentDiagram.svg)

The model takes one or more features as input and returns one prediction as output.  The process is as followed: 

1. pick starting values for parameters
2. compute loss
3. generate new values for parameters
4. compute loss
5. 3 - 4 loop until overall loss stops changing or at least changes extremely slowly
6. converged, return



## gradient descent

The brute force way of finding the convergence point is calculating the loss for all possible values of the parameters. Another popular mechanism called **gradient descent** is better.

1. pick starting values for parameters

2. calculate the gradient of the loss curve at current point

3. move in the direction of the negative gradient to next point, the increment is gradient times a scalar named **learning rate**

   > note that a gradient is a vector, the direction of gradient always points in the direction of steepest increase

4. 2 - 3 loop



If you specify a learning rate that is too small, learning will take too long; if too large, learning will perpetually bounce haphazardly across the bottom of the well:

![Same U-shaped curve. Lots of points are very close to each other and their trail is making extremely slow progress towards the bottom of the U.](https://developers.google.com/machine-learning/crash-course/images/LearningRateTooSmall.svg)![Same U-shaped curve. This one contains very few points. The trail of points jumps clean across the bottom of the U and then jumps back over again.](https://developers.google.com/machine-learning/crash-course/images/LearningRateTooLarge.svg)

There's a **Goldilocks learning rate** for every regression problem. The Goldilocks value is related to how flat the loss function is:

![Same U-shaped curve. The trail of points gets to the minimum point in about eight steps.](https://developers.google.com/machine-learning/crash-course/images/LearningRateJustRight.svg)



**batch**

A **batch** is the total number of examples you use to calculate the gradient in a single iteration. So far, we've assumed that the batch has been the entire data set. However, a large data set with randomly sampled examples probably contains redundant data. In fact, redundancy becomes more likely as the batch size grows. Some redundancy can be useful to smooth out noisy gradients, but enormous batches tend not to carry much more predictive value than large batches.

We could get the right gradient *on average* for much less computation by choosing examples at random from our data set. **Stochastic gradient descent** (**SGD**) takes this idea to the extreme--it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy.

**Mini-batch stochastic gradient descent** (**mini-batch SGD**) is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.