# linear regression

## model

$$
y_i=\beta_0+\beta_1x_{i1}+\cdots+\beta_px_{ip}+\varepsilon_i=\boldsymbol x_i^T \boldsymbol \beta +\varepsilon_i,\quad i=1,\cdots,n\\
\boldsymbol y =\boldsymbol X \boldsymbol \beta + \boldsymbol\varepsilon
$$



## ML model

$$
y'=b+w_1x_1+w_2x_2+w_3x_3+\cdots
$$

+ $$y'$$ is the predicted label
+ $$b$$ is the bias, sometimes referred to as $$w_0$$
+ $$w_i$$ is the weight of feature i
+ $$x_i$$ is a feature (a known input).





# Feature Crosses

## Encoding Nonlinearity

In Figures 1 and 2, imagine the following:

+ The blue dots represent sick trees.
+ The orange dots represent healthy trees.

![Blues dots occupy the northeast quadrant; orange dots occupy the southwest quadrant.](https://developers.google.com/machine-learning/crash-course/images/LinearProblem1.png)

**Figure 1. Is this a linear problem?**

Can you draw a line that neatly separates the sick trees from the healthy trees? Sure. This is a linear problem. The line won't be perfect. A sick tree or two might be on the "healthy" side, but your line will be a good predictor.

Now look at the following figure:

![Blues dots occupy the northeast and southwest quadrants; orange dots occupy the northwest and southeast quadrants.](https://developers.google.com/machine-learning/crash-course/images/LinearProblem2.png)

**Figure 2. Is this a linear problem?**

Can you draw a single straight line that neatly separates the sick trees from the healthy trees? No, you can't. This is a nonlinear problem. Any line you draw will be a poor predictor of tree health.



To solve the nonlinear problem shown in Figure 2, create a feature cross. A **feature cross** is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term *cross* comes from [*cross product*](https://wikipedia.org/wiki/Cross_product).) Let's create a feature cross named $$x_3$$ by crossing $$x_1$$ and $$x_2$$:
$$
x_3 = x_1x_2
$$
We treat this newly minted $$x_3$$ feature cross just like any other feature. The linear formula becomes:
$$
y=b+w_1x_1+w_2x_2+w_3x_3
$$
A linear algorithm can learn a weight for $$w_3$$ just as it would for $$w_1$$ and $$w_2$$. In other words, although $$w_3$$ encodes nonlinear information, you don’t need to change how the linear model trains to determine the value of $$w_3$$.

Thanks to stochastic gradient descent, linear models can be trained efficiently. Consequently, supplementing scaled linear models with feature crosses has traditionally been an efficient way to train on massive-scale data sets.



## Crossing One-Hot Vectors

In practice, machine learning models seldom cross continuous features. However, machine learning models do frequently cross one-hot feature vectors. Think of feature crosses of one-hot feature vectors as logical conjunctions. For example, suppose we have two features: country and language. A one-hot encoding of each generates vectors with binary features that can be interpreted as `country=USA, country=France` or `language=English, language=Spanish`. Then, if you do a feature cross of these one-hot encodings, you get binary features that can be interpreted as logical conjunctions, such as:

```
  country:usa AND language:spanish
```

As another example, suppose you bin latitude and longitude, producing separate one-hot five-element feature vectors. For instance, a given latitude and longitude could be represented as follows:

```
  binned_latitude = [0, 0, 0, 1, 0]
  binned_longitude = [0, 1, 0, 0, 0]
```

Suppose you create a feature cross of these two feature vectors:

```
  binned_latitude X binned_longitude
```

This feature cross is a 25-element one-hot vector (24 zeroes and 1 one). The single `1` in the cross identifies a particular conjunction of latitude and longitude. Your model can then learn particular associations about that conjunction.

Now suppose our model needs to predict how satisfied dog owners will be with dogs based on two features:

+ Behavior type (barking, crying, snuggling, etc.)
+ Time of day

If we build a feature cross from both these features:

```
  [behavior type X time of day]
```

then we'll end up with vastly more predictive ability than either feature on its own. For example, if a dog cries (happily) at 5:00 pm when the owner returns from work will likely be a great positive predictor of owner satisfaction. Crying (miserably, perhaps) at 3:00 am when the owner was sleeping soundly will likely be a strong negative predictor of owner satisfaction.

Linear learners scale well to massive data. Using feature crosses on massive data sets is one efficient strategy for learning highly complex models. Neural networks provide another strategy.





# Regularization for Simplicity

## L₂ Regularization

Consider the following **generalization curve**, which shows the loss for both the training set and test set against the number of training iterations.

![The loss function for the training set gradually declines. By contrast, the loss function for the validation set declines, but then starts to rise.](https://developers.google.com/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg)

**Figure 1. Loss on training set and validation set (correct: test set).**

Figure 1 shows a model in which training loss gradually decreases, but test loss eventually goes up. In other words, this generalization curve shows that the model is overfitting to the data in the validation set. Channeling our inner Ockham, perhaps we could prevent overfitting by penalizing complex models, a principle called **regularization**.

In other words, instead of simply aiming to minimize loss (empirical risk minimization):
$$
{\rm minimize(Loss(Data|Model))}
$$
we'll now minimize loss+complexity, which is called **structural risk minimization**:
$$
{\rm minimize(Loss(Data|Model) + complexity(Model))}
$$
Our training optimization algorithm is now a function of two terms: the **loss term**, which measures how well the model fits the data, and the **regularization term**, which measures model complexity.

Machine Learning Crash Course focuses on two common (and somewhat related) ways to think of model complexity:

+ Model complexity as a function of the *weights* of all the features in the model.
+ Model complexity as a function of the *total number of features* with nonzero weights. 

If model complexity is a function of weights, a feature weight with a high absolute value is more complex than a feature weight with a low absolute value.

We can quantify complexity using the ***L2* regularization** formula, which defines the regularization term as the sum of the squares of all the feature weights:
$$
L_2 {\rm \ regularization\ term}=||\boldsymbol w||_2^2=w_1^2+w_2^2+...+w_n^2
$$


In this formula, weights close to zero have little effect on model complexity, while outlier weights can have a huge impact.

For example, a linear model with the following weights:

$$
\{w_1=0.2,w_2=0.5,w_3=5,w_4=1,w_5=0.25,w_6=0.75\}
$$
Has an *L2* regularization term of 26.915:

$$
w_1^2+w_2^2+w_3^2+w_4^2+w_5^2+w_6^2 \\
=0.2^2+0.5^2+5^2+1^2+0.25^2+0.75^2 \\
=0.04+0.25+25+1+0.0625+0.5625 \\
=26.915
$$
But $$w_3$$, with a squared value of 25, contributes nearly all the complexity. The sum of the squares of all five other weights adds just 1.915 to the *L2* regularization term.



## Lambda

Model developers tune the overall impact of the regularization term by multiplying its value by a scalar known as **lambda** (also called the **regularization rate**). That is, model developers aim to do the following:
$$
{\rm minimize(Loss(Data|Model) + \lambda\ complexity(Model))}
$$
Performing *L2* regularization has the following effect on a model

+ Encourages weight values toward 0 (but not exactly 0)
+ Encourages the mean of the weights toward 0, with a normal distribution.

Increasing the lambda value strengthens the regularization effect.

When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:

+ If your lambda value is too high, your model will be simple, but you run the risk of *underfitting* your data. Your model won't learn enough about the training data to make useful predictions.
+ If your lambda value is too low, your model will be more complex, and you run the risk of *overfitting* your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.

The ideal value of lambda produces a model that generalizes well to new, previously unseen data. Unfortunately, that ideal value of lambda is data-dependent, so you'll need to do some tuning.

> a regularization rate of either 0.3 or 1 generally produced the lowest test loss