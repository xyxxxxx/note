When training data set is small and slightly noisy, overfitting is a real concern.

# L2 Regularization

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
But $w_3$ , with a squared value of 25, contributes nearly all the complexity. The sum of the squares of all five other weights adds just 1.915 to the *L2* regularization term.

# Lambda

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

# L1 Regularization

Sparse vectors often contain many dimensions. Creating a feature cross results in even more dimensions. Given such high-dimensional feature vectors, model size may become huge and require huge amounts of RAM.

In a high-dimensional sparse vector, it would be nice to encourage weights to drop to exactly `0` where possible. A weight of exactly 0 essentially removes the corresponding feature from the model. Zeroing out features will save RAM and may reduce noise in the model.

For example, consider a housing data set that covers not just California but the entire globe. Bucketing global latitude at the minute level (60 minutes per degree) gives about 10,000 dimensions in a sparse encoding; global longitude at the minute level gives about 20,000 dimensions. A feature cross of these two features would result in roughly 200,000,000 dimensions. Many of those 200,000,000 dimensions represent areas of such limited residence (for example, the middle of the ocean) that it would be difficult to use that data to generalize effectively. It would be silly to pay the RAM cost of storing these unneeded dimensions. Therefore, it would be nice to encourage the weights for the meaningless dimensions to drop to exactly 0, which would allow us to avoid paying for the storage cost of these model coefficients at inference time.

We might be able to encode this idea into the optimization problem done at training time, by adding an appropriately chosen regularization term.

Would L2 regularization accomplish this task? Unfortunately not. L2 regularization encourages weights to be small, but doesn't force them to exactly 0.0.

An alternative idea would be to try and create a regularization term that penalizes the count of non-zero coefficient values in a model. Increasing this count would only be justified if there was a sufficient gain in the model's ability to fit the data. Unfortunately, while this count-based approach is intuitively appealing, it would turn our convex optimization problem into a non-convex optimization problem (one of NP-hard problem). So this idea, known as L0 regularization isn't something we can use effectively in practice.

However, there is a regularization term called L1 regularization that serves as an approximation to L0, but has the advantage of being convex and thus efficient to compute. So we can use L1 regularization to encourage many of the uninformative coefficients in our model to be exactly 0, and thus reap RAM savings at inference time.

# L1 vs L2

|                                                 | L1                                        | L2                                          |
| ----------------------------------------------- | ----------------------------------------- | ------------------------------------------- |
| penalize                                        | $|w|$                                   | $w^2$                                     |
| derivative                                      | 1                                         | $2w$                                      |
| difference between <br />training and test loss | low                                       | high                                        |
| weight absolute value                           | low                                       | high                                        |
|                                                 |                                           |                                             |
| possible result of                              | dampen weights                            |                                             |
| increasing lambda                               | cannot converge                           |                                             |
|                                                 | higher loss                               |                                             |
| if two strongly correlated features             | assgin 0 to one weight                    | assgin almost the same value to two weights |
| applicable for                                  | models with many non-informative features |                                             |
|                                                 | wide models                               |                                             |
|                                                 |                                           |                                             |

**About derivative**

You can think of the derivative of L2 as a force that removes x% of the weight every time. Even if you remove x percent of a number *billions of times*, the diminished number will still never quite reach zero. At any rate, L2 does not normally drive weights to zero.

You can think of the derivative of L1 as a force that subtracts some constant from the weight every time. However, thanks to absolute values, L1 has a discontinuity at 0, which causes subtraction results that cross 0 to become zeroed out. For example, if subtraction would have forced a weight from +0.1 to -0.2, L1 will set the weight to exactly 0. Eureka, L1 zeroed out the weight.

