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

+ $y'$ is the predicted label
+ $b$ is the bias, sometimes referred to as $w_0$ 
+ $w_i$ is the weight of feature i
+ $x_i$ is a feature (a known input).

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

To solve the nonlinear problem shown in Figure 2, create a feature cross. A **feature cross** is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term *cross* comes from [*cross product*](https://wikipedia.org/wiki/Cross_product).) Let's create a feature cross named $x_3$ by crossing $x_1$ and $x_2$ :
$$
x_3 = x_1x_2
$$
We treat this newly minted $x_3$ feature cross just like any other feature. The linear formula becomes:
$$
y=b+w_1x_1+w_2x_2+w_3x_3
$$
A linear algorithm can learn a weight for $w_3$ just as it would for $w_1$ and $w_2$ . In other words, although $w_3$ encodes nonlinear information, you don’t need to change how the linear model trains to determine the value of $w_3$ .

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

