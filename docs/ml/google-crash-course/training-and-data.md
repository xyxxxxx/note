# Generalization

 Assume that each dot in these figures represents a tree's position in a forest. The two colors have the following meanings:

+ The blue dots represent sick trees.
+ The orange dots represent healthy trees.

![This figure contains about 50 dots, half of which are blue and the other half orange. The orange dots are mainly in the southwest quadrant, though a few orange dots sneak briefly into the other three quadrants. The blue dots are mainly in the northeast quadrant, but a few of the blue dots spill into other quadrants.](https://developers.google.com/machine-learning/crash-course/images/GeneralizationA.png)

**Figure 1. Sick (blue) and healthy (orange) trees.**

Figure 2 shows how a certain machine learning model separated the sick trees from the healthy trees. Note that this model produced a very low loss.

![This Figure contains the same arrangement of blue and orange dots as Figure 1. However, this figure accurately encloses nearly all of the blue dots and orange dots with a collection of complex shapes.](https://developers.google.com/machine-learning/crash-course/images/GeneralizationB.png)

**Figure 2. A complex model for distinguishing sick from healthy trees.**

Figure 3 shows what happened when we added new data to the model. It turned out that the model adapted very poorly to the new data. Notice that the model miscategorized much of the new data.

![Same illustration as Figure 2, except with about a 100 more dots added.  Many of the new dots fall well outside of the predicted model.](https://developers.google.com/machine-learning/crash-course/images/GeneralizationC.png)

**Figure 3. The model did a bad job predicting new data.**

The model shown in Figures 2 and 3 overfits the peculiarities of the data it trained on. An **overfit model** gets a low loss during training but does a poor job predicting new data. Overfitting is caused by making a model more complex than necessary. The fundamental tension of machine learning is between <u>fitting our data well</u>, but also <u>fitting the data as simply as possible</u>.

In modern times, we've formalized Ockham's razor into the fields of **statistical learning theory** and **computational learning theory**. These fields have developed **generalization bounds**--a statistical description of a model's ability to generalize to new data based on factors such as:

+ the complexity of the model
+ the model's performance on training data

The following three basic assumptions guide generalization:

+ We draw examples **independently and identically** (**i.i.d**) at random from the distribution. In other words, examples don't influence each other. (An alternate explanation: i.i.d. is a way of referring to the randomness of variables.)
+ The distribution is **stationary**; that is the distribution doesn't change within the data set.
+ We draw examples from partitions from the **same distribution.**

# Training and Test Sets

The whole data set is divided into two subsets:

+ **training set**—a subset to train a model.
+ **test set**—a subset to test the trained model.

Make sure that your test set meets the following two conditions:

+ Is large enough to yield statistically meaningful results.
+ Is representative of the data set as a whole. In other words, don't pick a test set with different characteristics than the training set.

For example, consider the following figure. Notice that the model learned for the training data is very simple. This model doesn't do a perfect job—a few predictions are wrong. However, this simple model does not overfit the training data.

![Two models: one run on training data and the other on test data.  The model is very simple, just a line dividing the orange dots from the blue dots.  The loss on the training data is similar to the loss on the test data.](https://developers.google.com/machine-learning/crash-course/images/TrainingDataVsTestData.svg?dcb_=0.550957828912408)

**Never train on test data.** If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set.

# Validation Set

We looked at a process of using a test set and a training set to drive iterations of model development. On each iteration, we'd train on the training data and evaluate on the test data, using the evaluation results on test data to guide choices of and changes to various model hyperparameters like learning rate and features. However, doing many rounds of this procedure might cause us to <u>implicitly fit to the peculiarities of our specific test set</u>.

![A workflow diagram consisting of three stages. 1. Train model on training set. 2. Evaluate model on test set. 3. Tweak model according to results on test set. Iterate on 1, 2, and 3, ultimately picking the model that does best on the test set.](https://developers.google.com/machine-learning/crash-course/images/WorkflowWithTestSet.svg)

You can greatly reduce your chances of overfitting by partitioning the data set into the three subsets shown in the following figure:

![A horizontal bar divided into three pieces: 70% of which is the training set, 15% the validation set, and 15% the test set](https://developers.google.com/machine-learning/crash-course/images/PartitionThreeSets.svg)

Use the **validation set** to evaluate results from the training set. Then, use the test set to double-check your evaluation *after* the model has "passed" the validation set. The following figure shows this new workflow:

![Similar workflow to Figure 1, except that instead of evaluating the model against the test set, the workflow evaluates the model against the validation set. Then, once the training set and validation set more-or-less agree, confirm the model against the test set.](https://developers.google.com/machine-learning/crash-course/images/WorkflowWithValidationSet.svg)

In this improved workflow:

1. Pick the model that does best on the validation set.
2. Double-check that model against the test set.

This is a better workflow because it creates fewer exposures to the test set.

# Representation

In traditional programming, the focus is on code. In machine learning projects, the focus shifts to representation. That is, one way developers hone a model is by adding and improving its features.

## Feature Engineering

The left side of Figure 1 illustrates raw data from an input data source; the right side illustrates a **feature vector**, which is the set of floating-point values comprising the examples in your data set. **Feature engineering** means transforming raw data into a feature vector. Expect to spend significant time doing feature engineering.

Many machine learning models must represent the features as real-numbered vectors since the feature values must be multiplied by the model weights.

![Raw data is mapped to a feature vector through a process called feature engineering.](https://developers.google.com/machine-learning/crash-course/images/RawDataToFeatureVector.svg)

**Figure 1. Feature engineering maps raw data to ML features.**

### Mapping numeric values

Integer and floating-point data don't need a special encoding because they can be multiplied by a numeric weight. As suggested in Figure 2, converting the raw integer value 6 to the feature value 6.0 is trivial:

![An example of a feature that can be copied directly from the raw data](https://developers.google.com/machine-learning/crash-course/images/FloatingPointFeatures.svg)

**Figure 2. Mapping integer values to floating-point values.**

### Mapping categorical values

[Categorical features](https://developers.google.com/machine-learning/glossary#categorical_data) have a discrete set of possible values. For example, there might be a feature called `street_name` with options that include:

```
{'Charleston Road', 'North Shoreline Boulevard', 'Shorebird Way', 'Rengstorff Avenue'}
```

Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values.

We can accomplish this by defining a mapping from the feature values, which we'll refer to as the **vocabulary** of possible values, to integers. Since not every street in the world will appear in our dataset, we can group all other streets into a catch-all "other" category, known as an **OOV (out-of-vocabulary) bucket**.

Using this approach, here's how we can map our street names to numbers:

+ map Charleston Road to 0
+ map North Shoreline Boulevard to 1
+ map Shorebird Way to 2
+ map Rengstorff Avenue to 3
+ map everything else (OOV) to 4

However, if we incorporate these index numbers directly into our model, it will impose some constraints that might be problematic:

+ We'll be learning a single weight that applies to all streets. For example, if we learn a weight of 6 for `street_name`, then we will multiply it by 0 for Charleston Road, by 1 for North Shoreline Boulevard, 2 for Shorebird Way and so on. Consider a model that predicts house prices using `street_name` as a feature. It is unlikely that there is a linear adjustment of price based on the street name, and furthermore this would assume you have ordered the streets based on their average house price. Our model needs the flexibility of learning different weights for each street that will be added to the price estimated using the other features.
+ We aren't accounting for cases where `street_name` may take multiple values. For example, many houses are located at the corner of two streets, and there's no way to encode that information in the `street_name` value if it contains a single index.

To remove both these constraints, we can instead create a binary vector for each categorical feature in our model that represents values as follows:

+ For values that apply to the example, set corresponding vector elements to `1`.
+ Set all other elements to `0`.

The length of this vector is equal to the number of elements in the vocabulary. This representation is called a **one-hot encoding** when a single value is 1, and a **multi-hot encoding** when multiple values are 1.

Figure 3 illustrates a one-hot encoding of a particular street: Shorebird Way. The element in the binary vector for Shorebird Way has a value of `1`, while the elements for all other streets have values of `0`.

![Mapping a string value ("Shorebird Way") to a sparse vector, via one-hot encoding.](https://developers.google.com/machine-learning/crash-course/images/OneHotEncoding.svg?dcb_=0.5033437504438356)

**Figure 3. Mapping street address via one-hot encoding.**

This approach effectively creates a Boolean variable for every feature value (e.g., street name). Here, if a house is on Shorebird Way then the binary value is 1 only for Shorebird Way. Thus, the model uses only the weight for Shorebird Way.

Similarly, if a house is at the corner of two streets, then two binary values are set to 1, and the model uses both their respective weights.

### Sparse Representation

Suppose that you had 1,000,000 different street names in your data set that you wanted to include as values for `street_name`. Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a very inefficient representation in terms of both storage and computation time when processing these vectors. In this situation, a common approach is to use a [sparse representation](https://developers.google.com/machine-learning/glossary#sparse_representation) in which only nonzero values are stored. In sparse representations, an independent model weight is still learned for each feature value, as described above.

## Qualities of Good Features

+ Avoid rarely used discrete feature values

  Good feature values should appear more than 5 or so times in a data set. Doing so enables a model to learn how this feature value relates to the label.

+ Prefer clear and obvious meanings

  Each feature should have a clear and obvious meaning to anyone on the project.

+ Don't mix "magic" values with actual data

  Good floating-point features don't contain peculiar out-of-range discontinuities or "magic" values.

+ Account for upstream instability

  The definition of a feature shouldn't change over time.

## Cleaning Data

As an ML engineer, you'll spend enormous amounts of your time tossing out bad examples and cleaning up the salvageable ones. Even a few "bad apples" can spoil a large data set.

### Scaling feature values

**Scaling** means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range (for example, 0 to 1 or -1 to +1). If a feature set consists of only a single feature, then scaling provides little to no practical benefit. If, however, a feature set consists of multiple features, then feature scaling provides the following benefits:

+ Helps gradient descent converge more quickly.
+ Helps avoid the "NaN trap", in which one number in the model becomes a NaN (e.g., when a value exceeds the floating-point precision limit during training), and—due to math operations—every other number in the model also eventually becomes a NaN.
+ Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.

You don't have to give every floating-point feature exactly the same scale. Nothing terrible will happen if Feature A is scaled from -1 to +1 while Feature B is scaled from -3 to +3. However, your model will react poorly if Feature B is scaled from 5000 to 100000.

### Binning

The following plot shows the relative prevalence of houses at different latitudes in California. Notice the clustering—Los Angeles is about at latitude 34 and San Francisco is roughly at latitude 38.

![A plot of houses per latitude. The plot is highly irregular, containing doldrums around latitude 36 and huge spikes around latitudes 34 and 38.](https://developers.google.com/machine-learning/crash-course/images/ScalingBinningPart1.svg?dcb_=0.7048506181411947)

In the data set, `latitude` is a floating-point value. However, it doesn't make sense to represent `latitude` as a floating-point feature in our model.

To make latitude a helpful predictor, let's divide latitudes into "bins" as suggested by the following figure:

![每个纬度的房屋数曲线图。曲线图分为](https://developers.google.com/machine-learning/crash-course/images/ScalingBinningPart2.svg)

Instead of having one floating-point feature, we now have 11 distinct boolean features (`LatitudeBin1`, `LatitudeBin2`, ..., `LatitudeBin11`). Having 11 separate features is somewhat inelegant, so let's unite them into a single 11-element vector. Doing so will enable us to represent latitude 37.4 as follows:

```
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
```

Thanks to binning, our model can now learn completely different weights for each latitude.

### Scrubbing

In real-life, many examples in data sets are unreliable due to one or more of the following:

+ **Omitted values.** For instance, a person forgot to enter a value for a house's age.
+ **Duplicate examples.** For example, a server mistakenly uploaded the same logs twice.
+ **Bad labels.** For instance, a person mislabeled a picture of an oak tree as a maple.
+ **Bad feature values.** For example, someone typed in an extra digit, or a thermometer was left out in the sun.

Once detected, you typically "fix" bad examples by removing them from the data set. To detect omitted values or duplicated examples, you can write a simple program. Detecting bad feature values or labels can be far trickier.

In addition to detecting bad individual examples, you must also detect bad data in the aggregate. Histograms are a great mechanism for visualizing your data in the aggregate. In addition, getting statistics like the following can help:

+ Maximum and minimum
+ Mean and median
+ Standard deviation

### Know your data

Follow these rules:

+ Keep in mind what you think your data should look like.
+ Verify that the data meets these expectations (or that you can explain why it doesn’t).
+ Double-check that the training data agrees with other sources (for example, dashboards).

Treat your data with all the care that you would treat any mission-critical code. Good ML relies on good data.

