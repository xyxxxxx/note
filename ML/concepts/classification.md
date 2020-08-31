# Logistic Regression

## Calculating a Probability

Logistic regression is an extremely efficient mechanism for calculating probabilities. Logistic regression models use the sigmoid function, defined as follows, to produces output that always falls between 0 and 1:
$$
y=\frac{1}{1+e^{-z}}
$$
The sigmoid function yields the following plot:

![Sigmoid function. The x axis is the raw inference value. The y axis extends from 0 to +1, exclusive.](https://developers.google.com/machine-learning/crash-course/images/SigmoidFunction.png)

If `z` represents the output of the linear layer of a model trained with logistic regression, then sigmoid(z) will yield a value (a probability) between 0 and 1. In mathematical terms:
$$
y′=\frac{1}{1+e^{-z}}
$$
where:

+ $$y'$$ is the output of the logistic regression model for a particular example.
+ $$z=b+w_1x_1+w_2x_2+…+w_Nx_N$$

Note that *z* is also referred to as the *log-odds* because the inverse of the sigmoid states that `z` can be defined as the log of the probability of the "1" label (e.g., "dog barks") divided by the probability of the "0" label (e.g., "dog doesn't bark"):
$$
z=\log (\frac{y}{1-y})
$$


## Loss and Regularization

### Loss function

The loss function for logistic regression is **Log Loss**, which is defined as follows:
$$
{\rm Log\ Loss}=\sum_{(x,y)∈D}−y\log⁡(y′)−(1−y)\log⁡(1−y′)
$$
where:

+ $$(x,y)∈D$$ is the data set containing many labeled examples, which are $$(x,y)$$ pairs.
+ $$y$$ is the label in a labeled example. Since this is logistic regression, every value of $$y$$ must either be 0 or 1.
+ $$y′$$ is the predicted value (somewhere between 0 and 1), given the set of features in $$x$$.



## Regularization in Logistic Regression

Regularization is extremely important in logistic regression modeling. Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions. Consequently, most logistic regression models use one of the following two strategies to dampen model complexity:

+ L2 regularization.
+ Early stopping: limiting the number of training steps or the learning rate.

Imagine that you assign a unique id to each example, and map each id to its own feature. If you don't specify a regularization function, the model will become completely overfit. That's because the model would try to drive loss to zero on all examples and never get there, driving the weights for each indicator feature to +infinity or -infinity. This can happen in high dimensional data with feature crosses, when there’s a huge mass of rare crosses that happen only on one example each.

Fortunately, using L2 or early stopping will prevent this problem.





# Classification

## Thresholds

Logistic regression returns a probability. You can use the returned probability "as is" (for example, the probability that the user will click on this ad is 0.00023) or convert the returned probability to a binary value (for example, this email is spam).

A logistic regression model that returns 0.9995 for a particular email message is predicting that it is very likely to be spam. Conversely, another email message with a prediction score of 0.0003 on that same logistic regression model is very likely not spam. However, what about an email message with a prediction score of 0.6? In order to map a logistic regression value to a binary category, you must define a **classification threshold** (also called the **decision threshold**). Thresholds are problem-dependent, and are therefore values that you must tune.



## Accuracy

Accuracy is one metric for evaluating classification models. Formally, **accuracy** has the following definition:
$$
{\rm Accuracy=\frac{Number\ of\ correct\ predictions}{Total\ number\ of\ predictions}}
$$
For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:
$$
{\rm Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$
Where *TP* = True Positives, *TN* = True Negatives, *FP* = False Positives, and *FN* = False Negatives.

Let's try calculating accuracy for the following model that classified 100 tumors as malignant (the positive class) or benign (the negative class):

| **True Positive (TP):**<br />Reality: Malignant<br />ML model predicted: Malignant<br />**Number of TP results: 1** | **False Positive (FP):**<br />Reality: Benign<br />ML model predicted: Malignant<br />**Number of FP results: 1** |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **False Negative (FN):**<br />Reality: Malignant<br />ML model predicted: Benign<br />**Number of FN results: 8** | **True Negative (TN):**<br />Reality: Benign<br />ML model predicted: Benign<br />**Number of TN results: 90** |

$$
{\rm Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}=\frac{1+90}{1+90+1+8}=0.91
$$

Accuracy comes out to 0.91, or 91% (91 correct predictions out of 100 total examples). While 91% accuracy may seem good at first glance, another tumor-classifier model that always predicts benign would achieve the exact same accuracy (91/100 correct predictions) on our examples.

Accuracy alone doesn't tell the full story when you're working with a **class-imbalanced data set**, like this one, where there is a significant disparity between the number of positive and negative labels.



## Precision and Recall

**Precision** is defined as follows:
$$
{\rm Precision}=\frac{TP}{TP+FP}
$$
Let's calculate precision for our tumor classifier from the previous section:

| **True Positives (TPs): 1**  | **False Positives (FPs): 1** |
| ---------------------------- | ---------------------------- |
| **False Negatives (FNs): 8** | **True Negatives (TNs): 90** |

$$
{\rm Precision}=\frac{TP}{TP+FP}=\frac{1}{1+1}=0.5
$$



**Recall** is defined as follows:
$$
{\rm Recall}=\frac{TP}{TP+FN}
$$
Let's calculate precision for our tumor classifier:
$$
{\rm Recall}=\frac{TP}{TP+FN}=\frac{1}{1+8}=0.11
$$


To fully evaluate the effectiveness of a model, you must examine **both** precision and recall. Unfortunately, precision and recall are often in tension. That is, improving precision typically reduces recall and vice versa. Explore this notion by looking at the following figure, which shows 30 predictions made by an email classification model. Those to the right of the classification threshold are classified as "spam", while those to the left are classified as "not spam".



## ROC Curve and AUC

An **ROC curve** (**receiver operating characteristic curve**) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

+ True Positive Rate
+ False Positive Rate

**True Positive Rate** (**TPR**) is a synonym for recall and is therefore defined as follows:
$$
TPR=\frac{TP}{TP+FN}
$$
**False Positive Rate** (**FPR**) is defined as follows:
$$
FPR=\frac{FP}{FP+TN}
$$
An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.

![ROC Curve showing TP Rate vs. FP Rate at different classification thresholds.](https://developers.google.com/machine-learning/crash-course/images/ROCCurve.svg)



**AUC** stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).

![AUC (Area under the ROC Curve).](https://developers.google.com/machine-learning/crash-course/images/AUC.svg)

AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.

AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.

AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

AUC is desirable for the following two reasons:

+ AUC is **scale-invariant**. It measures how well predictions are ranked, rather than their absolute values.
+ AUC is **classification-threshold-invariant**. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:

+ **Scale invariance is not always desirable.** For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that.
+ **Classification-threshold invariance is not always desirable.** In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.



