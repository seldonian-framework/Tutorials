---
layout: tutorial
permalink: /tutorials/new_supervised_model/
prev_url: /tutorials/science_GPA_tutorial/
prev_page_name: (E) Science paper GPA tutorial
next_url: /tutorials/pytorch_mnist/
next_page_name: (G) Creating your first Seldonian PyTorch model
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Tutorial F: Creating a new Seldonian supervised learning model </h2>
    <hr class="my-4">
    <h3>Introduction</h3>
    <p>
        This tutorial is intended to help you understand how to integrate a supervised machine learning model with the Seldonian Toolkit. As you may have noticed from the <a href="{{ "/overview/#algorithm" | relative_url}}">Overview</a> page, Seldonian algorithms are very general. They are, at least in principle, compatible with any machine learning model. The Seldonian Toolkit implements a particular Seldonian algorithm, and the current implementation of this algorithm is such that the machine learning model one adopts must meet these two requirements: 
    </p>
    <ul>
        <li>
            The model must be parametric, meaning that it is described by a fixed, finite number of parameters independent of the size of the dataset. 
        </li>
        <li>
            The model must be differentiable. Specifically, the model's forward pass (or "predict" function) must be differentiable.
        </li>
    </ul>
    <p> 
        The reason for these requirements is that we use gradient descent to find solutions that solve the optimization problem with added behavioral constraints (see the <a href="{{ "/tutorials/alg_details_tutorial" | relative_url}}">Algorithm details tutorial</a> for details). Examples of parametric and differentiable supervised learning models include linear regression, logistic regression and many neural networks. Tree-based models like decision trees and random forests are examples of nonparametric models, according to our definition of parametric above. Linear support vector machines are non-differentiable because they predict a binary value, i.e., whether a data point is on one side or the other of the proposed hyperplane. For classification models, if the output of the model is a probability, such as in logistic regression, then generally that model is differentiable. 
    </p>
   
   <p>
       In this tutorial, we will demonstrate how to integrate a model using an example. The Engine library already contains a few example Seldonian models, including linear regression, logistic regression (binary and multi-class), and a few neural networks. We will describe how the binary logistic regression model was constructed as a guide for how you can implement your own models with the toolkit
   </p>
<h3> Implementing a (binary) logistic regression model with the toolkit </h3>

<p>
    <a href="https://en.wikipedia.org/wiki/Logistic_regression#Model">Logistic regression</a> is used for classification, a sub-regime of supervised learning. In this section, we will build a binary logistic regression model. It is binary in the sense that the two possible label classes are 0 and 1. You might wonder why we cannot simply use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression">Scikit-learn's logistic regression model</a> in the toolkit. We could, but the reason why we do not will become apparent shortly.
</p>   
<p>
First, make sure you have the latest version of the engine installed. </p>    
{% highlight javascript %}
$ pip install --upgrade seldonian-engine
{% endhighlight javascript %}

<p>
    Models are implemented as Python classes in the toolkit. There are two basic requirements for creating a new Seldonian model class:
    <ol>
        <li>The class must inherit from the appropriate model base class: <code class="highlight">seldonian.models.RegressionModel</code> for regression models and <code class="highlight">seldonian.models.ClassificationModel</code> for classification models. The class must call the init method of the parent class in its own init method. </li>
        
        <li>The class must have a <code class="highlight">predict()</code> method in which it takes as input a weight vector, <code class="highlight">theta</code>, and a feature matrix, <code class="highlight">X</code>, and outputs the predicted continuous-valued label (for regression) or the probabilities of the predicted classes (for classification) for each sample row in <code class="highlight">X</code>. For the special case of binary classification, the model should output the probability of predicting the positive class for each input sample. This method is often referred to as the "forward pass" for a neural network. </li>
    </ol> 
    The second requirement is why we do not use Scikit-learn's logistic regression model. Their model's <code class="highlight">predict()</code> method does not take the model weights as an input, but instead uses the model's internally set weights to do the prediction. There is not a way, as far as we are aware, to manually set the weights of the Scikit-learn model. Their model weights get set when the model is trained. 
</p>
<p>
    Given these two requirements, the bulk of the work in creating a new Seldonian model class is typically in defining the <code class="highlight">predict()</code> method. For logistic regression, there is a straightforward equation for predicting the probability of the positive class: $$\hat{Y}(\theta,X) = \sigma\left(\theta^{T}X\right) + b,$$
    where $\hat{Y}$ are the predicted probabilities of the positive class, $\sigma(x) = \frac{1}{1+e^{-x}}$ is the sigmoid function, $\theta$ are the model weights, $X$ are the features, and $b$ is the intercept term (also called bias term). We now have everything we need to code up our new model class, which we will name <code class="highlight">MyBinaryLogisticRegressionModel</code>.
</p>

{% highlight python %}
from seldonian.models.models import ClassificationModel

class MyBinaryLogisticRegressionModel(ClassificationModel):
    def __init__(self):
        """ Implements binary logistic regression """
        super().__init__()
        self.has_intercept = True

    def predict(self,theta,X):
        """ Predict the probability of 
        having positive class label for each data point
        in X. 
            i = number of datapoints
            j = number of features (including bias term, if provided)

        :param theta: The parameter weights
        :type theta: array of length j 
        :param X: The features 
        :type X: array of shape (i,j)
        :return: predictions for each class each observation
        :rtype: array of length i
        """
        Z = theta[0] + (X @ theta[1:]) 
        Y_pred = 1/(1+np.exp(-Z))
        return Y_pred
{% endhighlight python %}
<p>
    First, notice that we meet requirement 1 by calling the parent class' <code class="highlight">__init__()</code> method from within our <code class="highlight">__init__()</code> method. We set <code class="highlight">self.has_intercept = True</code>, which tells the toolkit that there will be a bias term. This flag is only used when finding an initial solution to use in the optimization process if none is provided by the user. You could optionally define a method of this class that returns an initial solution to use. Requirement 2 is met with the implementation of the <code class="highlight">predict()</code> method. Notice that the bias term <code class="highlight">theta[0]</code> is the first element of the parameter weight array. <code class="highlight">X @ theta[1:]</code> is one way to express the matrix multiplication $\theta^{T}X$ in Python.
</p>

<p>
    At this point, this model is ready to use in the toolkit. This model is pretty minimal. It is designed to show you the minimum required aspects of a Seldonian model. An optional method you could implement is the gradient of the <code class="highlight">predict()</code> method. By providing this in the spec object via the <code class="highlight">custom_primary_gradient_fn</code> parameter, you may be able to speed up candidate selection. The engine will automatically find the gradient if you do not provide one, but it can be slow depending on the implementation of your <code class="highlight">predict()</code> method.
</p>
     
<h3>Summary</h3>
<p>In this tutorial, we demonstrated how to integrate a machine learning model with the Seldonian toolkit. We hope the example implementation of the logistic regression model will make it easier for you to implement your own safe and fair machine learning models.  </p>

</div> 