---
layout: tutorial
permalink: /tutorials/custom_base_variable_tutorial/
prev_url: /tutorials/gridworld_RL_tutorial/
prev_page_name: (H) Introduction to reinforcement learning with the Seldonian Toolkit
next_url: /tutorials/efficient_deep_networks/
next_page_name: (J) Efficiently training deep Seldonian networks
title: Seldonian \| Tutorial I
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    
<h2 align="center" class="mb-3">Tutorial I: Creating custom base variables in behavioral constraints</h2>

<hr class="my-4">
<p> This tutorial is currently being updated to reflect the latest changes to the Engine API. Thank you for your patience. </p>
<!-- <p> The purpose of this tutorial is to help you understand how to extend the functionality of string-based behavioral constraints to suit your custom constraints. If the answer to either of these questions is "yes" or "maybe," then keep reading:
<ul>
    <li>Does your constraint involve a statistical function (such as "precision") that is not currently supported by the engine? </li>
    <li>Do you have an alternative method for bounding a base variable in the constraint that is not currently supported by the engine? </li>
</ul>
</p>
<h3 class="my-4">Introduction</h3>

<p>
In cases where behavioral constraints can be expressed as mathematical inequalities, as is the case for some popular definitions of fairness (for example, see <a href="https://fairware.cs.umass.edu/papers/Verma.pdf">Verma et al., 2018</a>), the constraints can be passed to the Seldonian Engine as strings with specific syntax. Specifically, these strings can contain five types of things:
</p>
<ol>
<li>The mathematical operators: <code class='highlight'>(+,-,*,/)</code> </li>
<li>These four native Python math functions: <code class='highlight'>min(),max(),abs(),exp()</code> </li>
<li>Constant numbers, such as $-0.5$ or 7</li>
<li>The inequality strings "<=" or ">=" (optional)</li>
<li>Special strings that trigger a call to a function, such as "FPR" (false positive rate)</li>
</ol> 
<p>
An example constraint string that uses all five of these types is: "max(FPR/TPR,FNR/TNR) <= 0.5", which translates to "ensure that whatever is larger out of false positive rate divided by true positive rate vs. false negative rate divided by true negative rate is less than or equal to 0.5."
</p>
<p>
In this tutorial, we will focus on #5 in the list. Specifically, we will demonstrate how to define custom strings that map to your own functions. This will allow you to customize your constraints beyond what the engine already supports. 
</p>
<p>
The engine supports several built-in strings that trigger a call to a function. These are referred to as <a href="{{ "/glossary/#measure_function" | relative_url}}">measure functions</a> and include strings like "Mean_Squared_Error", "FPR" (false positive rate) and "J_pi_new" (the performance of the new policy in the reinforcement learning setting). These are the base variables in the <a href="{{ "/tutorials/alg_details_tutorial/#parse_tree" | relative_url}}">parse tree</a>. The complete list of existing built-in measure functions is <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.parse_tree.operators.html#seldonian.parse_tree.operators.measure_functions_dict">here</a>, separated by regime and subregime (e.g., classification vs. regression). We designed the engine to make it straightforward for developers to add new measure functions. If your desired constraint involves a statistical function not listed there, for example <a href="https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)">precision</a>, then the section <a href="#new_measure_function">Adding a new measure function</a> will help you create it.
</p>
<p> 
However, it is possible that creating a new measure function will not be sufficient for your use case. Recall from the <a href="{{ "/tutorials/alg_details_tutorial/#parse_tree" | relative_url}}">parse tree</a> discussion that confidence bounds are calculated on base variables and then propagated to the root of the tree to get the upper bound on the overall constraint, $g(\theta)$. The confidence bounds are first calculated on the mean of the base variable. If your desired constraint involves bounding something other than the mean of one of these base variables, then you will not be able to define your constraint in terms of measure functions. In that case, you will need to create a new type of base variable, which we call a "custom base variable." In the <a href="#custom_base_variable">Creating a custom base variable</a> section below, we demonstrate how to do this for a constraint involving the <a href="https://en.wikipedia.org/wiki/Expected_shortfall">conditional value at risk (CVaR)</a> statistic. 
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 class="my-4" id="new_measure_function"> Adding a new measure function </h3>
<p>
 The currently programmed measure functions are listed <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.parse_tree.operators.html#seldonian.parse_tree.operators.measure_functions_dict">here</a> by regime and subregime. Let's say your desired constraint involves precision, which is not an existing measure function, and the string you want to use to represent it is "PREC". And say that the constraint you are considering is that you want the precision of the model to differ by no more than 10% between males and females in the dataset.  This constraint could be expressed to the engine as: 

{% highlight python %}
abs((PREC | [M]) - (PREC | [F])) - 0.1,
{% endhighlight python %}

where  "M" and "F" refer to the male and female columns of your dataset, respectively. Here we will go through the steps for adding the functionality to allow this constraint string in the engine. This is just one example, showing how you can add functionality for the engine to work with other variables/strings. 
</p>
<ol>
<li>Fork the engine repository: https://github.com/seldonian-toolkit/Engine </li>
<li>Define a string to represent the precision operator that you will type into your constraint. In this example, we will call it "PREC" </li>
<li>Add "PREC" to the list of supervised classification measure functions in the <code class='highlight'>measure_functions_dict</code> here: <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py#L74">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py#L74</a></li>
<li>Add the following block of code to the <code class='highlight'>evaluate_statistic()</code> function in this module: <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/objectives.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/objectives.py</a>:

{% highlight python %}
if statistic_name == 'PREC':
    return Precision(
        theta,data_dict['features'],data_dict['labels'])
{% endhighlight python %}
This points to a function <code class='highlight'>Precision()</code> which we will implement shortly to calculate the mean precision value over a set of points, returning a float.
 </li>

 <li>Add the following block of code to the <code class='highlight'>sample_from_statistic()</code> function in the same file:

{% highlight python %}
if statistic_name == 'PREC':
    return vector_Precision(
        theta,data_dict['features'],data_dict['labels'])
{% endhighlight python %}
This points to a function <code class='highlight'>vector_Precision()</code> which we will implement shortly to calculate the precision values on each point in a set of points, returning a vector.
 </li>
 <li>Add a new function <code class='highlight'>Precision()</code> to the same file that calculates the precision on a set of points and returns a float. Note that we will use the existing functions in this file called <code class='highlight'>True_Positive_Rate()</code> and <code class='highlight'>False_Positive_Rate()</code> in our implementation.

{% highlight python %}
def Precision(model,theta,X,Y):
    """ Calculate the precision 
    on whole sample

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :return: Precision
    :rtype: float
    """
    tpr = True_Positive_Rate(model,theta,X,Y)
    fpr = False_Positive_Rate(model,theta,X,Y)
    res = tpr/(tpr + fpr)
    return res
{% endhighlight python %}
 </li>
 <li>Likewise, add a new function <code class='highlight'>vector_Precision()</code> to the same file that calculates the precision on each point in a set of points and returns a vector of floats. The outputs of this function are the $\widehat z(\theta,D)$ used to obtain the bound on the base variable in the <a href="{{ "/tutorials/alg_details_tutorial/#parse_tree" | relative_url}}">parse tree</a>. As before, we will use existing functions of the same file in our implementation of the vector precision.

{% highlight python %}
def vector_Precision(model,theta,X,Y):
    """ Calculate the probabilistic precision 
    on each point in a sample of points

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :return: Precision values
    :rtype: array of floats
    """
    tp_values = vector_True_Positive_Rate(model,theta,X,Y)
    fp_values = vector_False_Positive_Rate(model,theta,X,Y)
    res = tp_values/(tp_values + fp_values)
    return res
{% endhighlight python %}
 </li>
</ol>
<p>
At this point, you can now use the "PREC" in your constraint strings that you provide to the engine. The example constraint we considered at the beginning of this section: 
{% highlight python %}
abs((PREC | [M]) - (PREC | [F])) - 0.1,
{% endhighlight python %}
will now be correctly interpreted as: "ensure that the precision of the model should differ by no more than 10% between males (M) and females (F) in the dataset." The probability that this constraint will be upheld is controlled by what you set $\delta$ to, as in all other constraints. 
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 class="my-4" id="custom_base_variable"> Creating a custom base variable </h3>

<p>
If the functionality you need cannot be achieved by creating a new measure function, you may be able to achieve it by implementing a new base variable class. Let's consider an example where we have a regression problem with a single constraint: we want to ensure that the <a href="https://en.wikipedia.org/wiki/Expected_shortfall">conditional value at risk (CVaR)</a> of the squared residual is below some value. To provide a high-confidence guarantee of this constraint, the Seldonian algorithm will need to put a bound on the CVaR of the squared residual (CVaRSQE). Measure functions are used when we want to bound the mean of a quantity. For example, the mean of the squared residual is an existing measure function we call "Mean_Squared_Error" in the Engine library. That is not what we want to do here. Instead, we will need to write our own bounding function for the CVaRSQE. First, let's review what the CVaR statistic is and how it can be bounded.
</p>

<div class="container inset-box border border-dark border-2 p-3 my-2">
        <h2>$\text{CVaR}_{\alpha}$ statistic</h2>
        <p>Figure 1 shows the intuition behind the $\text{CVaR}_\alpha(X)$ statistic. The formal definition is more complicated as it can handle discrete and hybrid probability distributions, but we will not cover that here. In this tutorial, we will only consider $\text{CVaR}_\alpha(X)$ for continuous distributions. For more details on the $\text{CVaR}_\alpha(X)$ statistic, see: <a href="https://en.wikipedia.org/wiki/Expected_shortfall.">https://en.wikipedia.org/wiki/Expected_shortfall</a>
        </p>
        <div align="center">
            <figure class='mt-4'>
                <img src="{{ "/assets/img/cvar_alpha_definition.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-5 bg-white" style="width: 70%"  alt="parse tree"> 
                <figcaption class="figure-caption"><b>Figure 1</b> - Probability density function of a random variable $X$, where $E(X)$ is the expected value, $\text{VaR}(X)$ (frequently called $\text{VaR}_\alpha(X)$) is the position where $100\alpha$ percent of the distribution is greater than or equal to $\text{VaR}(X)$, and $\text{CVaR}(X)$ (frequently called $\text{CVaR}_\alpha(X)$) is the expected value given that the sample is at least $\text{VaR}(X)$. Source: <a href="https://en.wikipedia.org/wiki/Expected_shortfall">https://en.wikipedia.org/wiki/Expected_shortfall</a>.  </figcaption>
            </figure> 
        </div>
<p>
 We will use the concentration inequalities derived by Thomas and Learned-Miller (2019), Theorems 3 and 4, to define the bounds on $\text{CVaR}_\alpha(X)$ of the squared residual. In the <code class='glossary-term'>safety test</code>, we will use the exact form of the inequalities. Adopting variables from the Seldonian formalism, the upper bound we will implement for the <code class='glossary-term'>safety test</code> is:
</p>
$$
\begin{equation} 
Z_{N_{\text{safety}}+1} - \frac{1}{\alpha} \sum_{i=1}^{N_{\text{safety}}} (Z_{i+1}-Z_i) \left( \frac{i}{N_{\text{safety}}} - \sqrt{ \frac{ \ln(1/{\delta})}{2N_{\text{safety}}} } - (1-\alpha)\right)^{+},
\label{st_upper}
\end{equation}
$$ 
<p>
where $Z_1,\dotsc,Z_{N_{\text{safety}}}$ are the squared residuals from each of the $N_{\text{safety}}$ data points sorted in ascending order, $Z_{N_{\text{safety}}+1}=b$ is a theoretical upper bound on the squared residual which we will define below, $x^{+}:= \operatorname{max}(0,x)$, $\alpha$ is the confidence level above which the CVaR is calculated, $\delta$ is the confidence level for the safety constraint, and $N_{\text{safety}}$ is the size of the safety dataset. The lower bound for the <code class='glossary-term'>safety test</code> is:
</p>

$$ 
\begin{equation}
Z_{N_{\text{safety}}} - \frac{1}{\alpha} \sum_{i=0}^{N_{\text{safety}}-1} (Z_{i+1}-Z_i) \left( \operatorname{min}\left(1,\frac{i}{N_{\text{safety}}} + \sqrt{ \frac{ \ln(1/{\delta})}{2N_{\text{safety}}} } \right) - (1-\alpha)\right)^{+}, 
\label{st_lower}
\end{equation}
$$ 
<p>
where $Z_1,\dotsc,Z_{N_{\text{safety}}}$ are again the sorted squared residuals and $Z_0=a$ is a theoretical lower bound on the squared residual, which is $0$ as we will show below.
</p>
<p>
As discussed in the <a href="{{"/tutorials/alg_details_tutorial/#candidate_selection" | relative_url}}">candidate selection section of the algorithm details tutorial</a>, in <code class='glossary-term'>candidate selection</code> we want to search for a solution that optimizes the primary objective and is predicted to pass the <code class='glossary-term'>safety test</code>. In the $t$-test confidence bound, we inflated the confidence interval with a factor of 2 and used $N_{\text{safety}}$ instead of $N_{\text{cand}}$ to make a good prediction the <code class='glossary-term'>safety test</code> will pass. Here, we will do something similar. The upper bound we will implement for <code class='glossary-term'>candidate selection</code> is:
</p>
$$ 
\begin{equation}
Z_{N_{\text{cand}}+1} - \frac{1}{\alpha} \sum_{i=1}^{N_{\text{cand}}} (Z_{i+1}-Z_i) \left( \frac{i}{N_{\text{cand}}} - 2\sqrt{ \frac{ \ln(1/{\delta})}{2N_{\text{safety}}} } - (1-\alpha)\right)^{+}, 
\label{cs_upper} 
\end{equation}
$$ 
<p>
where $N_{\text{cand}}$ is the size of the candidate dataset, $N_{\text{safety}}$ is the size of the safety dataset, and all of the other terms have the same meaning as above. Notice the factor of $2$ that now appears before the square root term in equation \eqref{cs_upper}. Similarly, the lower bound we will implement for <code class='glossary-term'>candidate selection</code> is:
</p>
$$ 
\begin{equation}
Z_{N_{\text{cand}}} - \frac{1}{\alpha} \sum_{i=0}^{N_{\text{cand}}-1} (Z_{i+1}-Z_i) \left( \operatorname{min}\left(1,\frac{i}{N_{\text{cand}}} + 2\sqrt{ \frac{ \ln(1/{\delta})}{2N_{\text{safety}}} } \right) - (1-\alpha)\right)^{+}, 
\label{cs_lower}
\end{equation}
$$
</div> 

<p>
We now have the math to implement $\text{CVaR}_\alpha(X)$ as a custom base variable. It is important to note that the CVaR base variable we create will work for any regression dataset and appropriate model. To illustrate how it works, we will adopt a specific data set and machine learning model going forward. For simplicity, we will hardcode $\alpha=0.1$ in this base variable. One could make $\alpha$ part of the base variable that the user specifies if desired, but we will not implement that here. Before we proceed, we need to determine the theoretical lower and upper bounds on the squared residual, $a$ and $b$, respectively. This depends on the data distribution and the model, neither of which we have defined yet. In this tutorial, we will generate synthetic data using normal distributions. We will clip the labels to ensure that they are bounded. Specifically, we will generate data as follows:
</p>
$$
\begin{equation}
X = N(0,1), Y = \operatorname{clip}(X + N(0,0.2),(-3,3)) 
\label{data_distribution}
\end{equation}
$$
<p>
where $N(\mu,\sigma^2)$ is a normal distribution with mean $\mu$ and variance $\sigma^2$, and $\operatorname{clip}(y,(y_{\text{min}},y_{\text{max}}))$ clips the values of variable $y$ to $(y_{\text{min}},y_{\text{max}})$. We plot this distribution in Figure 2, using $y_{\text{min}}=-3,y_{\text{max}} = 3$ and $10^4$ points.
</p>

<div align="center">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/cvar_data_distribution.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-5 bg-white" style="width: 40%"  alt="parse tree"> 
        <figcaption class="figure-caption"><b>Figure 2</b> - $10^4$ samples from the data distribution we will use for calculating the $\text{CVaR}_\alpha(X)$ of the squared residual. </figcaption>
    </figure> 
</div>
<p>
  For the underlying machine learning model, we will adopt a bounded linear regression model. For details on why we need to bound the outputs of the model and how we do that, see the "Bounding the squared residuals" box below. Note that this is, in general, not a necessary step for creating a custom base node; it happens to be necessary for the $\text{CVaR}_\alpha(X)$ statistic. Feel free to skip over the box below if you are not interested in seeing these details. 
</p>
<div class="container inset-box border border-dark border-2 p-3 my-2">
        <h2>Bounding the squared residuals</h2>
        <p>
  A standard linear regression model trained on the data distribution described in equation \eqref{data_distribution} and shown in Figure 2 could predict the exact value of the label for any given value of $X$, so the theoretical lower bound on the squared residual, $a$, is 0. However, the theoretical upper bound, $b$, is infinite because the predicted value from the model is not bounded. Plugging in $Z_{N_{\text{safety}}+1}=b=\infty$ into equation \eqref{st_upper} or $Z_{N_{\text{cand}}+1}=b=\infty$ into equation \eqref{cs_upper} would result in an infinite upper bound on the $\text{CVaR}_\alpha(X)$ statistic, which would mean that the safety test could never pass. In order to provide some finite upper bound on the squared residual, we need to bound the outputs of the linear regression model as well. Clipping is not an option because it would render our model's predict function undifferentiable, making us unable to use gradient descent during <code class='glossary-term'>candidate selection</code>. Instead, we can bound the model predictions using a sigmoid function, like:  
</p>

$$
\begin{equation}
\widehat{y}' = \sigma(X\cdot\theta)(d-c)+c, 
\label{squashed_model}
\end{equation}
$$
<p>
  where $\sigma(x) = \frac{1}{1+e^{-x}}$, $X \cdot \theta$ is what an unmodified linear regression model would predict, and $[c,d]$ are the bounds we want the model predictions to fall within. This is a desirable modification to the model because it allows us to bound the model while keeping the predictions in the middle of the domain very close to those of the original model. For example, let's consider the model $\widehat{y}=2x$ and say we want to bound the predictions to $\widehat{y}' \in [-5,5]$. This is what the original model and the bounded model would look like:
</p>
  <div align="center">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/cvar_model_sigmoid.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-5 bg-white" style="width: 40%"  alt="parse tree"> 
        <figcaption class="figure-caption"><b>Figure 3</b> - Bounding the model $\widehat{y}=2x$ using the sigmoid function. The bounded model closely matches the unbounded model in the middle of the domain. </figcaption>
    </figure> 
    </div>

<p>
How we choose the bounds $[c,d]$ for the model predictions is determined by two criteria:
</p>
<ol>
<li>We want the model to easily be able to output all of the possible labels within $[y_{\text{min}}, y_{\text{max}}]$. In other words, we don't want the gradient of $\widehat{y}'$ to be close to zero near $y_{\text{min}}$ or $y_{\text{max}}$ because it will slow down learning. Therefore, we want $\widehat{y}'_{\text{min}} \lt y_{\text{min}}$ and $\widehat{y}'_{\text{max}} \gt y_{\text{max}}$
</li>
<li>We don't want the bounds to be so large that our upper bound on the $\text{CVaR}_\alpha(X)$ statistic is uninformative. Notice that equation \eqref{st_upper} scales linearly with  $Z_{N_{\text{safety}}+1}=b$, the upper bound. Making $b$ large will make this upper bound large. 
</li>
</ol>
<p>
While these two criteria are in direct conflict, we want our upper bound to be as informative as possible, so we pick a relatively small factor, $2$, to inflate the bounds of $\widehat{y}'$ relative to $y$. We also want the $\widehat{y}'$ bounds to be equally inflated on either side of the $y$ bounds to avoid biasing the model predictions. The label bounds were $[y_{\text{min}}, y_{\text{max}}] = [-3,3]$, so the bounds on our new model predictions will be: $[\widehat{y}'_{\text{min}}$, $\widehat{y}'_{\text{max}}] = [-6,6]$. With this in hand, we can calculate the max possible squared residual, $b$, which would happen when $y=y_{\text{min}}$ and $\widehat{y}'=\widehat{y}'_{\text{max}}$ or when $y=y_{\text{max}}$ and $\widehat{y}'=\widehat{y}'_{\text{min}}$, both of which would result in the same value for the squared residual: $b=(3-(-6))^2=9^2=81$. Again, $a=0$ because it is possible that $y=\widehat{y}'$.
</p>
<p>
    We will need to define a new model class that will provide linear regression model predictions that are bounded, as in equation \eqref{squashed_model}. Note that creating a new model class may not be necessary for your custom base node. We define the new model class in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/models.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/models.py</a>, inheriting from <code class='highlight'>LinearRegressionModel</code>. We named the model class <code class='highlight'>BoundedLinearRegressionModel</code>, which uniquely defines the class and conforms to upper camel case. In this class, all we need to do is override the <code class='highlight'>predict()</code> method of the parent class. We also define a helper function <code class='highlight'>_sigmoid()</code>. Here is the full implementation of this model, which is already part of the engine source code:
</p>
{% highlight python %}
class BoundedLinearRegressionModel(LinearRegressionModel):
    def __init__(self):
        """ Implements linear regression 
        with a bounded predict function.
        Overrides several parent methods. """
        super().__init__()
        self.model_class = LinearRegression

    def _sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def predict(self,theta,X):
        """ Overrides the original predict
        function to bound predictions 

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predicted labels
        :rtype: numpy ndarray
        """
        y_min,y_max = -3,3
        s=2 # 1 gives you the same bound size as y
        # Want range of Y_hat to be s times that of Y
        # and want size of interval on either side of Y_min and Y_max
        # to be the same. The unique solution to this is:
        y_hat_min = y_min*(1+s)/2 + y_max*(1-s)/2
        y_hat_max = y_max*(1+s)/2 + y_min*(1-s)/2
        Z = np.dot(X,theta)
        return self._sigmoid(Z)*(y_hat_max-y_hat_min) + y_hat_min
{% endhighlight python %}

</div>
<p>
We now have everything we need to implement this custom base node, the $\text{CVaR}_\alpha(X)$ statistic (with $\alpha=0.1$) of the squared residual. Below, we outline the steps for creating this custom base variable, which we will call <code class='highlight'>CVaRSQE</code>. We have already implemented all of these steps in the engine library, so feel free to reference the code as you go through the steps. The steps are written generally to help you write your own custom base variables.
</p>

<ol>

<li>Fork the engine repository: <a href="https://github.com/seldonian-toolkit/Engine">https://github.com/seldonian-toolkit/Engine</a> </li>

<li> Define a class in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/nodes.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/nodes.py</a> that inherits from the <code class='highlight'>BaseNode</code> class, ideally conforming to <a href="https://en.wikipedia.org/wiki/Camel_case">upper camel case</a> and named something that uniquely identifies your custom base node. The name must not already be an existing class in that file. We chose <code class='highlight'>CVaRSQeBaseNode(BaseNode)</code>.</li>

<li> Define an <code class='highlight'>__init__()</code> method for the new class which at minimum takes the following arguments and calls <code class='highlight'>super().__init__()</code> to register these arguments to the parent class. This is also where we want to set $\alpha=0.1$. 
{% highlight python %}
def __init__(self,
    name,
    lower=float('-inf'),
    upper=float('inf'),
    **kwargs):
    super().__init__(name,lower,upper,**kwargs)
    self.alpha = 0.1
{% endhighlight python %}
</li>

<li> Define a method of the class called <code class='highlight'>calculate_bounds()</code>. This method overrides the parent class method of the same name and will instruct the engine how to calculate the upper and lower confidence bounds on the CVaR statistic. Much of the code for this method will be taken from the parent method. For example, the <code class='highlight'>kwargs</code> passed to this method are the same that are passed to the parent method, so those are unpacked in the same way as in the parent method. Also, the calls to the methods that actually calculate the upper and lower bounds (e.g., <code class='highlight'>predict_HC_upper_bound()</code>) are similar. The things we need to do in this method that are not done in the parent method we are overriding are:
    <ul>
        <li>Calculate the theoretical lower and upper bounds on the squared residual, $a$ and $b$. </li>
        <li>Get the squared residuals and sort them in order to obtain the $Z_1,\dotsc,Z_n$ used in equations (\ref{st_upper}-\ref{cs_lower}). </li>
        <li>Create a <code class='highlight'>bound_kwargs</code> dictionary to pass to the methods that do the actual calculation of the upper and lower bounds. </li>
    </ul>
Here is our full implementation of this method:
{% highlight python %}
def calculate_bounds(self,
    **kwargs):
    from seldonian.models import objectives
    """Calculate confidence bounds using the concentration 
    inequalities in Thomas & Miller 2019, Theorem's 3 and 4.
    """ 
    branch = kwargs['branch']
    model = kwargs['model']
    theta = kwargs['theta']
    data_dict = kwargs['data_dict']

    X = data_dict['features']
    y = data_dict['labels']
    # assume labels have been clipped to -3,3
    # theoretical min and max (not actual min and max) are:
    y_min,y_max = -3,3
    # Increase bounds of y_hat to s times the size of y bounds
    s=2
    y_hat_min = y_min*(1+s)/2 + y_max*(1-s)/2
    y_hat_max = y_max*(1+s)/2 + y_min*(1-s)/2

    min_squared_error = 0
    max_squared_error = max(
        pow(y_hat_max-y_min,2),
        pow(y_max - y_hat_min,2))
    
    squared_errors = objectives.vector_Squared_Error(model,theta,X,y)

    a=min_squared_error
    b=max_squared_error
    # Need to sort squared residuals to get Z1, ..., Zn
    sorted_squared_errors = sorted(squared_errors)

    bound_kwargs = {
        "Z":sorted_squared_errors,
        "delta":self.delta,
        "n_safety":kwargs['datasize'],
        "a":a,
        "b":b
        }
    
    if self.will_lower_bound and self.will_upper_bound:
        if branch == 'candidate_selection':
            lower = self.predict_HC_lowerbound(**bound_kwargs)
            upper = self.predict_HC_upperbound(**bound_kwargs)  
        elif branch == 'safety_test':
            lower = self.compute_HC_lowerbound(**bound_kwargs)  
            upper = self.compute_HC_upperbound(**bound_kwargs)  
        return {'lower':lower,'upper':upper}
    
    elif self.will_lower_bound:
        if branch == 'candidate_selection':
            lower = self.predict_HC_lowerbound(**bound_kwargs)  
        elif branch == 'safety_test':
            lower = self.compute_HC_lowerbound(**bound_kwargs)  
        return {'lower':lower}

    elif self.will_upper_bound:
        if branch == 'candidate_selection':
            upper = self.predict_HC_upperbound(**bound_kwargs)  
        elif branch == 'safety_test':
            upper = self.compute_HC_upperbound(**bound_kwargs)  
        return {'upper':upper}

    raise AssertionError("will_lower_bound and will_upper_bound cannot both be False")

{% endhighlight python %}
</li>

<li> 
    Implement equations (\ref{st_upper}-\ref{cs_lower}) as the methods: <code class='highlight'>compute_HC_upper_bound()</code>, <code class='highlight'>compute_HC_lower_bound()</code>, <code class='highlight'>predict_HC_upper_bound()</code> and <code class='highlight'>predict_HC_upper_bound()</code>, respectively. The convention we use in this library is that the <code class='highlight'>compute_*</code> methods calculate the bounds for the <code class='glossary-term'>safety test</code> and the <code class='highlight'>predict_*</code> methods calculate the bounds for <code class='glossary-term'>candidate selection</code>. Here are the implementations of these four methods:
{% highlight python %}
def predict_HC_lowerbound(self,
    Z,
    delta,
    n_safety,
    a,**kwargs):
    """
    Calculate high confidence lower bound
    that we expect to pass the safety test.
    Used in candidate selection

    :param Z: 
        Vector containing sorted squared residuals
    :type Z: numpy ndarray of length n_candidate 
    :param delta: 
        Confidence level, e.g. 0.05
    :type delta: float
    :param n_safety: 
        The number of observations in the safety dataset
    :type n_safety: int
    :param a: The minimum possible value of the squared residual
    :type a: float
    """ 
    Znew = Z.copy()
    Znew = np.array([a] + Znew)
    n_candidate = len(Znew) - 1

    sqrt_term = np.sqrt((np.log(1/delta))/(2*n_safety))
    max_term = np.maximum(
        np.zeros(n_candidate),
        np.minimum(
            np.ones(n_candidate),
            np.arange(n_candidate)/n_candidate+2*sqrt_term)-(1-self.alpha))
    
    lower = Znew[-1] - 1/self.alpha*sum(np.diff(Znew)*max_term)

    return lower

def predict_HC_upperbound(self,
    Z,
    delta,
    n_safety,
    b,**kwargs):
    """
    Calculate high confidence upper bound
    that we expect to pass the safety test.
    Used in candidate selection

    :param Z: 
        Vector containing sorted squared residuals
    :type Z: numpy ndarray of length n_candidate 
    :param delta: 
        Confidence level, e.g. 0.05
    :type delta: float
    :param n_safety: 
        The number of observations in the safety dataset
    :type n_safety: int
    :param b: The maximum possible value of the squared residual
    :type b: float
    """  
    Znew = Z.copy()
    Znew.append(b)
    
    n_candidate = len(Znew) - 1

    # sqrt term is independent of loop index
    sqrt_term = np.sqrt((np.log(1/delta))/(2*n_safety))
    max_term = np.maximum(
        np.zeros(n_candidate),
        (1+np.arange(n_candidate))/n_candidate-2*sqrt_term-(1-self.alpha))
    upper = Znew[-1] - (1/self.alpha)*sum(np.diff(Znew)*max_term)
        
    return upper

def compute_HC_lowerbound(self,
    Z,
    delta,
    n_safety,
    a,**kwargs
    ):
    """
    Calculate high confidence lower bound
    Used in safety test.

    :param Z: 
        Vector containing sorted squared residuals
    :type Z: numpy ndarray of length n_safety
    :param delta: 
        Confidence level, e.g. 0.05
    :type delta: float
    :param n_safety: 
        The number of observations in the safety dataset
    :type n_safety: int
    :param a: The minimum possible value of the squared residual
    :type a: float
    """  
    Znew = Z.copy()
    Znew = np.array([a] + Znew)
    n_candidate = len(Znew) - 1

    sqrt_term = np.sqrt((np.log(1/delta))/(2*n_safety))
    max_term = np.maximum(
        np.zeros(n_safety),
        np.minimum(
            np.ones(n_safety),
            np.arange(n_safety)/n_safety+sqrt_term)-(1-self.alpha))
    
    lower = Znew[-1] - 1/self.alpha*sum(np.diff(Znew)*max_term)

    return lower

def compute_HC_upperbound(self,
    Z,
    delta,
    n_safety,
    b,**kwargs):
    """
    Calculate high confidence upper bound
    Used in safety test

    :param Z: 
        Vector containing sorted squared residuals
    :type Z: numpy ndarray of length n_safety
    :param delta: 
        Confidence level, e.g. 0.05
    :type delta: float
    :param n_safety: 
        The number of observations in the safety dataset
    :type n_safety: int
    :param b: The maximum possible value of the squared residual
    :type b: float
    """
    Znew = Z.copy()
    Znew.append(b)
    # sqrt term is independent of loop index
    sqrt_term = np.sqrt((np.log(1/delta))/(2*n_safety))
    max_term = np.maximum(
        np.zeros(n_safety),
        (1+np.arange(n_safety))/n_safety-sqrt_term-(1-self.alpha))
    upper = Znew[-1] - 1/self.alpha*sum(np.diff(Znew)*max_term)
        
    return upper
{% endhighlight python %}
</li>

<li>The last method we need here is <code class='highlight'>calculate_value()</code>, which calculates not the bound but the actual value of the CVaR statistic. This is not necessary for the Seldonian algorithm to run, but it is used in the experiments library for making the failure rate plot. This is only implemented for continuous probability distributions. Here is our implementation of this method:
{% highlight python %}
def calculate_value(self,**kwargs):
    """
    Calculate the actual value of CVAR_alpha,
    not the bound.
    """ 
    from seldonian.models import objectives
    model = kwargs['model']
    theta = kwargs['theta']
    data_dict = kwargs['data_dict']

    # Get squashed squared residuals
    X = data_dict['features']
    y = data_dict['labels']
    squared_errors = objectives.vector_Squared_Error(model,theta,X,y)
    # sort
    Z = np.array(sorted(squared_errors))
    # Now calculate cvar
    percentile_thresh = (1-self.alpha)*100
    # calculate var_alpha 
    var_alpha = np.percentile(Z,percentile_thresh)
    # cvar is the mean of all values >= var_alpha
    cvar_mask = Z >= var_alpha
    Z_cvar = Z[cvar_mask]
    cvar = np.mean(Z_cvar)
    return cvar
{% endhighlight python %}
</li>

<li> Define a string expression for your custom base node that you will use in your constraint string. The string needs to follow these rules: 
    <ul>
    <li>Needs to be a unique identifier.</li>
    <li>Must only consist only of alphabetical characters (upper and lower case allowed) and the underscore character. No spaces are allowed. </li>
    <li>Must not already be an existing measure function name. The names of all measure functions are provided in the sub-dictionaries of the <code class='highlight'>measure_functions_dict</code> dictionary in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py</a>. For example, <code class='highlight'>"PR"</code> and <code class='highlight'>"Mean_Error"</code> are already taken.</li>
    <li>Must not already be a custom base variable name. The names of the custom base nodes we have implemented so far are listed as keys of the <code class='highlight'>custom_base_node_dict</code> in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py</a>. For example, <code class='highlight'>MEDMF</code> is already taken.</li>
    </ul>
    We chose <code class='highlight'>CVaRSQeBaseNode</code> for our string expression.
</li>
<li> Add an entry to the dictionary: <code class='highlight'>custom_base_node_dict</code> in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py</a>, where the key is the string expression of your custom base node chosen in the previous step and the value is the name of the class you defined in step 2, like this:
{% highlight python %}
'CVaRSQE':CVaRSQeBaseNode
{% endhighlight python %}
</li>
</ol>
<p>
At this point, you will be able to use your custom base node in a constraint string that you provide to the engine. Our original desired constraint was: "ensure that the CVaR of the squared residual does not exceed some threshold value." Let's choose the threshold value to be $10.0$. We can now formalize our Seldonian machine learning problem:
</p>
<p>
    Using gradient descent on a linear regression model, minimize the mean squared residual, subject to the constraint:
</p>
<ul>
    <li>
        $g_{1}: \text{CVaRSQE} \leq 10.0$, and ${\delta}_1=0.1$, where $\text{CVaRSQE}$ is the custom base variable representing the CVaR of the squared residual. 
    </li>
</ul>
<p>
  We can now run this Seldonian algorithm using the script below. Most of the script is boilerplate code for running the Seldonian Engine, but we want to note a few things first:
</p>
<ul>
    <li>The custom base variable string, <code class='highlight'>CVaRSQE</code>, appears in our constraint string. </li>
    <li>To generate the data as described in equation \eqref{data_distribution}, we are using a function called <code class='highlight'>make_synthetic_regression_dataset</code>, which is part of the Engine library but is not shown in this tutorial. The function can be found in this file: https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/utils/tutorial_utils.py </li>
    <li>The number of points, $N = N_{\text{cand}} + N_{\text{safety}}$, that we use when generating the data is important. That is because the concentration bounds in equations (\ref{st_upper}–\ref{cs_lower}) depend on $N$ (either through $N_{\text{cand}}$ or $N_{\text{safety}}$ or both). It turns out that these bounds require a relatively large amount of data to be informative. The exact amount of data needed of course depends on the other variables $\delta$, $\alpha$, and the upper bound of the squared residual, $b$. For the values of those variables that we chose, we found that we needed $N\gtrsim50,000$ points in our synthetic dataset in order to pass the <code class='glossary-term'>safety test</code> for the constraint we defined. In the end, we chose $N=75,000$ to be conservative. </li>
    <li>We are using the custom model class: <code class='highlight'>BoundedLinearRegressionModel</code> that we defined in the "Bounding the squared residuals" box above. </li>
    <li>We are only fitting the slope of the line. Because we do not provide a custom initial solution, the default model weights are the zero vector, so a slope of zero in this case.</li>
    
</ul>

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">
{% highlight python %} 
# run_engine_custom_base_node.py
import autograd.numpy as np
from seldonian.utils.tutorial_utils import make_synthetic_regression_dataset
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.spec import SupervisedSpec
from seldonian.models.models import BoundedLinearRegressionModel
from seldonian.models import objectives


def main():
    """ Test that the gpa regression example runs 
    using the custom base node that calculates 
    CVaR alpha of the squared error. Make
    sure safety test passes and solution is correct.

    Check that the actual value of the constraint (not the bound)
    is also correctly calculated.
    """
    rseed=0
    np.random.seed(rseed) 
    constraint_strs = ['CVaRSQE <= 10.0']
    deltas = [0.1]

    numPoints = 75000
    dataset = make_synthetic_regression_dataset(
        numPoints,
        loc_X=0.0,
        loc_Y=0.0,
        sigma_X=1.0,
        sigma_Y=0.2,
        clipped=True)

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas)

    model = BoundedLinearRegressionModel()

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        sub_regime='regression',
        primary_objective=objectives.Mean_Squared_Error,
        use_builtin_primary_gradient_fn=False,
        custom_primary_gradient_fn=objectives.gradient_Bounded_Squared_Error,
        parse_trees=parse_trees,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 50,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(write_cs_logfile=True,debug=False)
    if passed_safety:
        print("Passed safety test!")
        print(f"solution={solution}")
    else:
        print("Failed safety test")
        print("No solution found")

if __name__ == "__main__":
    main()
{% endhighlight python %} 
</div>

<p> 
If we save this script to a file called <code>run_engine_custom_base_node.py</code> and run it via the command line like:
{% highlight bash %}
$ python run_engine_custom_base_node.py
{% endhighlight bash %}
We will see some output like:
</p>
{% highlight bash %}
initial solution: 
[0.]
Iteration 0
Iteration 10
Iteration 20
Iteration 30
Iteration 40
Wrote /Users/ahoag/beri/code/engine-repo/examples/custom_base_node/logs/candidate_selection_log20.p with candidate selection log info
Passed safety test!
solution=[0.34152296]
{% endhighlight bash %}
<p>
The exact solution might differ slightly due to your machine's random number generator, but the <code class='glossary-term'>safety test</code> should pass. The solution found is the final weights of the model, which in our case is just the slope of the line. If we plot this line on top of data generated using the synthetic data generator, we see that it is indeed an optimal fit to the data: 
</p>

<div align="center">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/cvar_data_with_bestfit.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-5 bg-white" style="width: 40%"  alt="parse tree"> 
        <figcaption class="figure-caption"><b>Figure 4</b> - $10^4$ samples from the data distribution (blue) input to the Seldonian algorithm and the bounded linear regression model (red) found by the algorithm that minimizes the primary objective (mean squared error) while ensuring that the safety constraint: $\text{CVaRSQE} \leq 10.0$ is met. </figcaption>
    </figure> 
</div>
<p>
Finally, we can plot the contents of the <code class='glossary-term'>candidate selection</code> log file using the plotting utility module:
</p>
{% highlight python %}
from seldonian.utils.plot_utils import plot_gradient_descent
from seldonian.utils.io_utils import load_pickle

def main():
    f = "/Users/ahoag/beri/code/engine-repo/examples/custom_base_node/logs/candidate_selection_log20.p"
    solution_dict = load_pickle(f)
    fig = plot_gradient_descent(
        solution_dict,
        primary_objective_name='Mean Squared Error',
        save=True,
        savename='custom_base_node_candidate_selection.png')

if __name__ == "__main__":
    main()
{% endhighlight python %}
<p>
which will produce a plot like this:
</p>
<div align="center">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/custom_base_node_candidate_selection.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-5 bg-white" style="width: 90%"  alt="parse tree"> 
        <figcaption class="figure-caption"><b>Figure 5</b> - The evolution of the primary objective function, $f$, the single Lagrange multiplier, $\lambda_1$, the single constraint function, $g_1$, and the Lagrangian, $\mathcal{L}$, during each step of gradient descent. The red region of the $g_1$ plot indicates where the <code class='glossary-term'>safety test</code> is predicted to be violated, i.e., the infeasible set. </figcaption>
    </figure> 
</div>
<p>
    The plot shows that despite the initial solution starting out in the infeasible set, a feasible solution was found in relatively few iterations. The default hyperparameters of gradient descent worked well in this case, but in other cases it may be necessary to adjust them. 
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3>Summary</h3>
<p>
    In this tutorial, we demonstrated two methods for customizing your behavioral constraints. The first method was to create a new measure function. We demonstrated how to implement this using precision as an example measure function that is not already implemented in the library. When one uses a measure function in their constraint, they are assuming that the high-confidence bounds are calculated on the mean of that function over the provided data. We showed a second method, creating a custom base variable, where that assumption does not have to hold. To illustrate this alternative method, we implemented a constraint that uses the conditional value at risk (CVaR) of the squared residual in a regression problem. We showed how to create a custom base variable to support this type of custom constraint. Finally, we used our new custom base variable in a Seldonian algorithm.  
</p>

</div> -->