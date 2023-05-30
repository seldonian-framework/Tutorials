---
layout: tutorial
permalink: /tutorials/simple_engine_tutorial/
prev_url: /tutorials/install_toolkit_tutorial/
prev_page_name: (B) Installing the Seldonian Toolkit
next_url: /tutorials/fair_loans_tutorial/
next_page_name: (D) Fairness for automated loan approval systems
title: Seldonian \| Tutorial C
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Tutorial C: Running the Seldonian Engine</h2>
    <a href="https://colab.research.google.com/gist/austinhoag/b417a7d6bfd3ffc4b99c0ab0b36c11c6/tutorial-c-running-the-seldonian-engine.ipynb" target="blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <hr class="my-4">


    <!-- Table of contents -->
    <h3> Contents </h3>
    <ul>
        <li> <a href="#intro">Introduction</a> </li>
        <li> <a href="#outline">Outline</a> </li>
        <li> <a href="#example">An example Seldonian machine learning problem</a></li>
        <li> <a href="#running_the_engine">Running the Seldonian Engine</a> </li>
        <li> <a href="#extracting">Extracting important quantities</a> </li>
        <li> <a href="#summary">Summary</a> </li>
    </ul>
    <hr class="my-4">

    <h3 id="intro">Introduction</h3>
    <p>The Seldonian Engine library is the core library of the Seldonian Toolkit. The engine implements a Seldonian algorithm that can be used to train ML models (supervised learning and offline reinforcement learning) subject to high-confidence fairness and/or safety constraints. In this tutorial, we demonstrate how to use the engine to learn a simple linear model which satisfies two safety constraints simultaneously. 
    </p>

    <p>
        There is another library in the Seldonian Toolkit called the Experiments library, which is used to evaluate the performance, data-efficiency, and safety of  Seldonian algorithms in more detail than can be done with the Engine alone. The Experiments library is is not covered in this tutorial, but see the <a href="{{ "/tutorials/fair_loans_tutorial/" | relative_url}}">next tutorial</a> for how to use it. 
    </p>

    <h3 id="outline">Outline</h3>
    <p>In this tutorial, you will learn how to:
    <ul>
        <li>Set up a Seldonian algorithm using the engine. </li>
        <li>Provide safety constraints.</li>
        <li>Run the Seldonian algorithm to obtain a model that satisfies the safety constraints. </li>
        <li>Understand the outputs of the engine.</li>
    </ul>
    Note that due to the choice of confidence-bound method used in this tutorial (Student's $t$-test), the algorithms in this tutorial are technically quasi-Seldonian algorithms (QSAs). See <a href="{{ "/overview/#algorithm" | relative_url}}">the overview</a> for more details.
    </p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h3 id="example"> An example Seldonian machine learning problem </h3>
    <p>
        Consider a simple regression problem with two continuous random variables X and Y. Let the goal be to predict the label Y using the single feature X. One approach to this problem is to use gradient descent on a linear regression model with the <i>mean squared error</i> (MSE) as the objective function. Recall that the mean squared error of predictions $\hat Y$ is the expected squared difference between the actual value of $Y$ and the prediction $\hat Y$, i.e., $\mathbf{E}[(Y-\hat Y)^2]$. We can approximate an optimal solution by minimizing the objective function with respect to the weights of the model, ${\theta}$, which in this case are just the intercept and slope of the line.
    </p>
    <p>
        Now, let's suppose we want to add the following two constraints into the problem:
    </p>
    <ol>
        <li>Ensure that the MSE is less than or equal to $2.0$ with a probability of at least $0.9$. </li>  
        <li>Ensure that the MSE is <i>greater than or equal to</i> $1.25$ with a probability of at least $0.9$.</li>
    </ol> 
    <p>
        Notice that this second constraint conflicts with the primary objective of minimizing the MSE. Though this particular constraint is contrived, it models the common setting of interest wherein safety and/or fairness constraints conflict with the primary objective.
    </p>
    <p>
        This problem can now be fully formulated as a Seldonian machine learning problem:
    </p>
    <p>
        Minimize the MSE, subject to the constraints:
    </p>
    <ul>
        <li>
            $g_{1}: \mathrm{Mean\_Squared\_Error} \leq 2.0$, and ${\delta}_1=0.1$.  
        </li>
        <li>
            $g_{2}: \mathrm{Mean\_Squared\_Error} \geq 1.25$, and ${\delta}_2=0.1$.
        </li>
    </ul>
    <p>
        First, notice that the values of ${\delta}_1$ and ${\delta}_2$ are both $0.1$. This is because in the <a href="{{ "/tutorials/alg_details_tutorial/#overview" | relative_url}}">Seldonian algorithm framework</a> constraints are enforced with a probability of at least $1-{\delta}$, and we stated that the constraints should be enforced with a probability of at least $0.9$. The Seldonian algorithm will attempt to satisfy both of these constraints simultaneously, while also minimizing the primary objective, the MSE. If it cannot find a solution that satisfies the constraints at the confidence levels provided, it will return "NSF", i.e., "No Solution Found". 
    </p>
    <p>
        Next, notice that here the MSE is <i>not</i> just the average squared error on the available training data. These constraints are much stronger: they are constraints on the MSE when the learned model is applied to <i>new data</i>. This is important because we don't just want machine learning models that appear to be safe or fair on the training data. We want machine learning models that are safe or fair when used to made decisions or predictions in the future.
    </p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h3 id="running_the_engine">Running the Seldonian Engine</h3>
    <p>
        To code this example using the engine, we need to follow these steps.
    </p>
    <ol>
        <li> Define the data — we will generate some synthetic data for X and Y for this case.</li>
        <li> Create parse trees from the behavioral constraints.</li>
        <li> Define the underlying machine learning model. </li>
        <li> Create a spec object containing all of this information and some hyperparameters — we can ignore many of these in this example. For a full list of parameters and their default values, see the API docs for <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.SupervisedSpec.html#seldonian.spec.SupervisedSpec">SupervisedSpec</a>.</li>
        <li> Run the Seldonian algorithm using the spec object. </li>
    </ol>
    <p>
    Let's write out the code to do this. Each step above is enumerated in comments in the code below. We will make heavy use of helper functions with many hidden defaults. In the tutorials that follow, we will explore how to customize running the engine.
    </p>

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet"> 

{% highlight python %}
# example.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from seldonian.spec import createSimpleSupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

if __name__ == "__main__":
    np.random.seed(0)
    num_points=1000  
    """ 1. Define the data - X ~ N(0,1), Y ~ X + N(0,1) """
    dataset = make_synthetic_regression_dataset(
        num_points=num_points)

    """ 2. Specify safety constraints """
    constraint_strs = ['Mean_Squared_Error >= 1.25',
        'Mean_Squared_Error <= 2.0']
    deltas = [0.1,0.1] # confidence levels


    """3. Create a spec object, using some
    hidden defaults we won't worry about here
    """
    spec = createSimpleSupervisedSpec(
        dataset=dataset,
        constraint_strs=constraint_strs,
        deltas=deltas,
        sub_regime='regression',
    )

    """ 4. Run seldonian algorithm using the spec object """
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run()
    print(passed_safety,solution)
{% endhighlight %}
</div>
<p>
If you copy the above code into a file called "example.py", run the example from the command line (once inside a virtual environment where you have installed the Engine library) by doing:
</p>

<p>
{% highlight bash %}
$ python example.py
{% endhighlight %}
</p>
<p>
    You should see some output like this:
{% highlight python %}
Have 200 epochs and 1 batches of size 400

Epoch: 0, batch iteration 0
Epoch: 1, batch iteration 0
Epoch: 2, batch iteration 0
Epoch: 3, batch iteration 0
Epoch: 4, batch iteration 0
...
True [0.16911355 0.1738146 ]
{% endhighlight %}
    </p>
    
    <p>
    The output shows some of the default values that were hidden in the script. For example, we are running gradient descent in "batch" mode, i.e., putting all of our candidate data (400 data points) in at once and running for 200 epochs. These settings can be changed, but we won't cover that in this tutorial.
    </p>
    
    <p>
    Notice in the last few lines of the script that <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.seldonian_algorithm.SeldonianAlgorithm.html#seldonian.seldonian_algorithm.SeldonianAlgorithm.run">SA.run()</a> returns two values. <code class='codesnippet'>passed_safety</code> is a Boolean indicating whether the candidate solution found during candidate selection passed the safety test. If <code class='codesnippet'>passed_safety==False </code>, then <code class='codesnippet'> solution="NSF"</code>, i.e., "No Solution Found". If <code class='codesnippet'>passed_safety==True</code>, then the solution is the array of model weights that resulted in the safety test passing. In this example, you should get <code class='codesnippet'>passed_safety=True</code> and a candidate solution of something like: <code class='codesnippet'>[0.16911355 0.1738146]</code>, although the exact numbers might differ slightly depending on your machine's random number generator. These numbers represent the y-intercept and slope of the line that the Seldonian algorithm found. 
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h3 id="extracting">Extracting important quantities</h3>
<p> 
    There are a few quantities of interest that are not automatically returned by <code class='codesnippet'>SA.run()</code>. One such quantity is the value of the primary objective function (the MSE, in this case) evaluated on the safety data for the model weights returned by the algorithm, $\hat{f}(\theta_{\text{cand}},D_{\text{safety}})$. Given that the solution passed the safety test, we know that $\hat{f}(\theta,D_{\text{safety}})$ will likely be between $1.25$ and $2.0$ (and the actual MSE on future data will be in this range with high probability). The <code class='codesnippet'>SA</code> object provides the introspection we need to extract this information through the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.seldonian_algorithm.SeldonianAlgorithm.html#seldonian.seldonian_algorithm.SeldonianAlgorithm.evaluate_primary_objective">SA.evaluate_primary_objective()</a> method:

{% highlight python %}
st_primary_objective = SA.evaluate_primary_objective(
    theta=solution,
    branch='safety_test')
print(st_primary_objective)
{% endhighlight %}

This should print a value around $1.61$, which satisfies the behavioral constraints. We can use the same method to check the value of the primary objective function evaluated on the candidate data at this solution:
</p>
{% highlight python %}
cs_primary_objective = SA.evaluate_primary_objective(
    theta=solution,
    branch='candidate_selection')
print(cs_primary_objective)
{% endhighlight %}
<p> 
    This should print a value of around $1.56$. While we know in this case that the safety test passed, i.e., the high-confidence upper bounds on the constraints were less than or equal to zero, we might be interested in what the actual values of those upper bounds were during the safety test. We can use the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.seldonian_algorithm.SeldonianAlgorithm.html#seldonian.seldonian_algorithm.SeldonianAlgorithm.get_st_upper_bounds">SA.get_st_upper_bounds()</a> method for this.
</p>
{% highlight python %}
>> print(SA.get_st_upper_bounds())
{'1.25-(Mean_Squared_Error)': -0.2448558988476761, 'Mean_Squared_Error-(2.0)': -0.2710930638194431}
{% endhighlight python %}

<p>
This returns a dictionary where the keys are the constraint strings and the values are the upper bounds. The values you see should be close to the values above, but may differ slightly. Here are some things to note about this dictionary:
<ul>
    <li>Both upper bounds are less than or equal to zero, as expected. </li>
    <li>The keys of this dictionary show the constraint strings in a slightly different form than how we originally defined them. They are written in the form: $g_i \leq 0$, where $g_i$ here represents the $i$th constraint function. For example, $1.25-(\text{Mean_Squared_Error})\leq0$ is mathematically equivalent to $\text{Mean_Squared_Error} \geq 1.25$, the form we used to specify our second constraint at the beginning of the tutorial. This rearrangement is done for consistency in interpreting the upper bounds.</li>
    <li>Because this information is returned in a dictionary, the order of the constraints is not guaranteed to be the same as the order in which we specified our constraints originally.</li>
</ul>
</p>

<p>
More introspection to the <code class='codesnippet'>SA</code> object is possible, but it is beyond the scope of this tutorial. 
</p>
</div> 
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h3 id="summary">Summary</h3>
    <p>In this tutorial, we demonstrated how to:</p>
    <ul>
        <li>Use the engine to set up a Seldonian machine learning algorithm.</li>
        <li>Run the algorithm using the engine.</li>
        <li> Extract and understand important quantities generated by the algorithm.</li>
    </ul>
<p>
</p>

</div>