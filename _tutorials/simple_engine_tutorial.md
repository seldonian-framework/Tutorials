---
layout: tutorial
permalink: /tutorials/simple_engine_tutorial/
prev_url: /tutorials/install_toolkit_tutorial/
prev_page_name: (B) Install Engine tutorial
next_url: /tutorials/fair_loans_tutorial/
next_page_name: (D) Fair loans tutorial
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Tutorial C: Running the Seldonian Engine</h2>
    <hr class="my-4">
    <h3>Outline</h3>
    <p>In this tutorial, you will learn how to:
    <ul>
        <li>Use the Engine to set up a (quasi)-Seldonian machine learning algorithm (QSA).</li>
        <li>Run the algorithm using the Engine and understand its output.</li>
    </ul>
    Note that due to the choice of confidence bound method used in this tutorial (Student's $t$-test), the algorithms in this tutorial are technically quasi-Seldonian algorithms (QSAs). See <a href="{{ "/overview/#algorithm" | relative_url}}">the overview</a> for more details.
    </p>
    <h3> An example Seldonian machine learning problem </h3>
    <p>
        Consider a simple supervised regression problem with two continous random variables X and Y. Let the goal be to predict the label Y using the single feature X. To solve this problem we can use linear regression with the <i>mean squared error</i> (MSE) as the objective function. Recall that the mean squared error of predictions $\hat Y$ is the expected squared difference between the actual value of $Y$ and the prediction $\hat Y$, i.e., $\mathbf{E}[(Y-\hat Y)^2]$. We can approximate an optimal solution by minimizing the objective function with respect to the weights of the model, ${\theta}$, which in this case are just the intercept and slope of the line.
    </p>
    <p>
        Now let's suppose we want to add the following two constraints into the problem:
    </p>
    <ol>
        <li>Ensure that the MSE is less than or equal to $2.0$ with a probability of at least $0.9$. </li>  
        <li>Ensure that the MSE is <i>greater than or equal to</i> $1.25$ with a probability of at least $0.9$.</li>
    </ol> 
    <p>
        Notice that this second constraint conflicts with the primary objective of minimizing the MSE. Though this particular constraint is contrived, it models the common setting of interest wherein safety and fairness constraints conflict with the primary objective.
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
        First, notice that the values of ${\delta}_1$ and ${\delta}_2$ are both $0.1$. This is because constraints are enforced with a probability of at least $1-{\delta}$, and we stated that the constraints should be enforced with a probability of at least $0.9$. The Seldonian algorithm will attempt to satisfy both of these constraints simultaneously, while also minimizing the primary objective. If it cannot find a solution that satisfies the constraints, it will return "NSF", i.e., "No Solution Found". 
    </p>
    <p>
        Next, notice that here the MSE is <i>not</i> just the average squared error on the available training data. These constraints are much stronger: they are constraints on the MSE when the learned model is applied to <i>new data</i>. This is important because we don't just want machine learning models that appear to be safe or fair on the training data. We want machine learning models that are safe or fair when used to made decisions or predictions in the future.
    </p>
    <p>
        To code up this example using the engine, we need to follow these steps.
    </p>
    <ol>
        <li> Define the data - we will generate some synthetic data for X and Y in this case.</li>
        <li> Create parse trees from the behavioral constraints.</li>
        <li> Define the underlying machine learning model. </li>
        <li> Create a spec object containing all of this information and some hyperparameters - we can ignore many of these in this example. For a full list of parameters and their default values see the API docs for <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.SupervisedSpec.html#seldonian.spec.SupervisedSpec">SupervisedSpec</a>.</li>
        <li> Run the Seldonian algorithm using the spec object. </li>
    </ol>
    <p>
    Let's write out the code to do this. Each step above is enumerated in comments in the code below. We will make heavy use of helper functions with many hidden defaults. In the tutorials that follow, we will explore how to customize running the Engine.
    </p>

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet"> 

{% highlight python %}
# example.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from seldonian.models.models import LinearRegressionModel
from seldonian.spec import SupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

if __name__ == "__main__":
    np.random.seed(0)
    num_points=1000  
    # 1. Define the data - X ~ N(0,1), Y ~ X + N(0,1)
    dataset = make_synthetic_regression_dataset(
        num_points=num_points)

    # 2. Create parse trees from the behavioral constraints 
    # constraint strings:
    constraint_strs = ['Mean_Squared_Error >= 1.25','Mean_Squared_Error <= 2.0']
    # confidence levels: 
    deltas = [0.1,0.1] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas)

    # 3. Define the underlying machine learning model
    model = LinearRegressionModel()

    """4. Create a spec object, using some
    hidden defaults we won't worry about here
    """
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='regression',
    )

    # 5. Run seldonian algorithm using the spec object
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run()
    print(passed_safety,solution)
{% endhighlight %}
</div>
<p>
If you copy the above code into a file called "example.py", run the example from the command line (once inside a virtual environment where you have pip-installed the engine) by doing:
</p>

<p>
{% highlight bash %}
$ python example.py
{% endhighlight %}
</p>
<p>
    You should see some output like:
{% highlight python %}
Iteration 0
Iteration 10
Iteration 20
Iteration 30
Iteration 40
...
Passed safety test
True [0.16911355 0.1738146 ]
{% endhighlight %}
    </p>
    <p>
    Notice in the last few lines of the script that <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.seldonian_algorithm.SeldonianAlgorithm.html#seldonian.seldonian_algorithm.SeldonianAlgorithm.run">SA.run()</a> returns two values. <code class="highlight">passed_safety</code> is a Boolean indicating whether the candidate solution found during candidate selection passed the safety test. If <code class="highlight">passed_safety==False </code>, then <code class='highlight'> solution="NSF"</code>, i.e., "No Solution Found". If <code class="highlight">passed_safety==True</code> then the solution is the array of model weights that cause the safety test to be passed. In this example, you should get <code class="highlight">passed_safety=True</code> and a candidate solution of something like: <code class="highlight">[0.16911355 0.1738146]</code>, although the exact numbers might differ slightly depending on your machine's random number generator.
</p>
<p> Also notice that <code class="highlight">SA.run()</code> does not return what the value of the primary objective actually was on the safety test. Given that it passed the safety test, we know that it should be between $1.25$ and $2.0$ (and the actual MSE on future data will be in this range with high probability). The <code class="highlight">SA</code> object provides the introspection we need to extract this information through the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.seldonian_algorithm.SeldonianAlgorithm.html#seldonian.seldonian_algorithm.SeldonianAlgorithm.evaluate_primary_objective">SA.evaluate_primary_objective()</a> method:

{% highlight python %}
# Check the value of the primary objective on the safety dataset
st_primary_objective = SA.evaluate_primary_objective(theta=solution,
branch='safety_test')
print(st_primary_objective)
{% endhighlight %}

This should print a value around: $1.61$, which satsifies the behavioral constraints. 
</p>

<p>
We might also wonder what the values of the primary objective and the behavioral constraint function were during the candidate selection process. All of this information and more is stored in a dictionary that is retrievable via the <a href="">SA.get_cs_result()</a> method:
{% highlight python %}
cs_dict = SA.get_cs_result() # returns a dictionary with a lot of quantities evaluated at each step of gradient descent
print(cs_dict.keys())
{% endhighlight %}
This will print all of the keys of this dictionary:
{% highlight python %}
['candidate_solution', 'best_index', 'best_feasible_g', 'best_feasible_f', 'solution_found', 'theta_vals', 'f_vals', 'g_vals', 'lamb_vals', 'L_vals']
{% endhighlight %}
So, to get the primary objective values we would do:
{% highlight python %}
print(cs_dict['f_vals'])
{% endhighlight %}
and to get the values of the constraint functions, $g_1$ and $g_2$, we would do:
{% highlight python %}
print(cs_dict['g_vals'])
{% endhighlight %}
Even if candidate selection returns "NSF", the <code class="highlight">cs_dict</code> stores these values.
</p>
    <h3>Summary</h3>
    <p>In this tutorial, we demonstrated how to:</p>
    <ul>
        <li>Use the Engine to set up a Seldonian machine learning algorithm.</li>
        <li>Run the algorithm using the Engine and understand its outputs.</li>
    </ul>
<p>
</p>

</div>