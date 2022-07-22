---
layout: tutorial
permalink: /tutorials/simple_engine_tutorial/
prev_url: /tutorials/install_engine_tutorial/
prev_page_name: Install Engine tutorial
next_url: /tutorials/fair_loans_tutorial/
next_page_name: Fair loans tutorial
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h1 class="mb-3">Tutorial: Getting started with the Seldonian Engine</h1>
    <hr class="my-4">
    <h3>Outline</h3>
    <p>In this tutorial, you will learn how to:</p>
    <ul>
        <li>Use the Engine to set up a Seldonian machine learning algorithm </li>
        <li>Run the algorithm using the Engine and understand its outputs</li>
    </ul>
    <h3> An example Seldonian machine learning problem </h3>
    <p>
        Consider a simple supervised regression problem with two continous random variables X and Y. Let the goal be to predict label Y using the single feature X. To solve this problem we can use linear regression with an objective function of the mean squared error (MSE). We can find the optimal solution by minimizing the objective function with respect to the weights of the model, ${\theta}$, which in this case are just the intercept and slope of the line.
    </p>
    <p>
        Now let's suppose we want to add the following two constraints into the problem:
    </p>
    <ol>
        <li>Ensure that the MSE is less than $2.0$ with a probability of at least $0.9$. </li>  
        <li>Ensure that the MSE is <i>greater than</i> $1.25$ with a probability of at least $0.9$.</li>
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
            $g_{1} = \mathrm{Mean\_Squared\_Error} - 2.0$, and ${\delta}_1=0.1$.  
        </li>
        <li>
            $g_{2} = 1.25 - \mathrm{Mean\_Squared\_Error}$, and ${\delta}_2=0.1$.
        </li>
    </ul>
    <p>
        First, notice that the values of ${\delta}_1$ and ${\delta}_2$ are both $0.1$. This is because constraints are enforced with a probability of at least $1-{\delta}$, and we stated that the constraints should be enforced with a probability of at least $0.9$. The Seldonian algorithm will attempt to satisfy both of these constraints simultaneously, while also minimizing the primary objective. If it cannot find a solution that satisfies the constraints, it will return "NSF", i.e. "No solution found". 
    </p>
    <p>
        To code up this example using the engine, we need to follow these steps:
    </p>
    <ol>
        <li> Define the data - we will generate some synthetic data for X and Y in this case.</li>
        <li> Define the metadata - in this case this consists of the column names for X and Y and the regime, which is "supervised".</li>
        <li> Put the data and metadata together into a DataSet object.</li>
        <li> Define the behavioral constraints (constraint strings and confidence levels), which we already did above.</li>
        <li> Make the parse trees from these behavioral constraints.</li>
        <li> Define the underlying machine learning model and primary objective. </li>
        <li> Define an initial solution function which takes the features and labels as inputs and outputs an initial weight vector to start candidate selection. In this case we will define a function <code class='highlight'> initial_solution() </code> function which just returns a zero vector as the initial solution.</li>
        <li> Decide what fraction of the data to split into candidate selection vs. the safety test.</li>
        <li> Decide what method to use for computing the high confidence upper bound on each $g_{i}$. We will use the Student $t$-statistic.</li>
        <li> Create a spec object containing all of this information and some hyperparameters - we can ignore many of these in this example. For a full list of parameters and their defaults see the API docs for <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.SupervisedSpec.html#seldonian.spec.SupervisedSpec">SupervisedSpec</a>.</li>
        <li> Run the Seldonian algorithm using the spec object. </li>
    </ol>
    Let's write out the code to do this. Each step above is enumerated in comments in the code below:
        

{% highlight python %}
# example.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
from seldonian.models.model import LinearRegressionModel
from seldonian.dataset import SupervisedDataSet
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.spec import SupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm

if __name__ == "__main__":
    np.random.seed(0)
    numPoints=1000

    # 1. Define the data
    def generate_data(numPoints,loc_X=0.0,
        loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0):
        """ The function we will use to generate 
        synthetic data
        """
        # Sample x from a standard normal distribution
        X = np.random.normal(loc_X, sigma_X, numPoints) 
        # Set y to be x, plus noise from a standard normal distribution
        Y = X + np.random.normal(loc_Y, sigma_Y, numPoints) 
        return (X,Y)
    X,Y = generate_data(numPoints)

    # 2. Define the metadataa
    columns = columns=['feature1','label']

    # 3. Make a dataset object
    rows = np.hstack([np.expand_dims(X,axis=1),
        np.expand_dims(Y,axis=1)])
    df = pd.DataFrame(rows,columns=columns)

    dataset = SupervisedDataSet(df,
        meta_information=columns,
        label_column='label',
        include_intercept_term=True)

    """ include_intercept_term=True
    adds a column of ones in the 
    feature array for convenience 
    during matrix multiplication.
    """

    # 4. Define the behavioral constraints
    constraint_strs = ['1.25 - Mean_Squared_Error','Mean_Squared_Error - 2.0']
    deltas = [0.1,0.1] # confidence levels

    # 5. Make the parse trees from these behavioral constraints 

    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]

        # Create parse tree object
        parse_tree = ParseTree(
            delta=delta,
            regime='supervised',
            sub_regime='regression',
            columns=columns)

        # Fill out tree
        parse_tree.create_from_ast(constraint_str)
        # assign deltas for each base node
        # use equal weighting for each unique base node
        parse_tree.assign_deltas(weight_method='equal')

        # Assign bounds needed on the base nodes
        parse_tree.assign_bounds_needed()
        
        parse_trees.append(parse_tree)

    # 6. Define the underlying machine learning model and primary objective 
    model_class = LinearRegressionModel

    primary_objective = model_class().sample_Mean_Squared_Error

    # 7. Define initial solution function
    def initial_solution(X,y):
        """ Initial solution will be [0,0] """
        return np.zeros(2)

    initial_solution_fn=initial_solution

    # 8. Decide what fraction of your data to split into
    # candidate selection vs. the safety test.
    frac_data_in_safety=0.6

    """ 9. Decide what method to use for computing the
    high confidence upper bound on each $g_{i}$.""" 
    bound_method='ttest'

    """10. Create a spec object, using some 
    hidden defaults we won't worry about here"""
    spec = SupervisedSpec(
        dataset=dataset,
        model_class=model_class,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        initial_solution_fn=initial_solution_fn,
        parse_trees=parse_trees,
        bound_method=bound_method,
    )

    # 11. Run seldonian algorithm using the spec object
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run()
    print(passed_safety,solution)
{% endhighlight %}
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
    Notice in the last few lines of the script that <code class="highlight"> SA.run() </code> returns two values. <code class="highlight">passed_safety</code> is a boolean indicating whether the candidate solution found during candidate selection passed the safety test. If <code class="highlight">passed_safety==False </code>, then <code class='highlight'> solution="NSF" </code>, i.e. "No Solution Found". If <code class="highlight"> passed_safety==True </code> then the solution is the array of model weights that cause the safety test to be passed. In this example, you should get <code class="highlight"> passed_safety=True </code> and a candidate solution of something like: <code class="highlight"> [0.16911355 0.1738146] </code>, although the exact numbers might differ slightly depending on your machine's random number generator.
</p>
<p> Also notice that <code class="highlight">SA.run()</code> does not return what the value of the primary objective actually was on the safety test. Given that it passed the safety test, we know that it must satisfy: $1.25 \leq {\theta} \leq 2.0$ (with high probability). The <code class="highlight">SA</code> object provides the introspection we need to extract this information:

{% highlight python %}
# Check the value of the primary objective on the safety dataset
st_primary_objective = SA.evaluate_primary_objective(theta=solution,
branch='safety_test')
print(st_primary_objective)
{% endhighlight %}

This should print a value around: $1.61$, which satsifies the behavioral constraints. 
</p>
    <h3>Summary</h3>
    <p>In this tutorial, we demonstrated how to:</p>
    <ul>
        <li>Use the Engine to set up a Seldonian machine learning algorithm </li>
        <li>Run the algorithm using the Engine and understand its outputs</li>
    </ul>
<p>
</p>

</div>