---
layout: tutorial
permalink: /tutorials/fair_loans_tutorial/
prev_url: /tutorials/simple_engine_tutorial/
prev_page_name: Simple Engine tutorial
next_url: /tutorials/science_GPA_tutorial/
next_page_name: Science paper GPA tutorial
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    
<h2 align="center" class="mb-3">Tutorial: Fairness for Automated Loan Approval Systems</h2>

<hr class="my-4">

<h3>Introduction</h3>

<p>This tutorial is intended to provide an end-to-end use case of the Seldonian Toolkit. We will be using the <a href="https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)">UCI Statlog (German Credit Data) Data Set</a>, which contains 20 attributes for a set of 1000 people and a binary-valued label column describing whether they are a high (value=1) or low credit risk (value=0). If someone is a high credit risk, a bank is less likely to provide them with a loan. Our goal in this tutorial will be to use the Seldonian Toolkit to create a model that makes predictions about credit risks that are fair with respect to gender (for this tutorial we consider the simplified binary gender setting). We will use several definitions of fairness, and we stress that these definitions may not be the correct ones to use in reality. They are simply examples to help you understand how to use this toolkit. 
</p>

<h3>Outline</h3>

<p>In this tutorial, you will learn how to:

<ul>
    <li>Format a supervised learning (classification) dataset so that it can be used in the Seldonian Toolkit.</li>
    <li>Build a Seldonian machine learning model that implements common fairness constraints.</li>
    <li>Run a Seldonian experiment, assessing the performance and safety of the Seldonian ML model relative to baseline models and other Fairness-aware ML models. </li>
</ul>
</p>
<h3 id="dataset_prep"> Dataset preparation </h3>

<p>
    We created a <a href="https://github.com/seldonian-toolkit/Engine/blob/main/examples/german_credit/loan_dataset_preprocessing.ipynb">Jupyter notebook</a> implementing the steps described in this section. If you would like to skip this section, you can find the correctly re-formatted dataset and metadata file that are the end product of the notebook here: <a href="https://github.com/seldonian-toolkit/Engine/tree/main/static/datasets/supervised/german_credit">https://github.com/seldonian-toolkit/Engine/tree/main/static/datasets/supervised/german_credit</a>. 
</p>

<p>
    UCI provides two versions of the dataset: "german.data" and "german.data-numeric". They also provide a file "german.doc" describing the "german.data" file only. We ignored the "german.data-numeric" file because there was no documentation for it. We downloaded the file "german.data" from here: <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/">https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/</a>. We converted it to a CSV file by replacing the space characters with commas. Attribute 9 according to "german.doc" is the personal status/sex of each person. This is a categorical column with 5 possible values, 2 of which describe females and 3 of which describe males. We created a new column that has a value of "F" if female (A92 or A95) and "M" if male (any other value) and dropped the personal status column. We decided to ignore the marriage status of the person for the purpose of this tutorial. 
</p>

<p> 
    Next, we one-hot encoded all thirteen categorical features, including the sex feature that we created in the previous step. We applied a standard scaler to the remaining numerical 7 features. The one-hot encoding step created an additional 39 columns, resulting in 59 total features. The final column in the dataset is the label, which we will refer to as "credit_rating" hereafter. We mapped the values of this column as such: (1,2) -> (0,1) so that they would behave well in our binary classification models. We combined the 59 features and the single label column into a single pandas dataframe and saved the file as a CSV file, which can be found <a href="https://github.com/seldonian-toolkit/Engine/blob/main/static/datasets/supervised/german_credit/german_loan_numeric_forseldonian.csv">here</a>.
</p>

<p> 
    We also prepared a JSON file containing the metadata that we will need to provide to the Seldonian Engine library <a href="https://github.com/seldonian-toolkit/Engine/blob/main/static/datasets/supervised/german_credit/metadata_german_loan.json">here</a>. The column names beginning with "c__" were the columns created by scikit-learn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html">OneHotEncoder</a>. The columns "M" and "F" are somewhat buried in the middle of the columns list, and correspond to the male and female one-hot encoded columns. The "sensitive_columns" key in the JSON file points to those columns. The "label_column" key in the JSON file points to the "credit_rating" column. 
</p>

<h3>Formulate the Seldonian ML problem</h3>

<p>
    As in the <a href="{{ page.prev_url | relative_url }}">previous tutorial</a>, we first need to define the standard machine learning problem in the absence of constraints. The decision of whether to deem someone as being a high or low credit risk is a binary classification problem, where the label "credit_rating" is 0 if the person is a low credit risk and 1 if the person is a high credit risk. We could use logistic regression and minimize an objective function, for example the logistic loss, via gradient descent to solve this standard machine learning problem.  
</p>

<p>
    Now let's suppose we want to add fairness constraints to this problem. The first fairness constraint that we will consider is called <i>disparate impact</i>, which ensures that the ratio of positive class predictions (in our case the prediction that someone is a high credit risk) between sensitive groups may not differ by more than some threshold. In the <a href="{{ page.prev_url | relative_url }}">previous tutorial</a>, we demonstrated how to write fairness constraints for a regression problem using the special measure function "Mean_Squared_Error" in the constraint string. For disparate impact, the measure function we will use is "PR", which stands for "positive rate", which is the fraction of predictions that predict 1, the positive class. Disparate impact between our two sensitive attribute columns "M" and "F" with a threshold value of 0.9 can be written as: $0.9 - \text{min}( (\text{PR} | [\text{M}]) / (\text{PR} | [\text{F}]), (\text{PR} | [\text{F}]) / (\text{PR} | [\text{M}]) )$.
Let us enforce this constraint function with a confidence of $0.95$. 
</p>

<p>
    The problem can now be fully formulated as a Seldonian machine learning problem:
</p>

<p>
    Using gradient descent on a logistic regression model, minimize the logistic loss, subject to the constraint:
<ul>
    <li>
        $g_{1} = 0.9 - \mathrm{min}( (\text{PR} | [\text{M}])/(\text{PR} | [\text{F}]),(\text{PR} | [\text{F}]) / (\text{PR} | [\text{M}]) )$, and ${\delta}_1=0.05$.  
    </li>
</ul>
</p>

<h3>Creating the specification object</h3>

<p>
    To be able to run the Seldonian algorithm using the Seldonian Toolkit libraries, we will need to create a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.SupervisedSpec.html#seldonian.spec.SupervisedSpec">SupervisedSpec</a> object. We will demonstrate two different ways to create this object for our example problem. 
</p>

<h5> Creating the specification object from a script </h5>
<p>
A complete script for creating the spec object for our Seldonian ML problem is shown below. This script will save the spec object as a pickle file called "spec.pkl" in the <code class='highlight'>save_dir</code> directory on your computer. That directory is currently set as the directory where you run this script, so change <code class='highlight'>save_dir</code> in the code snippet below to another directory if you want to save it elsewhere. Also, make sure to modify <code class='highlight'>data_pth</code> and <code class='highlight'>metadata_pth</code> to point to the locations where you downloaded the data and metadata files described in the <a href="#dataset_prep"> Dataset preparation section</a>, respectively. 
</p>

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">

{% highlight python %}
# createSpec.py
import os
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import load_json,save_pickle
from seldonian.spec import SupervisedSpec
from seldonian.models.models import LogisticRegressionModel

if __name__ == '__main__':
    data_pth = "../../static/datasets/supervised/german_credit/german_loan_numeric_forseldonian.csv"
    metadata_pth = "../../static/datasets/supervised/german_credit/metadata_german_loan.json"
    save_dir = '.'
    # Load metadata
    metadata_dict = load_json(metadata_pth)

    regime = metadata_dict['regime']
    columns = metadata_dict['columns']
    sensitive_columns = metadata_dict['sensitive_columns']
    sub_regime = metadata_dict['sub_regime']
    label_column = metadata_dict['label_column']
    
    # Use logistic regression model
    model_class = LogisticRegressionModel
    
    # Set the primary objective to be log loss
    primary_objective = model_class().sample_logistic_loss

    # Load dataset from file
    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        include_sensitive_columns=False,
        include_intercept_term=True,
        file_type='csv')
    
    # Define behavioral constraints
    constraint_strs = ['0.9 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'] 
    deltas = [0.05]

    # For each constraint (in this case only one), make a parse tree
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]
        delta = deltas[ii]

        # Create parse tree object
        parse_tree = ParseTree(delta=delta,regime='supervised',
            sub_regime='classification',columns=columns)

        # Fill out tree
        parse_tree.build_tree(
            constraint_str=constraint_str,
            delta_weight_method='equal')
        
        parse_trees.append(parse_tree)

    # Save spec object, using defaults where necessary
    spec = SupervisedSpec(
        dataset=dataset,
        model_class=model_class,
        frac_data_in_safety=0.6,
        primary_objective=primary_objective,
        parse_trees=parse_trees,
        initial_solution_fn=model_class().fit,
        use_builtin_primary_gradient_fn=True,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : 0.5,
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 1000,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )

    spec_save_name = os.path.join(save_dir,'spec.pkl')
    save_pickle(spec_save_name,spec)
    print(f"Saved Spec object to: {spec_save_name}")

{% endhighlight python %}
</div>

<p>
Let's take a close look at the instantiation of <code class='highlight'>SupervisedSpec</code> in the code above so we can understand each of the arguments:
{% highlight python %}
spec = SupervisedSpec(
        dataset=dataset,
        model_class=model_class,
        frac_data_in_safety=0.6,
        primary_objective=primary_objective,
        parse_trees=parse_trees,
        initial_solution_fn=model_class().fit,
        use_builtin_primary_gradient_fn=True,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : 0.5,
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 1000,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )
{% endhighlight python %}

First, the object takes the <code class='highlight'>dataset</code> and <code class='highlight'>model_class</code>. Then, we set <code class='highlight'>frac_data_in_safety=0.6</code>, which specifies that 60% of the data points in our dataset will be used for the safety test. The remaining 40% of the points will be used for candidate selection. Next, we specify the <code class='highlight'>primary_objective</code> function and <code class='highlight'>parse_trees</code> list that we defined above in the script. In our case we only have one parse tree (because there is one parse tree per constraint), but it still must be passed as a list. <code class='highlight'>initial_solution_fn</code> specifies the function we will use to provide the initial solution to candidate selection, which we are setting to <code class='highlight'>model_class().fit</code>. Because <code class='highlight'>model_class()</code> refers to our <code class='highlight'>LogisticRegressionModel()</code>, <code class='highlight'>model_class().fit</code> refers to that class' <code class='highlight'>fit</code> method. This method is just a wrapper for  <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit">scikit-learn's LogisticRegression fit method</a>. The reason we use this method to create an initial solution is so that we start gradient descent with model weights that minimize the primary objective (in the absence of constraints). Because we have constraints, this initial solution is not necessarily the true optimum of our optimization problem, but it can help us find the true optimum much more efficiently in some cases. 
</p>

<p>
The next argument is <code class='highlight'>use_builtin_primary_gradient_fn=True</code>. This is telling the code to use a function that is part of the Engine library already to calculate the gradient of the primary objective. Recall that earlier in the script we set the primary objective to be the logistic loss with the line: <code class='highlight'>primary_objective = model_class().sample_logistic_loss</code>. Built-in gradients exist for some common objective functions (see <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/models.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/models.py</a>), including the sample logistic loss. If you use a custom primary objective function, there will definitely not be a built-in gradient function for your objective and <code class='highlight'>use_builtin_primary_gradient_fn=True</code> will raise an error. Setting <code class='highlight'>use_builtin_primary_gradient_fn=False</code> will cause the Engine to use automatic differentiation to calculate the gradient of the primary objective instead. There is also a parameter for specifying a custom function for the gradient of the primary objective as well, but we will not cover that in this tutorial. 
</p>

<p>
The next argument is <code class='highlight'>optimization_technique='gradient_descent'</code>, which specifies how we will search for a candidate solution during candidate selection. The other option for this argument is "barrier_function", which we will not cover here. The argument <code class='highlight'>optimizer='adam'</code> instructs the code to use the adam optimizer during gradient descent. The final argument <code class='highlight'>optimization_hyperparams</code> is for setting the parameters of gradient descent, which include:
<ul>
<li>'lambda_init': the initial value of the Lagrange multiplier </li>
<li>'alpha_theta': the initial learning rate for the model parameters</li>
<li>'alpha_lamb': the initial learning rate for the Lagrange multiplier</li>
<li>'beta_velocity': the decay rate of the velocity (also called momentum) term</li>
<li>'beta_rmsprop': the decay rate of the rmsprop term</li>
<li>'num_iters': the number of iterations of gradient descent to run</li>
<li>'gradient_library': the library to use for calculating gradients automatically. Currently "autograd" is the only option.</li>
<li>'hyper_search': how to conduct search over hyperparameters. Set to None which does not perform a search. </li>
<li>'verbose': whether to print out iteration progress during gradient descent</li>
</ul>


For more details about the <code class='highlight'>SupervisedSpec</code> object see the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.SupervisedSpec.html">Engine docs </a>.
</p>


<h5> Creating the specification object from the Seldonian interface GUI</h5>

<p>
    The instructions for using the Seldonian Interface GUI are described: <a href="https://seldonian-toolkit.github.io/GUI/build/html/index.html">here</a>. Start up the GUI and then upload the data file you downloaded from the link in the <a href="#dataset_prep"> Dataset preparation section</a> in the "Data file" field of the "Data and metadata setup" section. Then select the "supervised" regime and "classification" sub-regime from the drop-downs in that section. Copy and paste the following text string into the "All attributes" field: 
</p>
<p>
    <code>
    c__account_status_A11,c__account_status_A12,c__account_status_A13,c__account_status_A14,c__credit_history_A30,c__credit_history_A31,c__credit_history_A32,c__credit_history_A33,c__credit_history_A34,c__purpose_A40,c__purpose_A41,c__purpose_A410,c__purpose_A42,c__purpose_A43,c__purpose_A44,c__purpose_A45,c__purpose_A46,c__purpose_A48,c__purpose_A49,c__savings_accounts_A61,c__savings_accounts_A62,c__savings_accounts_A63,c__savings_accounts_A64,c__savings_accounts_A65,c__employment_since_A71,c__employment_since_A72,c__employment_since_A73,c__employment_since_A74,c__employment_since_A75,F,M,c__other_debtors_A101,c__other_debtors_A102,c__other_debtors_A103,c__property_A121,c__property_A122,c__property_A123,c__property_A124,c__other_installment_plans_A141,c__other_installment_plans_A142,c__other_installment_plans_A143,c__housing_A151,c__housing_A152,c__housing_A153,c__job_A171,c__job_A172,c__job_A173,c__job_A174,c__telephone_A191,c__telephone_A192,c__foreign_worker_A201,c__foreign_worker_A202,n__months,n__credit_amount,n__installment_rate,n__present_residence_since,n__age_yrs,n__num_existing_credits,n__num_people_liable,credit_rating
    </code>
</p>

<p>
    This is the list of all of the columns in the dataset, including the label column. Enter: <code>M,F</code> into the "Sensitive attributes" field, and <code>credit_rating</code> into the "Label column" field.
</p>

<p>
    Scroll down to the "Constraint building blocks" area and click the "Disparate impact" button. This will auto-fill Constraint #1 with a preconfigured constraint for disparate impact, which will have the form that we defined above. The one change you will need to make is to remove the constant block titled "0.8" and create a new constant block titled "0.9". Drag the new constant block into the constraint so that it becomes solid. Type 0.05 into the field titled "${\delta} = $" just below where the constraint function was auto-filled. Then hit the submit button. A dialog box should show up displaying: "Saved ./spec.pkl", which indicates that the specification object has been saved as a pickle file to the directory where you launched the GUI. 
</p>

<h3> Running the Seldonian Engine </h3>
<p>
    We are now ready to run the Seldonian algorithm using the spec file generated in the previous step, regardless of the method used. The code below modifies some defaults of the spec object that we created and then runs the Seldonian algorithm using the modified spec object. Create a file called "loan_fairness.py" and copy the code above into the file. You may need to change the line <code class='highlight'>specfile = './spec.pkl'</code> to point it to where you created that file in the previous step.

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">

{% highlight python %}
import os

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
    # Load loan spec file
    specfile = './spec.pkl'
    spec = load_pickle(specfile)
    
    spec.use_builtin_primary_gradient_fn = False
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
    spec.optimization_hyperparams['num_iters'] = 1000
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test!")
    else:
        print("Failed safety test")
    print()
    print("Primary objective (log loss) evaluated on safety dataset:")
    print(SA.evaluate_primary_objective(branch='safety_test',theta=solution))
        

{% endhighlight python %}
</div>
Run the script via the command line:
{% highlight bash %}
$ python loan_fairness.py
{% endhighlight bash %}

You should see some output like:
{% highlight python %}
Initial solution: 
[-1.44155523e+00  8.09258063e-01  6.80745965e-01 -5.95671392e-01
 -8.94987044e-01  3.34404367e-01  6.06307871e-01 -1.39965678e-02
 -6.38347222e-02 -8.63535356e-01  1.64957467e-01 -7.47980062e-01
 -5.95286690e-01 -1.35646079e-01  6.01829638e-02  2.90844178e-01
  1.21636244e-01  8.61277619e-01 -4.05835816e-02  1.99435318e-02
  6.10082399e-01  8.51243436e-02 -1.63380124e-01 -1.91609879e-01
 -3.40871149e-01  4.81567617e-01  1.50176130e-01  7.28206974e-02
 -7.81674833e-01  7.64559796e-02  1.80107871e-01  6.37038368e-01
 -8.17800648e-01 -1.83439537e-01 -2.03213452e-01 -4.02715177e-02
  4.26270097e-01 -3.94382892e-02  3.39194218e-01 -3.00410337e-01
  5.90957423e-01  1.34099781e-01 -7.25711612e-01  2.69899358e-02
 -3.17786349e-01 -2.22729289e-02  3.12414934e-01  1.98307003e-01
 -1.98961411e-01  1.24484979e-03 -1.89925846e-03  3.99257232e-01
  4.10266535e-01  5.13111566e-01 -7.92978584e-02  3.03845174e-02
  2.43546364e-01  1.98443152e-01]
Iteration 0
Iteration 10
Iteration 20
Iteration 30
Iteration 40
...
Wrote /Users/ahoag/beri/code/engine-repo/examples/logs/candidate_selection_log0.p with candidate selection log info
Passed safety test!

Primary objective (log loss) evaluated on safety dataset:
0.5582021704446299
{% endhighlight %}
The exact numbers you see might differ slightly depending on your machine's random number generator. However, the safety test should pass and the log loss on the safety dataset should be very similar. 
</p>

<p> Note the line saying: 
{% highlight python %}
Wrote /Users/ahoag/beri/code/engine-repo/examples/logs/candidate_selection_log0.p with candidate 
{% endhighlight python %}
This is a pickle file containing the values of various parameters during each step of the gradient descent algorithm that was run during candidate selection. The path displayed here will differ and instead point to somewhere on your computer. As part of the Engine library, we provide a plotting function that is designed to help visualize the contents of this file. The following script will run that function on the file. Note that you will have to change the path for <code class='highlight'>cs_file</code> to point it to the file that was created on your machine. 
</p>
<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">

{% highlight python %}
from seldonian.utils.io_utils import load_pickle
from seldonian.utils.plot_utils import plot_gradient_descent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load loan spec file
    cs_file = '/Users/ahoag/beri/code/engine-repo/examples/logs/candidate_selection_log0.p'
    solution_dict = load_pickle(cs_file)
    
    fig = plot_gradient_descent(solution_dict,
        primary_objective_name='log loss',
        save=False,savename='test.png')
    plt.show()
{% endhighlight python %}

</div>

Running this script will generate a figure like this:

<div align="center">
    <figure>
        <img src="{{ "/assets/img/loan_cs.png" | relative_url}}" class="img-fluid mt-4" style="width: 75%"  alt="Candidate selection"> 
        <figcaption align="left"> <b>Figure 1</b> - How the parameters of the Lagrangian optimization problem changed during gradient descent on our loan fairness problem. The panels show the values of the (left) primary objective, $f({\theta})$, i.e.,  the log loss, (middle left) single lagrange multiplier, ${\lambda_1}$, (middle right) high confidence upper bound (HCUB) on the disparate impact constraint function, ${g_1}(\theta)$, and finally the Lagrangian $L(\theta,\lambda)$. The dotted lines indicate where the optimum was found. The optimum is defined as the feasible solution with the lowest value of the primary objective. A feasible solution is one where $\mathrm{HCUB}(g_i(\theta)) \leq 0, i \in \{1 ... n\}$. In this example, we only have one constraint and the infeasible region is shown in red in the middle right plot. </figcaption>
    </figure>
</div>
<p>
Visualizing candidate selection can help you tune your optimization hyperparameters in your spec object. For example, if $\theta$ is never escaping the infeasible region and your Seldonian algorithm is returning NSF (i.e., "No Solution Found"), then you may be able to obtain a solution by running gradient descent for more iterations or with different learning rates or beta values.  If you are still seeing NSF after hyperparameter exporation, you may not have enough data or your constraints may be too strict. Running a Seldonian Experiment can help determine why you are not able to obtain a solution.
</p>

<h3> Running a Seldonian Experiment </h3>

<p>
Seldonian Experiments are a way to thoroughly evaluate the performance and safety of Seldonian algorithms, beyond what can be achieved with a single run of the Engine. A Seldonian Experiment runs a Seldonian algorithm many times using variable amounts of input data and creates three plots: 1) Performance, 2) Solution rate, and 3) Failure rate as a function of the amount of data used. The <a href="https://seldonian-toolkit.github.io/Experiments"> Seldonian Experiments library</a> was designed to help implement Seldonian Experiments. We recommend reading the <a href="https://seldonian-toolkit.github.io/Experiments/build/html/overview.html">Experiments overview</a> before continuing here. If you have not already installed the Experiments library, follow the instructions <a href="{{ "/tutorials/install_toolkit_tutorial/" | relative_url}}">here</a> to do so.
</p>

<p> In order to calculate performance and failure rate for any given experimental trial, we need a ground truth dataset. To approximate ground truth, we bootstrap resample from our original dataset and assume that the bootstrap data distribution is the ground truth distribution. While this does not tell you what the performance rate or failure rate will be on your actual problem, it does give a reasonable estimate.
</p>

<p>
    Here is an outline of the experiment we will run: 
    <ul>
        <li>Create an array of data fractions, which will determine how much of the data to use as input to the Seldonian algorithm in each trial. We will use 15 different data fractions, which will be log-spaced between 0.001 and 1.0. This array times the number of data points in the original dataset (1000) will make up the horizontal axis of the three plots.</li>
        <li>Create 50 resampled datasets (one for each trial) by sampling with replacement from the dataset we used as input to the Engine above. Each resampled dataset will have the same number of rows (1000) as the original dataset. We use 50 trials so that we can compute uncertainties on the plotted quantities at each data fraction. <b>We will use the original dataset as the ground truth dataset</b> for calculating the performance and safety metrics.</li>
        <li>
        For each <code class='highlight'>data_frac</code> in the array of data fractions, run 50 trials. In each trial, use only the first <code class='highlight'>data_frac</code> fraction of the corresponding resampled dataset to run the Seldonian algorithm using the Seldonian Engine. We will use the same spec file we used above for each run of the Engine, where only the <code class='highlight'>dataset</code> parameter to <code class='highlight'>SupervisedSpec</code> will be modified for each trial. This will generate 15x50=750 total runs of the Seldonian algorithm. Each run will consist of a different set of fitted model parameters, or "NSF" if no solution was found. </li>
        <li>For each <code class='highlight'>data_frac</code>, if a solution was returned that passed the safety test, calculate the mean and standard error on the performance (e.g., logistic loss or accuracy) across the 50 trials at this <code class='highlight'>data_frac</code> using the fitted model parameters evaluated on the ground truth dataset. This will be the data used for the first of the three plots. Also record how often a solution was returned and passed the safety test across the 50 trials. This fraction, referred to as the "solution rate", will be used to make the second of the three plots. Finally, for the trials that returned solutions that passed the safety test, calculate the fraction of trials for which the disparate impact statistic, $g_1(\theta)$, was violated, i.e., $g_1(\theta) > 0$, on the ground truth dataset. The fraction violated will be referred to as the "failure rate" and will make up the third and final plot. </li>
    </ul>
</p>
<p>
We will run this experiment for the Seldonian algorithm as well as for three other models. Two are baseline models: 1) a random classifier that always predicts $p=0.5$ for the positive class regardless of input, 2) a simple logistic regression model with no behavioral constraints. The third model comes from another fairness-aware machine learning library called <a href="https://fairlearn.org/">Fairlearn</a>. We will describe the Fairlearn model used in more detail below. Each model requires its own experiment, but the main parameters of the experiment such as the number of trials and data fractions, as well as the metrics we will calculate (performance, solution rate, and failure rate), are identical. This will allow us to compare the Seldonian algorithm to these other models on the same Three Plots. 
</p>
<p>
    Now we will show how to implement the described experiment using the Experiments library. At the center of the Experiments library is the <code class='highlight'>PlotGenerator</code> class, and in our particular example the <code class='highlight'>SupervisedPlotGenerator</code> child class. The goal of the following script is to setup this object, use its methods to run our experiments, and then to make the three plots.  
</p>

<p>
    First, the imports we will need:

{% highlight python %}
import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score
{% endhighlight %}
</p>
<p>
Now we will set up the parameters for the experiments, such as the data fractions we want to use and how many trials at each data fraction we want to run. We will use log-spacing for the data fractions (the horizontal axis of each of the three plots) so that when plotted on a log scale the data points will be evenly separated. We will use 50 trials at each data fraction so that we can sample a reasonable spread on each quantity at each data fraction. 
</p>

<p>
Fairlearn's fairness definitions are rigid and do not exactly match the definition we used in the Engine. To approximate the same definition of disparate impact as ours, we use their definition of demographic parity with a ratio bound of four different values. We will show later that we can change our constraint to match theirs exactly, and the results we find do not change significantly. 
</p>
<p>
Each trial in an experiment is independent of all other trials, so parallelization can speed experiments up enormously. Set this parameter to however many CPUs you want to use. Note: using 7 CPUs, this entire script takes 5-10 minutes to run on an M1 Macbook Air. The results for each experiment we run will be saved in subdirectories of <code class='highlight'>results_dir</code>. <code class='highlight'>n_workers</code> is how many parallel processes will be used for running the experiments. 
</p>

<p>
{% highlight python %}
if __name__ == "__main__":
    # Parameter setup
    run_experiments = True
    make_plots = True
    save_plot = True
    constraint_name = 'disparate_impact'
    fairlearn_constraint_name = constraint_name
    fairlearn_epsilon_eval = 0.9 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
    fairlearn_eval_method = 'two-groups' # the method for evaluating the Fairlearn model, must match Seldonian constraint definition
    fairlearn_epsilons_constraint = [0.1,1.0,0.9,1.0] # the epsilons used in the fitting constraint
    performance_metric = 'log_loss'
    n_trials = 50
    data_fracs = np.logspace(-3,0,15)
    n_workers = 8
    verbose=True
    results_dir = f'results/loan_{constraint_name}_seldo_log_loss'
    plot_savename = os.path.join(results_dir,f'{constraint_name}_{performance_metric}.png')

{% endhighlight python %}
</p>
<p>
Now we will need to load the same spec object that we created for running the Engine. As before, change the path to the where you saved this file.
{% highlight python %}
   # Load spec
    specfile = '../interface_outputs/loan_disparate_impact_0p9/spec.pkl'
    spec = load_pickle(specfile)

    spec.primary_objective = spec.model_class().sample_logistic_loss
    spec.use_builtin_primary_gradient_fn = False
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
    spec.optimization_hyperparams['num_iters'] = 1500
{% endhighlight python %}
</p>
<p>
Next, we will set up the ground truth dataset on which we will calculate the performance and the failure rate. In this case, it is just the original dataset.

{% highlight python %}
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset
    label_column = dataset.label_column
    include_sensitive_columns = dataset.include_sensitive_columns
    include_intercept_term = dataset.include_intercept_term

    test_features = dataset.df.loc[:,
        dataset.df.columns != label_column]
    test_labels = dataset.df[label_column]

    if not include_sensitive_columns:
        test_features = test_features.drop(
            columns=dataset.sensitive_column_names) 

    if include_intercept_term:
        test_features.insert(0,'offset',1.0) # inserts a column of 1's in place  
{% endhighlight python %}
</p>
<p>
We need to define what function <code class='highlight'>perf_eval_fn</code> we will use to evaluate the performance of the model. In this case we will use the logistic (or "log") loss, which happens to be the same as our primary objective. We also define <code class='highlight'>perf_eval_kwargs</code> which will be passed to the <code class='highlight'>SupervisedPlotGenerator</code> so that we can evaluate the performance evaluation funciton on the model in each of our experiment trials. 

{% highlight python %}
    # Setup performance evaluation function and kwargs 
    # of the performance evaluation function

    def perf_eval_fn(y_pred,y,**kwargs):
        if performance_metric == 'log_loss':
            return log_loss(y,y_pred)
        elif performance_metric == 'accuracy':
            print("calculating accuracy")
            return accuracy_score(y,y_pred > 0.5)

    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        } 
{% endhighlight python %}
</p>

<p>
Now we instantiate the plot generator, passing in the parameters from variables we defined above.

{% highlight python %}
    plot_generator = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='resample',
        perf_eval_fn=perf_eval_fn,
        constraint_eval_fns=[],
        results_dir=results_dir,
        perf_eval_kwargs=perf_eval_kwargs,
        )
{% endhighlight python %}
</p>

<p>
We will first run our two baseline experiments, which we can do by calling the <code class='highlight'>run_baseline_experiment()</code> method of the plot generator and passing in the baseline model name of choice. 

{% highlight python %}
    # Baseline models
    if run_experiments:
        plot_generator.run_baseline_experiment(
            model_name='random_classifier',verbose=True)

        plot_generator.run_baseline_experiment(
            model_name='logistic_regression',verbose=True)

{% endhighlight python %}
</p>

<p>
Similarly, to run our Seldonian experiment, we call the corresponding method of the plot generator:

{% highlight python %}
        # Seldonian experiment
        plot_generator.run_seldonian_experiment(verbose=verbose)

{% endhighlight python %}
</p>

<p>
The last experiment we will run is the Fairlearn experiment. While Fairlearn does not have a disparate impact constraint, disparate impact can be constructed using their demographic parity constraint with a ratio bound. Under the hood, we are using the following Fairlearn model, where <code class='highlight'>fairlearn_epsilon_constraint</code> is one of the four values in the list we defined above. <b>Note that the following code is not part of the experiment script and you do not need to run it. It is simply to illuminate how we are implementing the Fairlearn model in our experiment. </b>. 
{% highlight python %}
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity
fairlearn_constraint = DemographicParity(
                ratio_bound=fairlearn_epsilon_constraint)
classifier = LogisticRegression()        
mitigator = ExponentiatedGradient(classifier, 
            fairlearn_constraint)
{% endhighlight python %}
</p>

<p>

The way Fairlearn handles sensitive columns is different than way we handle them in the Seldonian Engine library. In the Engine, we one-hot encode sensitive columns. In Fairlearn, they integer encode. This is why we have two columns, M and F, in the dataset used for the Engine, whereas they would have one for defining the sex. It turns out that our M column encodes both sexes since it is binary-valued (0=female, 1=male), so we can just tell Fairlearn to use the "M" column. We also can drop the "offset" column which is a column of ones that we use in the Engine if the dataset.include_intercept_term parameter is True, which in our case it was. 

{% highlight python %}
    ######################
    # Fairlearn experiment 
    ######################

    fairlearn_sensitive_feature_names=['M']
    
    # Make dict of test set features, labels and sensitive feature vectors
    if 'offset' in test_features.columns:
        test_features_fairlearn = test_features.drop(columns=['offset'])
    else:
        test_features_fairlearn = test_features
    fairlearn_eval_kwargs = {
        'X':test_features_fairlearn,
        'y':test_labels,
        'sensitive_features':dataset.df.loc[:,
            fairlearn_sensitive_feature_names]
        }

    if run_experiments:
        for fairlearn_epsilon_constraint in fairlearn_epsilons_constraint:
            plot_generator.run_fairlearn_experiment(
                verbose=verbose,
                fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
                fairlearn_constraint_name=fairlearn_constraint_name,
                fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
                fairlearn_epsilon_eval=fairlearn_epsilon_eval,
                fairlearn_eval_kwargs=fairlearn_eval_kwargs,
                )
{% endhighlight python %}
</p>
<p>
Finally, we make the three plots. When we set our variables at the top of the script, we set <code class='highlight'> save_plot = False</code>, so the plot will be displayed to the screen but not saved. If we want to save the plot and then view it from disk afterwards, set <code class='highlight'> save_plot = True</code> at the top of the script. 

{% highlight python %}        
    if make_plots:
        if save_plot:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,
                savename=plot_savename)
        else:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric)
{% endhighlight python %}
</p>

<p>
Here is the entire script all together, which we will call <code>generate_threeplots.py</code>:
{% highlight python %}      
# generate_threeplots.py  
import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score

if __name__ == "__main__":
    # Parameter setup
    run_experiments = True
    make_plots = True
    save_plot = False
    constraint_name = 'disparate_impact'
    fairlearn_constraint_name = constraint_name
    fairlearn_epsilon_eval = 0.9 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
    fairlearn_eval_method = 'two-groups' # the method for evaluating the Fairlearn model, must match Seldonian constraint definition
    fairlearn_epsilons_constraint = [0.01,0.1,0.9,1.0] # the epsilons used in the fitting constraint
    performance_metric = 'log_loss'
    n_trials = 50
    data_fracs = np.logspace(-3,0,15)
    n_workers = 7
    verbose=True
    results_dir = f'results/loan_{constraint_name}_seldo_log_loss'
    plot_savename = os.path.join(results_dir,f'{constraint_name}_{performance_metric}.png')

    # Load spec
    specfile = f'../interface_outputs/loan_{constraint_name}_seldodef/spec.pkl'
    spec = load_pickle(specfile)

    spec.primary_objective = spec.model_class().sample_logistic_loss
    spec.use_builtin_primary_gradient_fn = False
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
    spec.optimization_hyperparams['num_iters'] = 1500

    os.makedirs(results_dir,exist_ok=True)

    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset
    label_column = dataset.label_column
    include_sensitive_columns = dataset.include_sensitive_columns
    include_intercept_term = dataset.include_intercept_term

    test_features = dataset.df.loc[:,
        dataset.df.columns != label_column]
    test_labels = dataset.df[label_column]

    if not include_sensitive_columns:
        test_features = test_features.drop(
            columns=dataset.sensitive_column_names) 

    if include_intercept_term:
        test_features.insert(0,'offset',1.0) # inserts a column of 1's in place

    # Setup performance evaluation function and kwargs 
    # of the performance evaluation function

    # perf_eval_fn = lambda y_pred,y,X: fbeta_score(y,y_pred,beta=2)
    def perf_eval_fn(y_pred,y,**kwargs):
        if performance_metric == 'log_loss':
            return log_loss(y,y_pred)
        elif performance_metric == 'accuracy':
            print("calculating accuracy")
            return accuracy_score(y,y_pred > 0.5)

    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        }

    plot_generator = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='resample',
        perf_eval_fn=perf_eval_fn,
        constraint_eval_fns=[],
        results_dir=results_dir,
        perf_eval_kwargs=perf_eval_kwargs,
        )

    # Baseline models
    if run_experiments:
        plot_generator.run_baseline_experiment(
            model_name='random_classifier',verbose=True)

        plot_generator.run_baseline_experiment(
            model_name='logistic_regression',verbose=True)

        # Seldonian experiment
        plot_generator.run_seldonian_experiment(verbose=verbose)


    ######################
    # Fairlearn experiment 
    ######################

    fairlearn_sensitive_feature_names=['M']
    
    # Make dict of test set features, labels and sensitive feature vectors
    if 'offset' in test_features.columns:
        test_features_fairlearn = test_features.drop(columns=['offset'])
    else:
        test_features_fairlearn = test_features
    fairlearn_eval_kwargs = {
        'X':test_features_fairlearn,
        'y':test_labels,
        'sensitive_features':dataset.df.loc[:,
            fairlearn_sensitive_feature_names]
        }

    if run_experiments:
        for fairlearn_epsilon_constraint in fairlearn_epsilons_constraint:
            plot_generator.run_fairlearn_experiment(
                verbose=verbose,
                fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
                fairlearn_constraint_name=fairlearn_constraint_name,
                fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
                fairlearn_epsilon_eval=fairlearn_epsilon_eval,
                fairlearn_eval_kwargs=fairlearn_eval_kwargs,
                )

    if make_plots:
        if save_plot:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,
                savename=plot_savename)
        else:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric)
{% endhighlight python %}
</p>

<p>
Running the script will produce the following plot (or something very similar depending on your machine's random number generator):

<div align="center">
    <figure>
        <img src="{{ "/assets/img/disparate_impact_log_loss.png" | relative_url}}" class="img-fluid mt-4" style="width: 65%"  alt="Disparate impact log loss"> 
        <figcaption align="left"> <b>Figure 2</b> - The Three Plots of a Seldonian Experiment shown for the UCI German Credit dataset, enforcing a disparate impact fairness constraint with a threshold of 0.9. Each panel shows the mean (point) and standard error (shaded region) of a quantity for several models: the Quasi-Seldonian model (QSA, blue), the two baseline models: 1) a random classifier (pink) that predicts the positive class with $p=0.5$ every time and 2) a logistic regression model without any constraints added (brown), and the Fairlearn model with four different values of epsilon, the ratio bound. (Left) the logistic loss of the models as a function of the number of training samples (determined from the data fraction array). (Middle) the fraction of trials at each data fraction that returned a solution. (Right) the fraction of trials that violated the safety constraint on the ground truth dataset. The black dashed line is set at the $\delta=0.05$ value that we set in our behavioral constraint. </figcaption>
    </figure>
</div>

The QSA takes more data to return a solution (middle panel) than the other models because it makes a high confidence guarantee that the solution it returns will be safe. Looking at the right panel, that guarantee is validated because the QSA never violates the fairness constraint on the ground truth dataset. The QSA's performance approaches the performance of the logistic regression baseline as the number of training samples approaches 1000, the size of the original dataset. The logistic regression model and even the fairness-aware Fairlearn model return solutions with less data, but both violate the constraint quite often. This is true for all four values of the disparate impact threshold, epsilon, and despite the fact that Fairlearn is using a behavioral constraint akin to disparate impact.  
</p>

<p>
Some minor points of these plots are:
<ul>
<li> The performance is not plotted for trials that did not return a solution. For the smallest data fractions (e.g., 0.001), only the random classifier returns a solution because it defined to always return the same solution independent of input. </li>
<li>The failure rate is not plotted for trials that did not return a solution for all models except QSA. The logistic regression baseline and Fairlearn can fail to converge, for example. However, the QSA will always return either a solution or "NSF", even when only one data point (i.e., <code class='highlight'>data_frac=0.001</code>) is provided as input. In cases where it returns "NSF" that solution is considered safe.  </li>
<li> The random classifier does not violate the safety constraint ever because its positive rate is 0.5 for all datapoints, regardless of sex.</li>
</ul>
</p>

<p>
We mentioned that Fairlearn cannot exactly enforce the disparate impact constraint we defined: $0.9 - \text{min}( (\text{PR} | [\text{M}]) / (\text{PR} | [\text{F}]), (\text{PR} | [\text{F}]) / (\text{PR} | [\text{M}]) )$. This is because Fairlearn's <a href="https://fairlearn.org/v0.7.0/user_guide/mitigation.html#fairness-constraints-for-binary-classification">fairness constraints for binary classification</a> only compare statistics like positive rate between a single sensitive group and the mean of the group. The Seldonian Engine is flexible in how its constraints can be defined, and we can tweak our disparate impact constraint definition to match the Fairlearn definition. To match the Fairlearn definition, our constraint must take the form: $0.9 - \text{min}( (\text{PR} | [\text{M}]) / (\text{PR}), (\text{PR}) / (\text{PR} | [\text{M}]) )$, where the only thing we have changed is substituting $(\text{PR} | [\text{F}])$ (positive rate, given female) in our original constraint with $(\text{PR})$, the mean positive rate. 
</p>

<p> 
Let us re-run the same experiment above with this new constraint. First, we need to create a new spec file, where the only difference is the constraint definition. This will build a new parse tree from the new constraint. Replace the line in <code>createSpec.py</code> above:
{% highlight python %}
# Define behavioral constraints
constraint_strs = ['0.9 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))']
{% endhighlight python %}
with: 
{% highlight python %}
# Define behavioral constraints
constraint_strs = ['0.9 - min((PR | [M])/(PR),(PR)/(PR | [M]))']
{% endhighlight python %}
Then change the line where the spec file is saved so it does not overwrite the old spec file from:

{% highlight python %}
spec_save_name = os.path.join(save_dir,'spec.pkl')
{% endhighlight python %}

to something like:

{% highlight python %}
spec_save_name = os.path.join(save_dir,'spec_fairlearndef.pkl')
{% endhighlight python %}
</p>

<p> 
Once that file is saved, you are ready to run the experiments again. There are only a few changes you will need to make to the file <code>generate_threeplots.py</code>. They are:
<ul>
<li> Point <code class='highlight'>specfile</code> to the new spec file you just created.
</li>
<li> Change <code class='highlight'>fairlearn_eval_method</code> to <code class='highlight'>"native"</code>. </li>
<li> Change <code class='highlight'>results_dir</code> to a new directory so you do not overwrite the results from the previous experiment. </li>
</ul>  
</p>

<p>
Running the script with these changes will produce a plot that should look very similar to this: 

<div align="center">
    <figure>
        <img src="{{ "/assets/img/disparate_impact_log_loss_fairlearndef.png" | relative_url}}" class="img-fluid mt-4" style="width: 65%"  alt="Disparate impact log loss"> 
        <figcaption align="left"> <b>Figure 3</b> - Same as Figure 2, but with the definition of disparate impact that Fairlearn uses, i.e., $0.9 - \text{min}( (\text{PR} | [\text{M}]) / (\text{PR}), (\text{PR}) / (\text{PR} | [\text{M}]) )$. In this experiment, we only used a single Fairlearn model (with $\epsilon=0.9$), because the constraint was identical to the constraint used in the QSA model, which was not true in the previous experiment. </figcaption>
    </figure>
</div>

The results are very similar to the previous experiment. As before, the QSA takes more data than the baselines and Fairlearn to return solutions, but those solutions are safe. The solutions from the logistic regression baseline and the Fairlearn model are frequently unsafe. 
</p>

<p>
    We could use the same procedure we just carried out to change the fairness definition entirely. For example, another common definition of fairness is equalized odds, which ensures that the false positive rates and false negative rates simultaneously do not differ to within a certain tolerance. Let's define the constraint string that the Engine can understand: $\text{abs}((\text{FPR} | [\text{M}]) - (\text{FPR} | [\text{F}])) + \text{abs}((\text{FNR} | [\text{M}]) - (\text{FNR} | [\text{F}])) - 0.2$. Repeating the same procedure as above to replace the constraint with this one and ensuring:
{% highlight python %}
constraint_name = 'equalized_odds'
fairlearn_constraint_name = constraint_name
fairlearn_epsilon_eval = 0.2 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
fairlearn_eval_method = 'two-groups' # the epsilon used to evaluate g, needs to be same as epsilon in our definition
fairlearn_epsilons_constraint = [0.01,0.1,0.2,1.0] # the epsilons used in the fitting constraint
{% endhighlight python %}
We could run the experiment for this constraint, obtaining the following figure: 
<div align="center">
    <figure>
        <img src="{{ "/assets/img/equalized_odds_log_loss.png" | relative_url}}" class="img-fluid mt-4" style="width: 65%"  alt="Disparate impact log loss"> 
        <figcaption align="left"> <b>Figure 4</b> - Same as Figure 2, but enforcing equalized odds instead of disparate impact. </figcaption>
    </figure>
</div>
</p>
<h3>Summary</h3>
<p>
In this tutorial, we demonstrated how to use the Seldonian Toolkit to build a predictive model that enforces a variety of fairness constraints on the German Credit dataset. We covered how to format the dataset and metadata so that they can be used in the Seldonian Engine. Using the Engine, we ran a Seldonian algorithm and confirmed that we were able to find a safe solution. We then ran Seldonian Experiments to evaluate the true performance and safety of our quasi-Seldonian algorithm (QSA). We found that the QSA can flexibly satisfy a range of custom-defined fairness constraints, and that the model does not violate the constraints. We compared the QSA to several baselines that do not enforce any behavioral constraints as well as one model, Fairlearn, that does enforce constraints. We found that the performance of the QSA approaches the performance of a logistic regression model that lacks constraints. The logistic regression model, and the fairness-aware Fairlearn model, frequently violate the constraints for some definitions of fairness. 
</p>

</div>