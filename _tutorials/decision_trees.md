---
layout: tutorial
permalink: /tutorials/dtree_tutorial/
prev_url: /tutorials/new_baseline/
prev_page_name: (K) Creating a new baseline for Seldonian Experiments
title: Seldonian \| Tutorial K
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    
<h2 align="center" class="mb-3">Tutorial L: Creating Fair Decision Trees (for binary classification)</h2>
<a href="https://colab.research.google.com/github/seldonian-toolkit/Tutorials/blob/main/tutorial_d_fairness_for_automated_loan_approval_systems.ipynb" target="blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<hr class="my-4">

<!-- Table of contents -->
<h3> Contents </h3>
<ul>
    <li> <a href="#intro">Introduction</a> </li>
    <li> <a href="#outline">Outline</a> </li>
    <li> <a href="#model_desc">How does the SDTree model work?</a></li>
    <li> <a href="#gpa">Applying the SDTree to the GPA prediction problem from Tutorial E.</a> </li>
    <ul>
        <li> <a href="#spec_object">Creating the specification object</a> </li>
        <li> <a href="#gpa_experiments">Running the Seldonian Experiments</a> </li>
    </ul>
    <li> <a href="#summary">Summary</a> </li>
</ul>
<hr class="my-4">

<h3 id="intro">Introduction</h3>

<p>For tabular datasets, tree-based models are often the first choice for ML practitioners. Depending on the specific model, they can have advantages over more complex models like artificial neural networks, such as explainability, a smaller memory footprint, and lower latency (both at training and inference time). There are many types of tree-based models, but at the foundation of all of them is the decision tree. 
</p>

<!-- Instead, training a decision tree involves building a tree recursively, where each node represents a specific value of (usually) a single feature in the dataset that split the training data into (usually) two subsets.  Two common criteria for deciding the feature splits are the information gain or gini index (also called gini impurity). Using these criteria for splits in the tree often results in accurate predictions on unseen data, although vanilla decision trees can suffer from overfitting to the training data. -->

<p> As with most standard ML models, simply training a decision tree using an off-the-shelf algorithm such as Classification And Regression Trees (CART; Breiman et al. 1984) can lead to a model that makes unfair predictions with respect to sensitive attributes (e.g. race, gender), even when those attributes are not explictly used as features in the dataset. In the context of the Seldonian Toolkit, this means that if we use CART during candidate selection, the safety test is unlikely to pass. The likelihood of the safety test passing is of course dependent on the strictness of the behavioral constraints. We want candidate selection to be effective at enforcing <i>any</i> set of behavioral constraints, while simultaneously optimizing for the primary objective (e.g., accuracy). When it is not possible to satisfy the behavioral constraints, the safety test will inform us of this.
</p>

<p>The main methods we have found to be effective for general-purpose candidate selection are i) black-box optimization with a barrier function and ii) KKT optimization (for more details see the <a href="{{ "/tutorials/alg_details_tutorial/" | relative_url}}">algorithm details tutorial</a>). However, both of these approaches assume parameter-based models. The CART algorithm is incompatible with these techniques because the trees it trains are non-parameteric. Rather than alter CART, we decided to take an already-built decision tree and run a second optimization procedure on it. This new procedure relabels the leaf node probabilities using KKT optimization, which takes into account the behavioral constraints during relabeling. The result is a general-purpose Seldonian decision tree (SDTree) model that can be used to ensure that behavioral constraints are satisfied with high-confidence on tabular problems.
</p>

<p>
    <b>Note 1:</b> The SDTree model discussed in this tutorial is only designed for <b>binary classification</b> problems. With some modification, it could be used for regression and multi-class classification problems, though that is beyond the scope of this tutorial.
</p>

<p>
    <b>Note 2:</b> While the approach is agnostic to the algorithm or library used to build the initial decision tree, in practice we use scikit-learn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">DecisionTreeClassifier</a>, which implements a variation on the CART algorithm. In order to use other libraries for building the tree, one would need to add some code to the toolkit (see the section <a href="#model_desc">How does the SDTree model work?</a> below for details). If this is something you would find valuable, please open an <a href="https://github.com/seldonian-toolkit/Engine/issues">issue</a> or submit a <a href="https://github.com/seldonian-toolkit/Engine/pulls">pull request</a>.
</p>


<h3 id="outline">Outline</h3>

<p>In this tutorial, we will:

<ul>
    <li>Cover how the SDTree is trained in candidate selection.</li>
    <li>Apply the SDTree to the GPA prediction problem from <a href="{{ "/tutorials/science_GPA_tutorial/" | relative_url}}"> Tutorial E</a>. </li>
    <li>Apply the SDTree to the COMPAS criminal recidivism dataset. </li>
</ul>
</p>
</div>
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="model_desc"> How does the SDTree model work?</h3>

<p>
    <b>Note:</b> this section is quite technical, and reading it in its entirety is not necessary for simply using the SDTree model. This section is intended for developers who are seeking to modify or better understand this model. 
</p>

<p>
    In brief, the Seldonian decision tree model relabels the leaf node probabilities so that the model's predictions enforce the behavioral constraints while maintaining high accuracy. 
</p>


<div align="center">
    <figure>
        <img src="{{ "/assets/img/dtree_tutorial/example_sklearn_tree.png" | relative_url}}" class="img-fluid mt-4" style="width: 100%"  alt="Scikit-learn decision tree"> 
        <figcaption align="left"> <b>Figure 1</b> - Example decision tree with <code class="codesnippet">max_depth=2</code> built using scikit-learn's DecisionTreeClassifier on a binary classification problem. Each internal node (including the root node) displays the feature split condition, the gini index, the number of samples that reach the node, and the "value" vector. Each leaf node displays the gini index, the number of samples that reach the node, and the "value" vector. For internal nodes, the first element of the value vector is the number of samples that meet the split condition, and the second number is the number that do not. For leaf nodes, the two numbers in the value vector indicate the number of samples whose true labels are 0 and 1, respectively. </figcaption>
    </figure>
</div>

<p>
    Let's assume we have obtained the decision tree in Figure 1 by training an instance of scikit-learn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">DecisionTreeClassifier</a> on some imaginary tabular dataset. Our goal is to make this model Seldonian by enforcing a behavioral constraint(s) with high confidence, e.g., $1-\delta = 0.95$. First, we obtain the probabilities of predicting the positive and negative class in each leaf node. The most straightforward way to do this is to consider the probabilities to be the fraction of samples that have each label. Taking the left-most leaf node as an example, the probabilities for predicting the 0th and 1st classes are:
</p>

$$ Pr(\hat{y} = 0) = \frac{8812}{8812+4471} \approx 0.66 $$
$$ Pr(\hat{y} = 1) = \frac{4471}{8812+4471} \approx 0.34 $$

<p>
    Because these probabilities are redundant (i.e., $ Pr(\hat{y} = 1) = 1.0 - Pr(\hat{y} = 0)$), we only need to compute one of them for each leaf node. Our convention is to only compute $Pr(\hat{y} = 1)$. Now, we want to tune these probabilities in the KKT optimization procedure subject to our behavioral constraint(s). However, probabilities are bounded between 0 and 1, and the optimization procedure works best with unbounded parameters. Therefore, for the $k$th leaf node, we define a parameter, $\theta_k \in (-\infty,\infty)$, such that:
</p>

$$ 
\begin{equation}
Pr(\hat{y} = 1)_k = \sigma\left(\theta_k\right) = \frac{1}{1+e^{-\theta_k}},
\label{theta2probs}
\end{equation}
$$
<p>
    which ensures that $Pr(\hat{y} = 1) \in (0,1)$. It is also helpful to express $\theta_k$ in terms of $Pr(\hat{y} = 1)_k$:
</p>

$$
\begin{equation} \theta_k = \ln\left(\frac{1}{\frac{1}{Pr(\hat{y} = 1)_k}-1}\right) \label{probs2theta}
\end{equation} 
$$ 

<p>
    As a result, our parameter vector $\theta$ has length $K$ where $K$ is the number of leaf nodes in the tree. We initialize theta using the probabilites of the decision tree that was fit by scikit-learn, mapping from probabilities to theta using Equation \ref{probs2theta}. At each step in the KKT optimization, we take the updated value of $\theta$, map it back to $Pr(\hat{y} = 1)$ for each leaf node using Equation \ref{theta2probs} and update the probability in the tree. We then make predictions using the updated tree. We repeat this process until the stopping criteria of the KKT procedure are met.
</p>

<p>
    The KKT optimization procedure uses the gradients of the primary objective and the upper bound on the constraint functions to take steps in parameter space. Both of the functions whose gradients are needed involve a forward pass of the decision tree. In the toolkit, these gradients are calculated using the <a href="https://github.com/HIPS/autograd">autograd</a> library, which (with some exceptions for NumPy and SciPy) is incompatible with external libraries such as scikit-learn. A way around this is to manually define the gradient of the forward pass, or more specifically the Jacobian matrix, and then instruct autograd to use the manual definition of the gradient. Fortunately, the Jacobian is straightforward in this case. Consider what the forward pass of this model does: for a given input data sample $X_i$, it computes the probability of classifying that sample into the positive class, $Pr(\hat{y}(X_i,\theta) = 1)$. The Jacobian matrix can then be written: 
</p>
    $$J_{i,j}=\frac{\partial \left( Pr(\hat{y}(X_i,\theta) = 1) \right)}{\partial \theta_j}$$
<p>
    If there are M samples and K leaf nodes, the Jacobian is a MxK matrix. Because the $\theta$ vector consists of a list of the leaf node probabilities (after transformation via Equation \ref{probs2theta}), the Jacobian has a value of 1 when the leaf node that is hit by sample $i$ has the probability value mapped from the corresponding value $\theta_k$ via equation \ref{theta2probs}, and 0 otherwise. Therefore, each row of the Jacobian has a single value of 1 and is 0 otherwise. 
</p>

<p>
    To make this more concrete, consider the decision tree in Figure 1 and the first 3 samples of an imaginary dataset. If the first sample's decision path through the ends at the second leaf node (from the left), the second sample ends at the first leaf node, and the third sample ends at the final (fourth) leaf node, the first 3 rows of the Jacobian would look like this:
</p>
{% highlight python %}
[
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    ...
]
{% endhighlight python %}

<p>
    We implemented a Seldonian decision tree model that uses scikit-learn's DecisionTreeClassifier as the initial model in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/trees/sktree_model.py">this module</a>. Here is the constructor of this class:
</p>
{% highlight python %}
class SKTreeModel(ClassificationModel):
    def __init__(self,**dt_kwargs):
        self.classifier = DecisionTreeClassifier(**dt_kwargs)
        self.has_intercept = False
        self.params_updated = False
{% endhighlight python %}
<p>
    The only inputs to the class are <code class="codesnippet">dt_kwargs</code>, which are any arguments that scikit-learn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">DecisionTreeClassifier</a> accepts, such as <code class="codesnippet">max_depth</code>. This enables training a starting model using any of the same hyperparameters that one can use with scikit-learn.
</p>
<p>
    Also in that module, we implement the autograd workaround, demonstrating the pattern one should follow if extending the toolkit to support decision trees built using other external libraries. 
</p>

</div>


<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="gpa">Applying the SDTree to the GPA prediction problem from Tutorial E.</h3>

<p>
    We strongly encourage reading the <a href="{{ "/tutorials/science_GPA_tutorial/" | relative_url}}">Tutorial E: Predicting student GPAs from application materials with fairness guarantees</a> before proceeding here. In that tutorial, we created a Seldonian binary classifier and applied it to five different fairness constraints. The underlying ML model we used in that tutorial was a logistic regressor. In this tutorial, we will compare the performance and safety properties of the Seldonian decision tree model to the Seldonian logistic regressor as well as several standard ML models that do not consider fairness for the same five fairness constraints.
</p>

<h5 id="gpa_spec">Creating the specification object</h5>

<p>
    There are two differences between the script we used to create spec objects in Tutorial E and the one we will create here: i) the model object is now the SDTree and ii) we customize the hyperparameters beyond the defaults here. In this case, we will instantiate the model with a max_depth of 5. We chose this to keep the model relatively simple, and we found qualitatively that the results did not differ significantly with larger max depths. max_depth is a hyperparameter that one would ideally tune. Below is a code snippet for creating the spec objects for each of the five fairness constraints from Tutorial E. Note that the data path and metadata path are local paths and must be adjusted to wherever you downloaded those two files. The two files can be downloaded from <a href="https://github.com/seldonian-toolkit/Engine/tree/main/static/datasets/supervised/GPA">here</a>.
</p>

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">

{% highlight python %}
import os
import autograd.numpy as np

from seldonian.dataset import DataSetLoader
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.utils.io_utils import save_pickle
from seldonian.spec import SupervisedSpec
from seldonian.models import objectives
from experiments.perf_eval_funcs import probabilistic_accuracy

from seldonian.models.trees.sktree_model import SKTreeModel, probs2theta

def initial_solution_fn(model,features,labels):
    probs = model.fit(features,labels)
    return probs2theta(probs)

if __name__ == "__main__":
    # 1. Dataset
    data_pth = '../../../../engine-repo-dev/static/datasets/supervised/GPA/gpa_classification_dataset.csv'
    metadata_pth = '../../../../engine-repo-dev/static/datasets/supervised/GPA/metadata_classification.json'
    regime='supervised_learning'
    sub_regime='classification'
    
    loader = DataSetLoader(
        regime=regime)
    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')
    all_feature_names = dataset.meta.feature_col_names
    sensitive_col_names = dataset.meta.sensitive_col_names
    frac_data_in_safety = 0.6
    
    # 2. Specify behavioral constraints 
    deltas = [0.05] # confidence levels
    for constraint_name in ["disparate_impact",
        "demographic_parity","equalized_odds",
        "equal_opportunity","predictive_equality"]:
        # Define behavioral constraints
        if constraint_name == 'disparate_impact':
            epsilon = 0.8
            constraint_strs = [f'min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M])) >= {epsilon}'] 
        elif constraint_name == 'demographic_parity':
            epsilon = 0.2
            constraint_strs = [f'abs((PR | [M]) - (PR | [F])) <= {epsilon}']
        elif constraint_name == 'equalized_odds':
            epsilon = 0.35
            constraint_strs = [f'abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) <= {epsilon}']
        elif constraint_name == 'equal_opportunity':
            epsilon = 0.2
            constraint_strs = [f'abs((FNR | [M]) - (FNR | [F])) <= {epsilon}']
        elif constraint_name == 'predictive_equality':
            epsilon = 0.2
            constraint_strs = [f'abs((FPR | [M]) - (FPR | [F])) <= {epsilon}']

        parse_trees = make_parse_trees_from_constraints(
            constraint_strs,deltas,regime=regime,
            sub_regime=sub_regime,columns=sensitive_col_names)

        # 3. Define the underlying machine learning model
        max_depth = 5
        model = SKTreeModel(max_depth=max_depth)

        # 4. Create a spec object
        # Save spec object, using defaults where necessary
        spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            sub_regime=sub_regime,
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=objectives.Error_Rate,
            use_builtin_primary_gradient_fn=False,
            initial_solution_fn=initial_solution_fn,
            optimization_technique='gradient_descent',
            optimizer="adam",
            optimization_hyperparams={
                'lambda_init'   : np.array([0.5]),
                'alpha_theta'   : 0.005,
                'alpha_lamb'    : 0.005,
                'beta_velocity' : 0.9,
                'beta_rmsprop'  : 0.95,
                'use_batches'   : False,
                'num_iters'     : 800,
                'gradient_library': "autograd",
                'hyper_search'  : None,
                'verbose'       : True,
            }
        )
        os.makedirs('./specfiles',exist_ok=True)
        spec_save_name = f'specfiles/gpa_{constraint_name}_{epsilon}_fracsafety_{frac_data_in_safety}_sktree_maxdepth{max_depth}_reparam_spec.pkl'
        save_pickle(spec_save_name,spec,verbose=True)
{% endhighlight python %}

<p>
    Running this code will create fives specfiles in the directory: <code>specfiles/</code>. We will skip running the engine and go straight to running an experiment. 
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h5 id="gpa_experiments">Running the Seldonian Experiments</h5>

<p>
    Here we show a script for running the Seldonian experiment for the first constraint: disparate impact. Changing the script to run the experiments for the other constraints amounts to changing the <code class="codesnippet">constraint_name</code> and <code class="codesnippet">epsilon</code> variables.
</p>

<p>
    The experiment we will run will consist of 20 trials, which is fewer than we ran in Tutorial E, but will be sufficient for comparing to the Seldonian logistic regressor and the baseline models. Note that in this experiment, we are omitting the Fairlearn models to which we compared in Tutorial E because that comparison would be redundant here. In this tutorial, we use a held-out test set of 1/3 of the original GPA dataset for calculating the performance (left plot) and fairness violation (right plot). We use the remaining 2/3 of the data for resampling to make the trial datasets. This is a slight difference between the experiment we ran in Tutorial E, but it does not affect the comparison between the models. Another minor difference between this tutorial and tutorial E is the use of probabilistic accuracy rather than deterministic accuracy for the performance metric. Again, this does not affect the comparison between the models because we use the same performance metric for all models in this tutorial. 
</p>

{% highlight python %}
import os
import numpy as np 
from sklearn.model_selection import train_test_split

from seldonian.utils.io_utils import load_pickle
from seldonian.dataset import SupervisedDataSet

from experiments.perf_eval_funcs import probabilistic_accuracy
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.baselines.logistic_regression import BinaryLogisticRegressionBaseline
from experiments.baselines.random_classifiers import (
    UniformRandomClassifierBaseline,WeightedRandomClassifierBaseline)
from experiments.baselines.decision_tree import DecisionTreeClassifierBaseline
from experiments.baselines.random_forest import RandomForestClassifierBaseline

from seldonian.models.trees.sktree_model import (
    SKTreeModel, probs2theta, sigmoid
)

def initial_solution_fn(model,features,labels):
    probs = model.fit(features,labels)
    return probs2theta(probs)

def perf_eval_fn(y_pred,y,**kwargs):
    return probabilistic_accuracy(y_pred,y)
        
def main():
    # Basic setup
    seed=4
    np.random.seed(seed)
    run_experiments = True
    make_plots = True
    save_plot = False
    include_legend = True
    frac_data_in_safety = 0.6

    model_label_dict = {
        'qsa':'Seldonian decision tree',
        'logreg_qsa':'Seldonian logistic regression',
        'decision_tree':'Sklearn decision tree (no constraint)',
        'logistic_regression': 'Logistic regression (no constraint)',
        }

    # Change these to the fairness constraint of interest
    constraint_name = 'disparate_impact' 
    epsilon = 0.8

    # experiment parameters
    max_depth = 5
    performance_metric = '1 - error rate'
    n_trials = 20
    data_fracs = np.logspace(-4,0,15)
    n_workers = 6 # for parallel processing
    results_dir = f'results/sklearn_tree_testset0.33_{constraint_name}_{epsilon}_maxdepth{max_depth}_probaccuracy'
    plot_savename = os.path.join(results_dir,f'gpa_dtree_vs_logreg_{constraint_name}_{epsilon}.png')
    verbose=True

    # Load spec
    specfile = f'specfiles/gpa_{constraint_name}_{epsilon}_fracsafety_{frac_data_in_safety}_sktree_maxdepth{max_depth}_reparam_spec.pkl'
    spec = load_pickle(specfile)
    spec.initial_solution_fn = initial_solution_fn
    os.makedirs(results_dir,exist_ok=True)

    # Reset spec dataset to only use 2/3 of the original data, 
    # use remaining 1/3 for ground truth set for the experiment
    # Use entire original dataset as ground truth for test set
    orig_features = spec.dataset.features
    orig_labels = spec.dataset.labels
    orig_sensitive_attrs = spec.dataset.sensitive_attrs
    # First, shuffle features
    (train_features,test_features,train_labels,
    test_labels,train_sensitive_attrs,
    test_sensitive_attrs
        ) = train_test_split(
            orig_features,
            orig_labels,
            orig_sensitive_attrs,
            shuffle=True,
            test_size=0.33,
            random_state=seed)

    new_dataset = SupervisedDataSet(
        features=train_features, 
        labels=train_labels,
        sensitive_attrs=train_sensitive_attrs, 
        num_datapoints=len(train_features),
        meta=spec.dataset.meta)

    # Set spec dataset to this new dataset
    spec.dataset = new_dataset
    # Setup performance evaluation function and kwargs 

    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        'performance_metric':performance_metric
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

    # # Baseline models
    
    if run_experiments:
        lr_baseline = BinaryLogisticRegressionBaseline()
        plot_generator.run_baseline_experiment(
            baseline_model=lr_baseline,verbose=True)

        dt_classifier = DecisionTreeClassifierBaseline(max_depth=max_depth)
        plot_generator.run_baseline_experiment(
            baseline_model=dt_classifier,verbose=True)
        
        # Seldonian experiment
        plot_generator.run_seldonian_experiment(verbose=verbose)


    if make_plots:
        plot_generator.make_plots(fontsize=12,legend_fontsize=12,
            performance_label=performance_metric,
            ncols_legend=2,
            include_legend=include_legend,
            model_label_dict=model_label_dict,
            save_format="png",
            savename=plot_savename if save_plot else None)

if __name__ == "__main__":
    main()

{% endhighlight python %}
<p>
    In addition to running this script for the five fairness constraints, we also separately re-ran the experiments in Tutorial E using the same setup as these experiments (i.e., same performance metric and test set), so that we could plot the SDTree model and the Seldonian logistic regressor model alongside each other. Once we ran all of these experiments, we produced the following five plots:
</p>

<div align="center">
    <figure>
        <img src="{{ "/assets/img/dtree_tutorial/gpa_dtree_vs_logreg_disparate_impact_0.8.png" | relative_url}}" class="img-fluid mt-4" style="width: 100%"  alt="Disparate impact"> 
        <img src="{{ "/assets/img/dtree_tutorial/gpa_dtree_vs_logreg_demographic_parity_0.2.png" | relative_url}}" class="img-fluid mt-2" style="width: 100%"  alt="Demograhpic parity"> 
        <img src="{{ "/assets/img/dtree_tutorial/gpa_dtree_vs_logreg_equalized_odds_0.35.png" | relative_url}}" class="img-fluid mt-2" style="width: 100%"  alt="Equalized odds"> 
        <img src="{{ "/assets/img/dtree_tutorial/gpa_dtree_vs_logreg_equal_opportunity_0.2.png" | relative_url}}" class="img-fluid mt-2" style="width: 100%"  alt="Equal opportunity"> 
        <img src="{{ "/assets/img/dtree_tutorial/gpa_dtree_vs_logreg_predictive_equality_0.2.png" | relative_url}}" class="img-fluid mt-2" style="width: 100%"  alt="Predictive equality"> 
        <figcaption align="left"> <b>Figure 2</b> - Experiment plots for the GPA prediction problem for five fairness constraints. From top to bottom: disparate impact, demographic parity, equalized odds, equal opportunity, and predictive equality. The Seldonian decision tree model (blue) is compared to the Seldonian logistic regressor (red) developed in Tutorial E and two baseline models trained without knowledge of the fairness constraints: i) a scikit-learn decision tree classifier (orange) and ii) a scikit-learn logistic regressor (green). The colored points and bands in each panel show the mean standard error over 20 trials. Each row of plots is an experiment for a different fairness constraint. From top to bottom: disparate impact, demographic parity, equalized odds, equal opportunity, predictive equality. The legend in the bottom panel applies to all five panels. </figcaption>
    </figure>
</div>  

<p>
    
</p>
</div>

