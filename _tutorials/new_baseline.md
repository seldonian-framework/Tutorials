---
layout: tutorial
permalink: /tutorials/new_baseline/
prev_url: /tutorials/efficient_deep_networks/
prev_page_name: (J) Efficiently training deep Seldonian networks
next_url: /tutorials/dtree_tutorial/
next_page_name: (L) Creating Fair Decision Trees and Random Forests
title: Seldonian \| Tutorial H
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Tutorial K: Creating a new baseline for (supervised) Seldonian Experiments </h2>
    <hr class="my-4">
    <h3> Contents </h3>
    <ul>
        <li> <a href="#intro">Introduction</a> </li>
        <li> <a href="#requirements">Requirements for a new baseline</a> </li>
        <li> <a href="#random_forest">Creating a random forest classifier baseline</a> </li>
        <li> <a href="#experiment">Using the random forest classifier baseline in an experiment</a> </li>
        <li> <a href="#summary">Summary</a> </li>
    </ul>
    <hr class="my-4">
    <h3 id="intro">Introduction</h3>
    <p>
        In a Seldonian Experiment, it is often useful to be able to compare the Seldonian model(s) you have trained to other machine learning models, which we refer to as "baselines" from here on out. These baseline models fall into two categories: i) "standard" ML models that are not fairness-aware and ii) fairness-aware (or safety aware) ML models. This tutorial tackles how to compare to baseline models of the first kind, i.e., standard ML models, in Seldonian Experiments. 
    </p>

    <p> 
        If you have gone through any of the previous tutorials which run Seldonian Experiments, such as the <a href="{{ "/tutorials/fair_loans_tutorial" | relative_url }}">Fair Loans Tutorial</a>, then you have probably noticed that the Seldonian Experiments library has some pre-defined baseline models already available. These existing baselines include models for logistic regression, linear regression, and a convolutional neural network implemented with PyTorch. The source code for these baselines can be found <a href="https://github.com/seldonian-toolkit/Experiments/tree/main/experiments/baselines">here</a>. We could not possibly include all models with which users might want to compare, so we designed the Experiments library to make it straightforward to add your own baseline models. 
    </p>

    <p>
        In this tutorial, we will demonstrate how a user might add a new baseline model for their specific  supervised learning problem. We will implement a random forest classifier baseline model for binary classification and show how to run an experiment comparing a Seldonian model to it. 
    </p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="requirements">Requirements for a new baseline</h3>

<p>
    Baselines are defined as Python classes. There are three requirements for defining a new baseline class. 
</p>    

<ol>
    <li>
        The class must define a <code class="codesnippet">model_name</code> string attribute in its <code class='codesnippet'>__init__()</code> method. Make this string unique and descriptive for each baseline. 
    </li>
    <li>
        The class must have a <code class='codesnippet'>train()</code> method that takes features and labels as input and trains the model. This method should return the trained model parameters, if there are any. If not (e.g., for non-parametric models), return None. If returning None, save the trained model as a class attribute so that it can be referenced in the <code class="codesnippet">predict()</code> method (see next requirement).
    </li>
    <li>
        The class must have a <code class='codesnippet'>predict()</code> method which takes the fitted model parameters (the return value of <code class='codesnippet'>train()</code>) and features as input and returns the predicted labels. For classification problems, the predictions should be probabilities for each class, as opposed to the integer class label.
    </li>
</ol>

<p>
<b>Note:</b> These requirements are similar to those for the underlying supervised models used in the Seldonian Engine, and it turns out that those models (without constraints) can be used as baselines with the simple addition of a <code class='codesnippet'>train()</code> method. One method of achieving this is to create a new class for the baseline which inherits from the Seldonian model and then add a <code class="codesnippet">train()</code> method. It is often useful to compare the model trained subject to constraints with the engine to the same model trained without constraints, and this trick allows you do that easily. For an example of how this method works, see the source code for the built-in logistic regression baseline model: <a href="https://github.com/seldonian-toolkit/Experiments/blob/8c191cae03dee042d0a397e2c1abec8c4be403d7/experiments/baselines/logistic_regression.py#L5">logistic_regression.py</a>. Note that the <code class="codesnippet">predict()</code> method is a parent method, so it is inheritied from the parent and not reiterated in the child class. 
</p>

</div>  

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="random_forest">Creating a random forest classifier baseline</h3>

<p>
    To create a class for a random forest classifier baseline, we will wrap <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">scikit-learn's random forest classifier</a> functionality to do all of the hard work of training and making predictions. Below is all of the code we need to implement this baseline. 
</p>

{% highlight python %}
# random_forest.py
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierBaseline():
    def __init__(self,n_estimators=100):
        """Implements a random forest classifier baseline for 
        binary classification
        :param n_estimators: The number of trees to use in the forest.
        """
        self.model_name = "random_forest"
        self.n_estimators = n_estimators

    def train(self,X,y):
        """Instantiate a new model instance and train (fit) 
        it to the training data, X,y.
        :param X: features
        :type X: 2D np.ndarray 
        :param y: Labels
        :type y: 1D np.ndarray
        """
        self.trained_model = RandomForestClassifier(n_estimators=self.n_estimators)
        self.trained_model.fit(X,y) # parent method
        return 

    def predict(self,theta,X):
        """Use the trained model to predict positive class probabilities.
        theta isn't used here because there are no fitted parameters 
        in random forests. 
        :param theta: Model weights, None in this case
        :param X: Test features
        :type X: 2D np.ndarray 
        """
        probs_bothclasses = self.trained_model.predict_proba(X)
        probs_posclass = probs_bothclasses[:,1]
        return probs_posclass

{% endhighlight python %}

<p>
    In <code class='codesnippet'>__init__()</code>, we define the model name, which fulfills requirement #1. <code class="codesnippet">__init__()</code> takes in one optional argument, <code class="codesnippet">n_estimators</code>, which is the number of trees we want to use in the random forest classifier. This argument was added as an example of how you would pass hyperparameters to the classifier. scikit-learn's classifiers have lots of hyperparameters, and in practice if you wanted the ability to specify more hyperparameters, you would simply add more arguments (or a dictionary of arguments) to <code class="codesnippet">__init__()</code>.  
</p>

<p>
    The <code class='codesnippet'>train()</code> method takes features <code class='codesnippet'>X</code> and labels <code class='codesnippet'>y</code>, trains the model and returns <code class="codesnippet">None</code>, fulfilling requirement #2. Looking into the body of this method, we make a class attribute <code class='codesnippet'>self.trained_model</code>, an instantiation of a scikit-learn <code class='codesnippet'>RandomForestClassifier</code> class, so that we can refer to it later in the <code class="codesnippet">predict()</code> method. Note that when we instantiate this class, we pass in the hyperparameter <code class="codesnippet">n_estimators</code> via the class attribute we set in <code class="codesnippet">__init__()</code>. Next, we train the model using the <code class='codesnippet'>fit()</code> method of the scikit-learn model. It is important that a new classifier is instantiated in each call to <code class="codesnippet">train()</code> because we want to make sure we train a new model in each experiment trial, instead of retraining the same model instance in each trial. 
</p>

<p>
     The <code class='codesnippet'>predict()</code> method takes the fitted model parameters <code class='codesnippet'>theta</code> and the test features <code class='codesnippet'>X</code> as input and returns the predicted class probabilities, fulfilling requirement #3. For random forest models, we do not have fitted model parameters, which is why <code class="codesnippet">train()</code> returned <code class="codesnippet">None</code>. As a result, <code class='codesnippet'>theta</code> will be passed into this method as <code class="codesnippet">None</code>, and notice that we don't use <code class="codesnippet">theta</code> anywhere in the body of the method. The reason that <code class='codesnippet'>theta</code> is a required input to <code class='codesnippet'>predict()</code> is because for other model types (such as logistic regression), the fitted parameters may be needed to make the predictions, and the experiments library does not know ahead of time whether your model is parametric or non-parametric.  
</p>

<p>
    Instead of using <code class='codesnippet'>theta</code> in <code class='codesnippet'>predict()</code>, we obtain the fitted model via the <code class='codesnippet'>self.trained_model</code> attribute. <code class='codesnippet'>train()</code> is run before <code class='codesnippet'>predict()</code> in each experiment trial, so <code class='codesnippet'>self.trained_model</code> will be set by the time <code class="codesnippet">predict()</code> is called. To make the class probability predictions, we use scikit-learn's <code class='codesnippet'>predict_proba()</code> method. For binary classification problems, <code class='codesnippet'>predict_proba()</code> returns a two dimensional array, where the first column is the probabilities of the 0th class and the second column is the probabilities of the 1st class. In the Seldonian Toolkit (both the Engine and the Experiments libraries), a convention we adopt for binary classification problems is that the <code class='codesnippet'>predict()</code> methods only return the probabilities of the 1st class (the "positive" class), since the 0th are redundant with these. This is why we include the last line: <code class='codesnippet'>probs_posclass = probs_bothclasses[:,1]</code>
</p>

</div>   

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="experiment">Using the random forest classifier baseline in an experiment</h3>

<p>
    Using this new baseline in an experiment is simple. Let's say we saved the baseline we created above to a file called <code>random_forest.py</code>. In our experiment script, all we need to do is import the baseline class and tell the plot generator to use an instance of this class. Below is a minimal example showing how to do this with the loans dataset described in the <a href="{{ "/tutorials/fair_loans_tutorial" | relative_url }}">Fair Loans Tutorial</a>. Our experiment will consist of 10 trials with 10 different data fractions ranging from 0.001 to 1.0. We will specify the <code class="codesnippet">n_estimator</code> hyperparameter to be different than the default. We will reference a specification file (contains the "spec" object) that we created in that tutorial, which can be found in this <a href="https://github.com/seldonian-toolkit/Experiments/tree/16a69cdaa48d598d2f396f4a86621975fc4afdb4/examples/loans/data/spec">directory</a>. 
</p>

<p>
    Below is the code to run the experiment. 
</p>

{% highlight python %}
# experiment_with_random_forest_baseline.py
import os
import numpy as np 

from seldonian.utils.io_utils import load_pickle
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.perf_eval_funcs import binary_logistic_loss
from experiments.baselines.logistic_regression import BinaryLogisticRegressionBaseline
from experiments.baselines.random_forest import RandomForestClassifierBaseline

if __name__ == "__main__":
    run_experiments = True
    make_plots = True
    model_label_dict = {
        'qsa':'Seldonian model',
        'random_forest': 'Random forest (no constraints)',
        }
    n_trials = 10
    data_fracs = np.logspace(-3,0,10)
    n_workers = 6
    verbose=False
    results_dir = f'results/loans_random_forest'
    os.makedirs(results_dir,exist_ok=True)

    plot_savename = os.path.join(results_dir,"loans_random_forest.pdf")

    # Load spec
    specfile = f'./data/spec/loans_disparate_impact_0.9_spec.pkl'
    spec = load_pickle(specfile)

    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset

    perf_eval_kwargs = {
        'X':dataset.features,
        'y':dataset.labels,
        }

    plot_generator = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='resample',
        perf_eval_fn=binary_logistic_loss,
        constraint_eval_fns=[],
        results_dir=results_dir,
        perf_eval_kwargs=perf_eval_kwargs,
        )

    if run_experiments:
        # Use our new baseline model
        rf_baseline = RandomForestClassifierBaseline(n_estimators=50)
        plot_generator.run_baseline_experiment(
            baseline_model=rf_baseline,verbose=verbose)

        # Seldonian experiment
        plot_generator.run_seldonian_experiment(verbose=verbose)

    if make_plots:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label="Log loss",
                performance_yscale='log',
                model_label_dict=model_label_dict,
                savename=plot_savename,
            )

{% endhighlight python %}

<p>
    If we save the above code in a file called <code>experiment_with_random_forest_baseline.py</code>, and run it from the command line, it will generate a plot file called <code>results/loans_random_forest/loans_random_forest.pdf</code>, which should look like this:
</p>

<div align="center mb-4">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/new_baseline_tutorial/loans_random_forest.png" | relative_url }}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Loan experiment with random forest baseline">
    </figure> 
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="summary">Summary</h3>

In this tutorial, we demonstrated how to create a new baseline for Seldonian Experiments. We described the general requirements for implementing a new baseline and provided an example implementation in a Seldonian Experiment using the dataset from the Fair Loans Tutorial. Our implementation wrapped scikit-learn's random forest classifier, and the method we used can be used to implement any scikit-learn supervised ML model as a baseline in a Seldonian Experiment. Baselines using other Python librarires can similarly be implemented using the patterns described in this tutorial. If you develop a baseline that you think could be generally useful to others, feel free to submit a pull request on the <a href="https://github.com/seldonian-toolkit/Experiments/pulls">Experiments library Github page</a>.

</div>

