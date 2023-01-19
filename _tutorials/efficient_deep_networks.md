---
layout: tutorial
permalink: /tutorials/efficient_deep_networks/
prev_url: /tutorials/custom_base_variable_tutorial/
prev_page_name: (I) Creating custom base variables in behavioral constraints
---
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Tutorial J: Efficiently training deep Seldonian networks</h2>
    <hr class="my-4" />
    <h3> Contents </h3>
    <ul>
        <li> <a href="#intro">Introduction</a> </li>
        <li> <a href="#dataset_prep">Dataset preparation</a> </li>
        <li> <a href="#formulate">Formulate the Seldonian ML problem</a> </li>
        <li> <a href="#spec_object">Creating the specification object</a></li>
        <li> <a href="#experiments">Running a Seldonian experiment</a></li>
        <li> <a href="#discussion">Discussion</a></li>
        <li> <a href="#summary">Summary</a></li>
        <li> <a href="#references">References</a></li>
        
    </ul>
    <hr class="my-4">
    <h3 id="intro">Introduction</h3>
    <p>
    	Modern deep networks can have billions of trainable parameters, making them computationally expensive to train. Furthermore, hyperparameter tuning usually involves re-training the model many times with different hyperparameters, making it a potential bottleneck in machine learning workflows. In Seldonian algorithms, the addition of constraints increases the computational burden of model training. The number of constraints and the complexity of each constraint will determine how much additional compute is necessary to perform <a href="{{ "/tutorials/alg_details_tutorial/#candidate_selection" | relative_url }}">candidate selection</a>. The toolkit can leverage GPUs when the model is written in PyTorch or Tensorflow to mitigate the computational burden (see <a href="{{ "/tutorials/pytorch_mnist" | relative_url }}">Tutorial G: Creating your first Seldonian PyTorch model</a>, for example). However, training large models subject to constraints is an unavoidably expensive computation.
    </p>
    <p>
        The computational burden of candidate selection can be vastly minimized if we simplify the model we need to train subject to the constraints. We hypothesize that not all of the hidden layers of a deep learning model need to be trained with knowledge of the constraint. Early layers of a deep network generally learn basic features, and as depth increases so does the complexity of the features that can be learned. For example, in a convolutional neural network learning to distinguish between cats and dogs, the first few layers might learn basic edge filters, and the deeper layers might learn more complex features such as head and whisker shapes. The final few layers contain the task-specific information. For many types of constraints, it is unlikely that the parameters of the early layers, such as the shape detectors, need to be adjusted to accommodate the constraints. We conjecture that <b>only the final layer(s) need to be trained in the Seldonian algorithm</b>, and the rest of the network can be trained in the conventional way without constraints. 
    </p>
    <p>
        In this tutorial, we formalize the process described above for efficiently training deep Seldonian algorithms using the toolkit. We apply this method to the <a href="{{ "/examples/facial_recognition/" | relative_url }}">Gender bias in facial recognition example</a>. We find that in a fraction of the runtime required to create a Seldonian algorithm using the full deep network, we can use a smaller model to achieve equal performance and fairness. 
    </p>
	</div>


<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="outline" align="center" class="mb-3">Outline of method</h3>
<p>
    Here we sketch an outline of the procedure to create an efficient deep Seldonian network by reducing the size of the model that is used in the Seldonian algorithm. The first few steps are to be performed outside of the Seldonian Toolkit. The model must be implemented with PyTorch or Tensorflow. 
</p>

<ol>
    <u>Outside of the Seldonian Toolkit:</u>
    <li>Randomize your data, then split it into two datasets. The optimal split will depend on your specific problem, but we suggest starting with a 50/50 split. </li>
    <li>Train the full network on one of these sets using your favorite training method, e.g., <a href="https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop">PyTorch</a>. There is no need to include any information about behavioral constraints in this training process. Save the weights of the trained model so that you can access them again later.</li>
    <li> Separate out the "body" and the "head" of the full network into two separate models, a "body-only" model and a "head-only" model. In this tutorial, the head of the network refers to the final fully connected layer (the output layer) of the network. In general, one could split the network anywhere.  </li> 
    <li>Assign the weights from the trained full network to the new body-only model so that this body-only model is "trained." Or, simply remove the head of the network so that you are left only with the trained body.</li>
    <li>Pass <i>all</i> of the data (both datasets from step 1) through the trained "body-only" model. Save the outputs of this model. These are your new "latent features" that you will use as input to the Seldonian Toolkit.</li>
    <br>
    <u>Using the Seldonian Toolkit:</u>
    <li>The head-only model is the model you will use in the toolkit. This head-only model should be initially untrained when used in the toolkit, so don't apply the weights learned in step 2 to the head.</li>
    <li>Assign <code class="highlight">frac_data_in_safety</code> in the spec object to be the same split fraction as you used in step 1. It is important that the latent features that you put into the candidate dataset in the toolkit come from the dataset that you used to train the full model in step 2. In other words, <b>no safety data can come from the dataset that was used to train the full model in step 2</b> because that would invalidate the safety/fairness guarantees. </li>
    <li>Run the Seldonian engine/experiments as normal, except now your model is a simple linear model instead of a deep network. It should be <b>much</b> faster than using the full model as the Seldonian model, especially if your initial network is large. </li> 
</ol>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="dataset_prep" align="center" class="mb-3">Dataset preparation</h3>

<p>
    We will train the gender classifier on the UTKFace dataset<sup><a href="#fn16" id="ref16">16</a></sup>, a public dataset consisting of over 20,000 faces with annotations for gender, race, and age. The <a href="https://susanqq.github.io/UTKFace/">UTKFace</a> webpage hosts the official dataset. The images vary in pose, facial expression, illumination, occlusion, and resolution. Like many large supervised learning datasets, the official dataset contains some mislabeling, and some images have letters and watermarks. We downloaded an aligned and cropped version of the dataset containing 23705 faces in CSV format from <a href="https://www.kaggle.com/code/erdal002/cnn-keras-and-pytorch-gender-classification/data">here</a>.
</p>

<div align="center">
    <figure class='mt-4'>
            <img src="{{ "/assets/img/gender_classifier/UTKFace_gender.png" | relative_url }}" class="img-fluid mx-auto inline-block rounded shadow p-3 bg-white" style="width: 33%"  alt="">
            <img src="{{ "/assets/img/gender_classifier/UTKFace_race.png" | relative_url }}" class="img-fluid mx-auto inline-block rounded shadow p-3 bg-white" style="width: 33%"  alt="">
            <img src="{{ "/assets/img/gender_classifier/UTKFace_age.png" | relative_url }}" class="img-fluid mx-auto inline-block rounded shadow p-3 bg-white" style="width: 33%"  alt="">
    </figure> 
    <figcaption><b>Figure 1</b>: Distribution of gender, race, and age attributes in the UTKFace dataset.   </figcaption>
</div>

<div align="center" class='my-4'>
    <figure class='my-4'>
            <img src="{{ "/assets/img/gender_classifier/UTKFace_examples.png" | relative_url }}" class="img-fluid mx-auto  d-block rounded shadow p-3 bg-white" style="width: 66%"  alt="">
    </figure> 
    <figcaption><b>Figure 2</b>: Ten random samples from the UTKFace dataset, demonstrating the diversity in personal attributes as well as image quality. </figcaption>
</div>

<p>

     We used the following code to load, normalize, and shuffle the data. We then extracted the features, labels, and sensitive attributes. We used only the images for the features. The labels are the binary gender values (0=male, 1=female). The sensitive attributes are also the gender values. The toolkit requires any sensitive attributes to be one-hot encoded, so we make binary valued columns for gender - one where 1 indicates male and the other where 1 indicates female. We also clip off the last 5 data points (after shuffling) so that it will be easy to make all batches the same size when running gradient descent. 
</p>
    {% highlight python %}
    # format_data.py
    import autograd.numpy as np   # Thinly-wrapped version of Numpy
    import pandas as pd
    import os

    from seldonian.utils.io_utils import load_pickle,save_pickle

    
    N=23700 # Clips off 5 samples (at random) to make total divisible by 150,
    # the desired batch size
    
    savename_features = './features.pkl'
    savename_labels = './labels.pkl'
    savename_sensitive_attrs = './sensitive_attrs.pkl'

    print("loading data...")
    data = pd.read_csv('../../../facial_recognition/Kaggle_UTKFace/age_gender.csv')
    # Shuffle data since it is in order of age, then gender
    data = data.sample(n=len(data),random_state=42).iloc[:N]
    # Convert pixels from string to numpy array
    print("Converting pixels to array...")
    data['pixels']=data['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))

    # normalize pixels data
    print("Normalizing and reshaping pixel data...")
    data['pixels'] = data['pixels'].apply(lambda x: x/255)

    # Reshape pixels array
    X = np.array(data['pixels'].tolist())

    ## Converting pixels from 1D to 4D
    features = X.reshape(X.shape[0],1,48,48)

    # Extract gender labels
    labels = data['gender'].values

    # Make one-hot sensitive feature columns
    M=data['gender'].values
    mask=~(M.astype("bool"))
    F=mask.astype('int64')
    sensitive_attrs = np.hstack((M.reshape(-1,1),F.reshape(-1,1)))

    # Save to pickle files
    print("Saving features, labels, and sensitive_attrs to pickle files")
    save_pickle(savename_features,features)
    save_pickle(savename_labels,labels)
    save_pickle(savename_sensitive_attrs,sensitive_attrs)
    
    {% endhighlight python %}

<p>
    The preprocessing steps in the above code take a minute to run. We saved the features, labels and sensitive attributes to disk so that we can load these quantities from disk later, rather than recomputing them. 
</p>

</div>
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="formulate" align="center" class="mb-3">Formulate the Seldonian ML problem</h3>
<p>
	We consider the following definition of fairness, which ensures that the accuracies for male faces and female faces are within $20\%$ of each other. The constraint string for this definition is: 
</p>

<p style='margin-left: 30px;'>
$\min((\mathrm{ACC} | [\mathrm{M}])/(\mathrm{ACC} | [\mathrm{F}]),(\mathrm{ACC} | [\mathrm{F}])/(\mathrm{ACC} | [\mathrm{M}])) \geq 0.8$, 
</p>

<p>
where $\mathrm{ACC}$ is the <a href="{{ "/glossary/#measure_function" | relative_url }}">measure function</a> for accuracy, and $\mathrm{M}$ and $\mathrm{F}$ refer to the male and female sensitive attributes, respectively. We will enforce this constraint with a confidence level of $\delta=0.05$, i.e., we want to ensure it holds with $95\%$ confidence.
</p>

<p>
<b>Note</b>: We stress that this definition of fairness is just an example. By using it in this example, we are not suggesting that it is a <i>correct</i> fairness definition. Sociologists and legislators are responsible for determining the proper fairness definitions for FRTs.
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="spec_object" align="center" class="mb-3">Running the Seldonian Engine</h3>
<p>
The engine requires a spec object to run. The spec object takes as input a model, dataset, and parse trees, among other parameters. We will use a convolutional neural network (CNN) with four convolutional layers, implemented using PyTorch. For details on how to use PyTorch models in the toolkit, see <a href="{{ "/tutorials/pytorch_mnist" | relative_url }}">Tutorial G: Creating your first Seldonian PyTorch model</a>. The code for the model is shown below, which we save in a file called <code>facial_recog_cnn.py</code> in a folder where we plan to run the Seldonian Engine:
</p>

{% highlight python %}
# facial_recog_cnn.py
from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
import torch.nn as nn
import torch

class FacialRecogCNNModel(nn.Module):
    def __init__(self):
        super(FacialRecogCNNModel, self).__init__()
        # Define all layers here
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.Batch1=nn.BatchNorm2d(16)
        self.Batch2=nn.BatchNorm2d(32)
        self.Batch3=nn.BatchNorm2d(64)
        self.Batch4=nn.BatchNorm2d(128)
        
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(128 * 1 * 1, 128) 
        self.fc2=nn.Linear(128,256)
        self.fc3=nn.Linear(256,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Call all layers here. This does the forward pass.
        out = self.cnn1(x) 
        out = self.relu(out)
        out = self.maxpool(out)
        out=self.Batch1(out)

        out = self.cnn2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out=self.Batch2(out)
        
        out = self.cnn3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out=self.Batch3(out)
        
        out = self.cnn4(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out=self.Batch4(out)
        
        # Resize
        # Original size: (100, 32, 7, 7)
        # New out size: (100, 32*7*7)
        out = torch.flatten(out,start_dim=1)

        # Linear functions (readout)
        out = self.fc1(out)
        out=self.fc2(out)
        out=self.fc3(out)

        # Softmax to make probabilities
        out=self.softmax(out)[:,1] 

        return out

class PytorchFacialRecog(SupervisedPytorchBaseModel):
    def __init__(self,device):
        """ Implements a CNN with PyTorch. 
        CNN consists of four hidden layers followed 
        by a linear + softmax output layer.

        Inputs are N,1,48,48 where N is the number of them,
        1 channel and 48x48 pixels.
        """
        super().__init__(device)

    def create_model(self,**kwargs):
        """ Create the pytorch model and return it
        """
        return FacialRecogCNNModel()
{% endhighlight python %}

<p>
 The model class <code class="highlight">PytorchFacialRecog</code> is the Seldonian model, which is just a wrapper for the PyTorch model class <code class="highlight">FacialRecogCNNModel</code>. 
</p>

<p>
We will use this model, along with a dataset and parse trees to create the full specification object. We also specify the optimization strategy, mini-batch gradient descent with a batch size of 237 and 40 epochs. The candidate data have $23700*0.5 = 11850$ datapoints, so all 50 batches will be of equal size with a batch size of 237. We also define the primary objective function to be the binary logistic loss. We create the spec object and then run the engine in a file called <code>run_engine.py</code>. Note that in this file, we import the model class we just defined from the file above. The model file must be in the same folder as this script in order for the import to work. 
</p>
{% highlight python %}
# run_engine.py
from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from facial_recog_cnn import PytorchFacialRecog
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)


N=23700
print("Loading features,labels,sensitive_attrs from file...")
features = load_pickle(savename_features)
labels = load_pickle(savename_labels)
sensitive_attrs = load_pickle(savename_sensitive_attrs)

assert len(features) == N
assert len(labels) == N
assert len(sensitive_attrs) == N
frac_data_in_safety = 0.5
sensitive_col_names = ['M','F']

meta_information = {}
meta_information['feature_col_names'] = ['img']
meta_information['label_col_names'] = ['label']
meta_information['sensitive_col_names'] = sensitive_col_names
meta_information['sub_regime'] = sub_regime
print("Making SupervisedDataSet...")
dataset = SupervisedDataSet(
    features=features,
    labels=labels,
    sensitive_attrs=sensitive_attrs,
    num_datapoints=N,
    meta_information=meta_information)

constraint_strs = ['min((ACC | [M])/(ACC | [F]),(ACC | [F])/(ACC | [M])) >= 0.8']
deltas = [0.05] 
print("Making parse trees for constraint(s):")
print(constraint_strs," with deltas: ", deltas)
parse_trees = make_parse_trees_from_constraints(
    constraint_strs,deltas,regime=regime,
    sub_regime=sub_regime,columns=sensitive_col_names)

# Put on Mac M1 GPU via Metal performance shader (MPS) device.
# For NVIDIA graphics cards use "cuda" as the device string.
device = torch.device("mps")
model = PytorchFacialRecog(device)

initial_solution_fn = model.get_model_params
spec = SupervisedSpec(
    dataset=dataset,
    model=model,
    parse_trees=parse_trees,
    frac_data_in_safety=frac_data_in_safety,
    primary_objective=objectives.binary_logistic_loss,
    use_builtin_primary_gradient_fn=False,
    sub_regime=sub_regime,
    initial_solution_fn=initial_solution_fn,
    optimization_technique='gradient_descent',
    optimizer='adam',
    optimization_hyperparams={
        'lambda_init'   : np.array([0.5]),
        'alpha_theta'   : 0.001,
        'alpha_lamb'    : 0.001,
        'beta_velocity' : 0.9,
        'beta_rmsprop'  : 0.95,
        'use_batches'   : True,
        'batch_size'    : 237,
        'n_epochs'      : 40,
        'gradient_library': "autograd",
        'hyper_search'  : None,
        'verbose'       : True,
    },
    
    batch_size_safety=2000
)
save_pickle('./spec.pkl',spec,verbose=True)
SA = SeldonianAlgorithm(spec)
passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
{% endhighlight python %}

<p>
    If we run the above file, we find that the safety test passes. The gradient descent log is written out, and if we plot it we can see how the primary objective function and evolved during the optimization process.
</p>

<div align="center">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/gender_classifier/gender_classifier_cs.png" | relative_url }}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Gradient descent">
    </figure> 
    <figcaption><b>Figure 2</b>: Gradient descent on the gender classification problem. The log loss (left) initially increases because the solution is predicted to fail the safety test (middle right). Solutions are eventually found that have both low log loss and are predicted to pass the safety test. The optimal solution is shown by the dotted black lines, i.e., the solution with the lowest log loss that is predicted to pass the safety test.   </figcaption>
</div>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="experiments" name="experiments" align="center" class="mb-3">Running a Seldonian Experiment</h3>

<p>
	We run a Seldonian Experiment, comparing the quasi-Seldonian model to two baseline models. The first baseline is the same CNN without the constraint, and the second is a random classifier that predicts that an image is of a man with $p=0.5$, regardless of input. The performance metric is accuracy. We use 10 trials and a data fraction array of: <code class="highlight">[0.001,0.005,0.01,0.1,0.33,0.66,1.0]</code>. The full code to run the experiment is below:
</p>

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">
{% highlight python %}
# run_experiment.py
import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score

import torch

if __name__ == "__main__":
    # Parameter setup
    run_experiments = True
    make_plots = True
    save_plot = True
    include_legend = True
    performance_metric = 'Accuracy'
    model_label_dict = {
        'qsa':'quasi-Seldonian CNN',
        'facial_recog_cnn': 'CNN (no constraints)',
        'random_classifier': 'Random classifier'}
    n_trials = 10
    data_fracs = [0.001,0.005,0.01,0.1,0.33,0.66,1.0]
    
    batch_epoch_dict = {
        0.001:[24,50],
        0.005:[119,50],
        0.01:[237,75],
        0.1:[237,30],
        0.33:[237,20],
        0.66:[237,10],
        1.0: [237,10]
    }
    
    n_workers = 1
    results_dir = f'../../results/facial_recog_2022Dec19'
    plot_savename = os.path.join(results_dir,f'facial_recog_{performance_metric}.png')

    verbose=True

    # Load spec
    specfile = './spec.pkl'
    spec = load_pickle(specfile)

    os.makedirs(results_dir,exist_ok=True)

    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset
    test_features = dataset.features
    test_labels = dataset.labels

    # Setup performance evaluation function and kwargs 
    # of the performance evaluation function

    def perf_eval_fn(y_pred,y,**kwargs):
        if performance_metric == 'log_loss':
            return log_loss(y,y_pred)
        elif performance_metric == 'Accuracy':
            return accuracy_score(y,y_pred > 0.5)
    device = spec.model.device
    
    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        'device':device,
        'eval_batch_size':2000
        }

    constraint_eval_kwargs = {
        'eval_batch_size':2000
        }

    plot_generator = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='resample',
        perf_eval_fn=perf_eval_fn,
        constraint_eval_fns=[],
        constraint_eval_kwargs=constraint_eval_kwargs,
        results_dir=results_dir,
        perf_eval_kwargs=perf_eval_kwargs,
        batch_epoch_dict=batch_epoch_dict,
        )

    # # Baseline models
    if run_experiments:
        plot_generator.run_baseline_experiment(
            model_name='random_classifier',verbose=True)

        plot_generator.run_baseline_experiment(
            model_name='facial_recog_cnn',verbose=True)

        Seldonian experiment
        plot_generator.run_seldonian_experiment(verbose=verbose)


    if make_plots:
        plot_generator.make_plots(
            model_label_dict=model_label_dict,
            fontsize=12,legend_fontsize=8,
            performance_label=performance_metric,
            include_legend=include_legend,
            savename=plot_savename if save_plot else None)
{% endhighlight python %}
</div>
<p>
    When running the above code, the file with the model <code>facial_recog_cnn.py</code> must be in the same folder. Running the above code generates the following plot:
</p>
<div align="center">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/gender_classifier/facial_recog_accuracy.png" | relative_url }}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Gender classifier experiment">
    </figure> 
    <figcaption><b>Figure 3</b>: A Seldonian Experiment subject to the constraint, $g$, shown at the top of the figure. The three panels are accuracy (left), probability of solution (middle), and probability that the constraint was violated (right). In each panel, the mean (points) and standard error (uncertainty bands) over 10 trials are shown. We compare the CNN learned with a quasi-Seldonian algorithm (QSA, blue) to the same CNN learned with standard gradient descent and no constraint (orange) and a random classifier (green). </figcaption>
</div>
</div>


<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="discussion" align="center" class="mb-3">Discussion</h3>

<p>
The experiment plots show that if we train the CNN in the standard way without using any constraints, it violates the fairness constraint frequently (right panel of Figure 3). The fairness constraint is not all that restrictive - it only requires that the accuracies between males and females be within $20\%$ of each other. The CNN without the constraint is therefore quite biased in terms of its accuracy for faces of different genders. Using the Seldonian toolkit to create a quasi-Seldonian CNN trained subject to the fairness constraint, we achieve a model that never violates the constraint. The quasi-Seldonian model takes around 5000 points before it will return a solution every time. With less data, the algorithm returns "No Solution Found," meaning that it was unable to find a safe solution. The accuracy of the quasi-Seldonian CNN approaches the accuracy of the CNN without the constraint, with only a difference of $\sim4\%$. We stress that we did not perform rigorous hyperparameter tuning to achieve this result. It is likely that this gap could be narrowed further with tuning during candidate selection. We included the random classifier to show that baseline performance. The random classifier never violates the constraint because it has the same prediction function regardless of input. 
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="summary" align="center" class="mb-3">Summary</h3>

<p>
In this example, we demonstrated how to use the Seldonian Toolkit to build a gender classifier that satisfies a custom-defined fairness constraint. We covered how to format the dataset and metadata so that they are compatible with the toolkit. We ran the Seldonian Engine once to show that the model is capable of passing the safety test and to understand how the model is trading off its primary loss function with the constraint function. We then ran a Seldonian Experiment to evaluate the true performance and safety of the QSA and to compare it to baseline models. We found that the QSA performs almost as well as the CNN without any constraints after it has enough data to return a solution. Critically, the QSA never violates the constraint, whereas the CNN without constraints frequently violates it. The constraint is not all that restrictive, only requiring that the model's accuracy should not differ by more than $20\%$ when predicting male versus female faces. 
</p>

<p>
    We want to stress that the dataset, model and fairness constraint we used in this example are all interchangable. The toolkit supports any supervised PyTorch model, as long as it is differentiable. Likewise, the fairness constraint we used is not intended to be <i>correct</i> or reused elsewhere. It was chosen as an example of how one could define their own fairness constraint or to adopt an existing one. One could even define multiple fairness constraints that they want to hold simultaneously. We hope this example helps to illustrate how to use to the Seldonian Toolkit to apply fairness constraints to a deep learning model and to evaluate how the performance, data-efficiency, and safety trade off. 
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="summary" align="center" class="mb-3">References</h3>

<p id="fn1">1. <a href="https://pages.nist.gov/frvt/reports/1N/frvt_1N_report.pdf">Face Recognition
Vendor Test (FRVT)
Part 2: Identification</a> <a href="#ref1" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn2">2. <a href="https://www.thehindu.com/news/cities/bangalore/face-recognition-technology-helps-find-missing-woman/article36372677.ece">Face-recognition technology helps find missing woman despite mask</a> <a href="#ref2" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn3">3. <a href="https://www.nytimes.com/2020/02/07/business/clearview-facial-recognition-child-sexual-abuse.html">Clearview’s Facial Recognition App Is Identifying Child Victims of Abuse</a> <a href="#ref3" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn4">4. <a href="https://www.gao.gov/products/gao-22-106154#:~:text=U.S.%20Customs%20and%20Border%20Protection,for%20travelers%20entering%20the%20country.">Facial Recognition Technology:
CBP Traveler Identity Verification and Efforts to Address Privacy Issues</a> <a href="#ref4" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn5">5. <a href="https://www.nyc.gov/site/nypd/about/about-nypd/equipment-tech/facial-recognition.page">NYPD Questions and Answers
Facial Recognition</a> <a href="#ref5" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn6">6. <a href="https://www.nytimes.com/2020/01/12/technology/facial-recognition-police.html">How the Police Use Facial Recognition, and Where It Falls Short</a> <a href="#ref6" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn7">7. <a href="https://www.nytimes.com/2016/06/26/opinion/sunday/artificial-intelligences-white-guy-problem.html">Artificial Intelligence’s White Guy Problem</a> <a href="#ref7" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn8">8. <a href="http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf">Gender Shades: Intersectional Accuracy Disparities in
Commercial Gender Classification</a> <a href="#ref8" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn9">9. <a href="https://www.ibm.com/policy/facial-recognition-sunset-racial-justice-reforms/">IBM CEO's Letter to Congress on Racial Justice Reform</a> <a href="#ref9" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn10">10. <a href="https://www.reuters.com/technology/exclusive-amazon-extends-moratorium-police-use-facial-recognition-software-2021-05-18/">Amazon extends moratorium on police use of facial recognition software</a> <a href="#ref10" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn11">11. <a href="https://www.nytimes.com/2021/11/02/technology/facebook-facial-recognition.html">Facebook, Citing Societal Concerns, Plans to Shut Down Facial Recognition System</a> <a href="#ref11" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn12">12. <a href="https://www.banfacialrecognition.com/map/">BAN FACIAL RECOGNITION</a> <a href="#ref12" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn13">13. <a href="https://chalearnlap.cvc.uab.cat/challenge/38/description/">2020 Looking at People Fair Face Recognition challenge ECCV</a> <a href="#ref13" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn14">14. <a href="https://microsoft.github.io/DigiFace1M/">DigiFace-1M: 1 Million Digital Face Images for Face Recognition</a> <a href="#ref14" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn15">15. <a href="https://openaccess.thecvf.com/content_WACV_2020/papers/Albiero_Does_Face_Recognition_Accuracy_Get_Better_With_Age_Deep_Face_WACV_2020_paper.pdf">Does Face Recognition Accuracy Get Better With Age?
Deep Face Matchers Say No</a> <a href="#ref15" title="Jump back to footnote in the text.">↩</a></p>

<p id="fn16">16. <a href="https://susanqq.github.io/UTKFace/">UTKFace
Large Scale Face Dataset</a> <a href="#ref16" title="Jump back to footnote in the text.">↩</a></p>

</div>