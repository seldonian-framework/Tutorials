---
layout: tutorial
permalink: /tutorials/pytorch_model/
prev_url: /tutorials/gridworld_RL_tutorial/
prev_page_name: (G) Reinforcement learning first tutorial
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Tutorial H: PyTorch Deep learning with the toolkit (with GPU acceleration) </h2>
    <hr class="my-4">

    

    <h3 id="intro">Introduction</h3>
    <p>
        Deep learning is a cornerstone of modern machine learning. It is being used in nearly every industry, and it is increasingly affecting our lives. Due to the complexity of some deep learning models (e.g., <a href="https://deepai.org/machine-learning-glossary-and-terms/hidden-layer-machine-learning">hidden layers</a>), they can be opaque to the people applying them. This makes it especially important to be able to put constraints on their behavior. 
    </p>
    <p> 
        <a href="https://pytorch.org/">PyTorch</a> is an open-source machine learning framework written in Python that is extremely popular for deep learning. PyTorch uses objects called tensors (<code class="highlight">torch.Tensor</code>) to represent multi-dimensional arrays of data. The Seldonian toolkit uses NumPy arrays to represent data and as a whole is not written in PyTorch. Fortunately, PyTorch tensors can be converted to NumPy arrays with relative ease, and vice versa. The toolkit takes care of this hand-off, allowing developers to focus on model implementation. 
    </p>

    <p>
       <b>Note 1:</b> Currently, PyTorch is only supported for supervised learning models. We plan to support deep reinforcement learning models in PyTorch in the near future. 
    </p>

    <p>
        <b>Note 2:</b> In the toolkit, PyTorch is only used to define the model and how to do the forward and backward pass. PyTorch is <i>not</i> being used to perform the entire gradient descent process (although in principle it could).
    </p>

    <p>
        <b>Note 3:</b> To enable GPU acceleration (see below), you will need to have the proper setup for your machine. For example, CUDA needs to be enabled if you have an NVIDIA graphics card. The good news is that even if you don't have access to a GPU, you can still easily run the model on your CPU. To learn more about GPU-acceleration for PyTorch, see the "Install Pytorch" section of this page: <a href="https://pytorch.org/">https://pytorch.org/</a>
    </p>


    <h3>Outline</h3> <code class="highlight"></code>
    <p>In this tutorial, you will learn how to:
    <ul>
        <li>Build and run a GPU-accelerated convolutional neural network (CNN) on the MNIST database of handwritten digits with a simple safety constraint. </li>
        <li>Create your own deep Seldonian supervised machine learning models with PyTorch using the PyTorch model base class.</li>
    </ul>
    </p> 

<h3> Implementing a convolutional neural network with PyTorch </h3>

<p>
    In this section, we will build a CNN in PyTorch that is compatible with the Seldonian Toolkit. First, make sure you have the latest version of the libraries in the toolkit installed. </p>    
{% highlight javascript %}
$ pip install --upgrade seldonian-engine
$ pip install --upgrade seldonian-experiments
{% endhighlight javascript %}

<p>
    It is important to make a clear distinction when referring to "models" throughout this tutorial. We will use the term "Seldonian model" to refer to the highest level model abstraction in the toolkit. The Seldonian model is the thing that communicates with the rest of the toolkit. The Seldonian model we will build in this tutorial consists of a "PyTorch model," a term which we will use to refer to the actual PyTorch implementation of a neural network (or some other group of architectures). The PyTorch model does communicate with the other pieces of the Seldonian Toolkit, whereas the Seldonian model does. 
</p>
<p>
    There are three requirements for creating a new Seldonian model that implements a Pytorch model:
    <ol>
        <li>The Seldonian model class must inherit from the base class: <code class="highlight">seldonian.models.pytorch_model.SupervisedPytorchBaseModel</code>. </li>
        <li>The Seldonian model class must take as input a <code class="highlight">device</code> string, which specifies the hardware (e.g. CPU vs. GPU) on which to run the model.</li>
        <li>The Seldonian model class must have a <code class="highlight">create_model()</code> method in which it defines the PyTorch model and returns it as an instance of <code class="highlight">torch.nn.Module</code> or <code class="highlight">torch.nn.Sequential</code> - the PyTorch model must have a <code class="highlight">forward()</code> method. </li>
    </ol> 
    We will name our new model <code class="highlight">PyTorchCNN</code> and use the network architecture described in this article: <a href="https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118">https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118</a>. Here is the full Seldonian PyTorch model that meets the three requirements: 
</p>
{% highlight python %}
from .pytorch_model import SupervisedPytorchBaseModel
import torch.nn as nn

class PytorchCNN(SupervisedPytorchBaseModel):
    def __init__(self,device):
        """ Implements a CNN with PyTorch. 
        CNN consists of two hidden layers followed 
        by a linear + softmax output layer 

        :param input_dim: Number of features
        :param output_dim: Size of output layer (number of label columns)
        """
        super().__init__(device)

    def create_model(self,**kwargs):
        """ Create the pytorch model and return it
        Inputs are N,1,28,28 where N is the number of them,
        1 channel and 28x28 pixels.
        Do Conv2d,ReLU,maxpool twice then
        output in a fully connected layer to 10 output classes
        and softmax outputs to get probabilities.
        """
        cnn = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),   
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
            nn.Softmax(dim=1)
        )       
        return cnn
{% endhighlight python %}
<p>
    It is very important that the <code class="highlight">create_model()</code> method returns the PyTorch model object. In this case, we used PyTorch's <a href="https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html">sequential</a> container to hold a sequence of Pytorch <a href="https://pytorch.org/docs/stable/nn.html">nn</a> layers. This is just one pattern that PyTorch provides and it is not required in the <code class="highlight">create_model()</code> method. As long as the method returns the outputs of the model after passing the inputs through each layer, this method will work. We applied a softmax layer at the end of the model so that the outputs of the model are probabilities. We did this because the primary objective function we will use when we run the model in the following example expects probabilities. 
</p> 

<p>
    At this point, this model is ready to use. As you can see, most of the work is done for you in the base class, whose methods are designed to work regardless of the model architecture. It is possible that you will want or need to override some of the base class methods in some cases, however. 
</p>
     
<h3> MNIST with a safety constraint </h3>

<p> The Modified National Institute of Standards and Technology <a href="https://en.wikipedia.org/wiki/MNIST_database">(MNIST)</a> database contains 60,000 training images and 10,000 testing images of handwritten digits 0-9. It is a commonly used dataset in machine learning research. There are models that are vastly superior to the simple CNN we built in terms of accuracy (equivalently, error rate), but that is not the point. Our objective in this tutorial is to show that it is possible to enforce a safety constraint in a deep learning setting. 
</p>

<h5>Formulate the Seldonian ML problem</h5>
<p>We first need to define the standard machine learning problem in the absence of constraints. We have a multiclass classification problem with 10 output classes. We could train the two-hidden-layer convolutional neural network we defined in the previous section to minimize some cost function. Specifically, we could use gradient descent to minimize the cross entropy (also called logistic loss for multiclass classification problems).  </p>

<p>
    Now let's suppose we want to add a safety constraint. One safety constraint we could enforce is that the accuracy (the fraction of correctly labeled digits) of our trained model must be at least 0.95, and we want that constraint to hold with 95% confidence. The problem can now be fully formulated as a Seldonian machine learning problem:
</p>

<p>
    Using gradient descent on the CNN we defined in the previous section, minimize the cross entropy of the model, subject to the safety constraint:
    <ul>
        <li> $g_1$ : $\text{ACC} \geq 0.95$, and $\delta=0.05$, where $\text{ACC}$ is the measure function for accuracy. </li>
    </ul>    
    Note that if this safety constraint is fulfilled, we can have high confidence (95% confidence) that the accuracy of the model, when applied to unseen data, is at least 0.95. This is <i>not</i> a property of even the most sophisticated models that can achieve accuracies of $\gt0.99$ on MNIST. 
</p>

<h5>Running the Seldonian algorithm</h5>
<p>
    If you are reading this tutorial, chances are you have already used the engine to run Seldonian algorithms. If not, please review the <a href="{{ "/tutorials/fair_loans_tutorial" | relative_url}}">Fair loans tutorial</a>. We need to create a <code class="highlight">SupervisedSpec</code> object, consisting of everything we will need to run the Seldonian algorithm. We will write a script that does this and then runs the algorithm. First, define the imports that we will need. The <code class="highlight">PytorchCNN</code> model that we defined above is already part of the library, living in a module called <code>seldonian.models.pytorch_cnn</code>, so we we import the model from that module. 
</p>
{% highlight python %}
#pytorch_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.models.pytorch_cnn import PytorchCNN
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
{% endhighlight python %}

<p>Next, set a random seed so that we can reproduce the result each time we run the script, set the regime and subregime of the problem, and then fetch the data. We obtain the data using PyTorch's <code class="highlight">torchvision</code> API, which provides a copy of the MNIST dataset. We will retrieve the training and test sets and then combine them into a single dataset. We then extract the features and labels from the combined data. The Seldonian algorithm will partition these into candidate and safety data according to the value of <code class="highlight">frac_data_in_safety</code>that we specify. </p>

{% highlight python %}
if __name__ == "__main__":
    torch.manual_seed(0)
    regime='supervised_learning'
    sub_regime='multiclass_classification'
    data_folder = '../../../notebooks/data'
    train_data = datasets.MNIST(
        root = data_folder,
        train = True,                         
        transform = ToTensor(), 
        download = False,            
    )
    test_data = datasets.MNIST(
        root = data_folder,
        train = False,                         
        transform = ToTensor(), 
        download = False,            
    )
    # Combine train and test data into a single tensor of 70,000 examples
    all_data = torch.vstack((train_data.data,test_data.data))
    all_targets = torch.hstack((train_data.targets,test_data.targets))
    N=len(all_targets) 
    assert N == 70000
    frac_data_in_safety = 0.5
    features = np.array(all_data.reshape(N,1,28,28),dtype='float32')/255.0
    labels = np.array(all_targets) # these are 1D so don't need to reshape them
{% endhighlight python %}

<p>Notice that we reshaped the features, changed their data type, and divided all of the pixel values by 255.0. This is all done so that the features are compliant with the input layer of the model. Next, let's create the <code class="highlight">SupervisedDataset</code> object. </p>
{% highlight python %}
    meta_information = {}
    meta_information['feature_col_names'] = ['img']
    meta_information['label_col_names'] = ['label']
    meta_information['sensitive_col_names'] = []
    meta_information['sub_regime'] = sub_regime

    dataset = SupervisedDataSet(
        features=features,
        labels=labels,
        sensitive_attrs=[],
        num_datapoints=N,
        meta_information=meta_information)
{% endhighlight python %}
<p>There are no sensitive attributes in this dataset. Let's define the constraint and make the parse tree for it.</p>
{% highlight python %}
    constraint_strs = ['ACC >= 0.95']
    deltas = [0.05] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime)
{% endhighlight python %}
<p>
    Now, let's create the model instance, specifying the device we want to run the model on. 
</p>
{% highlight python %}
    device = torch.device("mps")
    model = PytorchCNN(device)
{% endhighlight python %}
<p>
    In my case, I am running the script on an M1 Macbook Air that has a GPU. To specify that I want to do the model computations using this GPU, I use the device string <code class="highlight">"mps"</code>. If you have an NVIDIA chip (common on modern Windows machines), you can try the device string <code class="highlight">"cuda"</code>. If you do not have a GPU enabled, you can run on the CPU by using the device string <code class="highlight">"cpu"</code>. If you are not sure, you can check if cuda is available by doing:
</p>

{% highlight python %}
import torch
print(torch.cuda.is_available())
{% endhighlight python %}

<p> 
    If this prints <code class="highlight">True</code>, then you should be able to use <code class="highlight">cuda</code> as your device string. Similarly, on a mac with an M1 chip, you can check if the Metal Performance Shaders (MPS) driver is available by doing: 
</p>
{% highlight python %}
import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
{% endhighlight python %}
<p>
    If both are <code class="highlight">True</code>, then you should be able to use <code class="highlight">"mps"</code> as your device. If not, see this page for help setting up your environment to use the MPS driver: <a href="https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c">https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c</a>
</p>

<p> Now that the model is instantiated, we can get the randomized initial weights that PyTorch assigned to the parameters and use that as the initial values of $\theta$, the model weights, in gradient descent. We can also now specify that we want to use the cross entropy for our primary objective function, which is called <code class="highlight">multiclass_logistic_loss</code> in the toolkit: </p>
{% highlight python %}
    initial_solution_fn = model.get_initial_weights
    primary_objective_fn = objectives.multiclass_logistic_loss
{% endhighlight python %}

<p>
    We are now ready to create the spec object:
</p>
{% highlight python %}
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective_fn,
        use_builtin_primary_gradient_fn=False,
        sub_regime=sub_regime,
        initial_solution_fn=initial_solution_fn,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.001,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : True,
            'batch_size'    : 150,
            'n_epochs'      : 5,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        },
    )
{% endhighlight python %}
<p>
    Notice in <code class="highlight">optimization_hyperparams</code> that we are specifying <code class="highlight">'use_batches' : True</code>, indicating that we want to use batches in gradient descent. The batch size we request is <code class="highlight">150</code>, and we want to run for 5 epochs. Using batches of around this size will make gradient descent run significantly faster. If we don't use batches, we will be running every image in the candidate dataset (35,000 in this example) through the forward and backward pass of the model on every single step of gradient descent, which will be extremely slow. Also note that we had to play around with batch size, number of epochs, and the theta learning rate, <code class="highlight">alpha_theta</code>, to get gradient descent to converge appropriately. 
</p>

<p>
    Finally, we are ready to run the Seldonian algorithm using this spec object. 
</p>
{% highlight python %}
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=False,write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test.")
    else:
        print("Failed safety test")
    print("Primary objective evaluated on safety test:")
    print(st_primary_objective)
{% endhighlight python %}
<p> Here is the whole script all together:</p>

<div>
    <input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">
{% highlight python %}
#pytorch_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.models.pytorch_cnn import PytorchCNN
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    torch.manual_seed(0)
    regime='supervised_learning'
    sub_regime='multiclass_classification'
    data_folder = '../../../notebooks/data'
    train_data = datasets.MNIST(
        root = data_folder,
        train = True,                         
        transform = ToTensor(), 
        download = False,            
    )
    test_data = datasets.MNIST(
        root = data_folder,
        train = False,                         
        transform = ToTensor(), 
        download = False,            
    )
    # Combine train and test data into a single tensor of 70,000 examples
    all_data = torch.vstack((train_data.data,test_data.data))
    all_targets = torch.hstack((train_data.targets,test_data.targets))
    N=len(all_targets) 
    assert N == 70000
    frac_data_in_safety = 0.5
    features = np.array(all_data.reshape(N,1,28,28),dtype='float32')/255.0
    labels = np.array(all_targets) # these are 1D so don't need to reshape them

    meta_information = {}
    meta_information['feature_col_names'] = ['img']
    meta_information['label_col_names'] = ['label']
    meta_information['sensitive_col_names'] = []
    meta_information['sub_regime'] = sub_regime

    dataset = SupervisedDataSet(
        features=features,
        labels=labels,
        sensitive_attrs=[],
        num_datapoints=N,
        meta_information=meta_information)

    constraint_strs = ['ACC >= 0.95']
    deltas = [0.05] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime)
    device = torch.device("mps")
    model = PytorchCNN(device)

    initial_solution_fn = model.get_initial_weights
    
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=objectives.multiclass_logistic_loss,
        use_builtin_primary_gradient_fn=False,
        sub_regime=sub_regime,
        initial_solution_fn=initial_solution_fn,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.001,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : True,
            'batch_size'    : 150,
            'n_epochs'      : 5,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        },
    )

    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=False,write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test.")
    else:
        print("Failed safety test")
    print("Primary objective evaluated on safety test:")
    print(st_primary_objective)
{% endhighlight python %}
<p>
    If you save this script to a file named <code>pytorch_mnist.py</code> and run it like:
</p>
{% highlight javascript %}
$ python pytorch_mnist.py
{% endhighlight javascript %}
</div>
<p>
    You will see the following output:
</p>
{% highlight python %}
Have 5 epochs and 234 batches of size 150

Epoch: 0, batch iteration 0
Epoch: 0, batch iteration 10
Epoch: 0, batch iteration 20
Epoch: 0, batch iteration 30
Epoch: 0, batch iteration 40
...
Epoch: 4, batch iteration 200
Epoch: 4, batch iteration 210
Epoch: 4, batch iteration 220
Epoch: 4, batch iteration 230
Wrote /Users/ahoag/beri/code/engine-repo-dev/examples/pytorch_mnist_batch/logs/candidate_selection_log10.p with candidate selection log info
Passed safety test.
Primary objective evaluated on safety test:
0.0812836065524022
{% endhighlight python %}

<p>
    The location of the candidate selection log info file will be in a <code>logs/</code> subfolder of wherever you ran the script. The safety test should also pass for you, though the value of the primary objective on the safety test might differ slightly due to your machine's random number generator. The important thing is that the gradient descent curve is similar. Plot it using the following Python code, replacing the path to the <code class="highlight">logfile</code> with the location where that file was saved on your machine.
</p>
{% highlight python %}
from seldonian.utils.plot_utils import plot_gradient_descent
from seldonian.utils.io_utils import load_pickle
logfile = "/Users/ahoag/beri/code/engine-repo-dev/examples/pytorch_mnist_batch/logs/candidate_selection_log10.p"
sol_dict = load_pickle(logfile)
plot_gradient_descent(sol_dict,'cross entropy')
{% endhighlight python %}
<p>
    You should see a similar plot to this one:
</p>
<div align="center">
    <figure>
        <img src="{{ "/assets/img/mnist_cs.png" | relative_url}}" class="img-fluid mt-4" style="width: 75%"  alt="Candidate selection"> 
        <figcaption align="left"> <b>Figure 1</b> - How the parameters of the Lagrangian optimization problem changed during gradient descent on the MNIST task. The panels show the values of the (left) primary objective $\hat{f}(\theta,D_\mathrm{cand})$ (in this case the cross entropy), (middle left) single Lagrange multiplier, ${\lambda_1}$, (middle right) predicted high-confidence upper bound (HCUB) on the  constraint function, $\hat{g}_1(\theta,D_\mathrm{cand}) = \text{ACC} - 0.95$, and (right) the Lagrangian $\mathcal{L}(\theta,\lambda)$. The dotted lines indicate where the optimum was found. The optimum is defined as the feasible solution with the lowest value of the primary objective. A feasible solution is one where $\mathrm{HCUB}(\hat{g}_i(\theta,D_\mathrm{cand})) \leq 0, i \in \{1 ... n\}$. In this example, we only have one constraint, and the infeasible region is shown in red in the middle right plot. The noise in the curves for $\hat{f}$ and $\mathcal{L}$ is due to the fact that we batched the candidate data in gradient descent.</figcaption>
    </figure>
</div>

<h3> The PyTorch model base class</h3>
<p> Before diving into the code, it is helpful to understand what the role of the model is in the Seldonian algorithm, in general. The first place the model enters the algorithm is in candidate selection. If you are planning to use PyTorch to build your model, the chances are your model is quite complex, i.e., it has a lot of parameters. In that case, you almost always want to use gradient descent (or some variation) for your optimization process in candidate selection, as opposed to alternatives like black box optimization. 
</p>

<p>
    During each step of gradient descent, we first evaluate the primary objective function, $\hat{f}(\theta,D)$, and the upper bounds on the constraint functions, $U(\hat{g_i}(\theta,D))$. Part of evaluating these functions involves making a forward pass through the model to predict outputs given inputs, $(\theta, D)$. At the end of each iteration of gradient descent, we update the model weights using a factor that includes  the gradients of $\hat{f}$ and $U$ with respect to $\theta$, $\frac{\partial \hat{f}(\theta,D)}{\partial \theta}$ and $\frac{\partial (U(\hat{g_i}(\theta,D)))}{\partial \theta}$, respectively. Obtaining these gradients requires making a backward pass through the model. For the models in the library that are written solely in NumPy and native Python, the backward pass is done using automatic differentiation using a library called <a href="https://github.com/HIPS/autograd">autograd</a> (not to be confused with PyTorch's internal automatic differentiation which is also called autograd!), and we don't need to worry about coding that part. However, the autograd library that we use in the toolkit does not know how to differentiate PyTorch tensors, so we need to explicitly write code to do the backward pass if using a PyTorch model. We also need to tell the toolkit's autograd library not to automatically differentiate the PyTorch model's forward pass method. 
 </p>

<p>
   The abstract base class <code class='highlight'>seldonian.models.pytorch_model.SupervisedPytorchBaseModel</code> does all of this for you. It implements the forward pass and the backward pass (as well as a number of other methods which we will explain below), and it takes care of telling autograd to ignore its forward and backward pass methods. At this point, you might ask why don't we just use PyTorch's internal automatic differentiation library throughout the toolkit. The reason is that this would require writing the entire toolkit in PyTorch, requiring us to represent all data as PyTorch tensors. This would make it harder to include models from other popular deep learning libraries like TensorFlow. The approach we decided to take is agnostic to the deep learning library, but in this tutorial we focus on PyTorch. 

</p>

<p>
   We will present the base class one method at a time for clarity, but the source code for the entire base class can be found here: <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/pytorch_model.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/pytorch_model.py</a>. Let's start by looking at the <code class="highlight">__init__()</code> method.
</p>


{% highlight python %}
class SupervisedPytorchBaseModel(SupervisedModel):
    def __init__(self,device,**kwargs):
        """ Base class for Supervised learning Seldonian
        models implemented in PyTorch

        :param device: The PyTorch device string indicating the
            hardware on which to run the model,
            e.g. "cpu", "cuda", "mps".
        :type device: str
        """
        super().__init__()
        self.device=device
        self.pytorch_model = self.create_model(**kwargs)
        self.pytorch_model.to(self.device)
        self.param_sizes = self.get_param_sizes()
{% endhighlight python %}

<p>
    The first thing to notice is that there is only one required parameter, <code class="highlight">device</code>. This is how you tell PyTorch what hardware to use when computing, e.g., CPU vs. GPU. The actual PyTorch model object is created in a class method <code class="highlight">self.create_model()</code>, which must be defined in your child class. For convenience, the number of parameters in each of the model's layers are stored in a list <code class="highlight">self.param_sizes</code>. More on this later. The next method we will look at is <code class="highlight">predict()</code>
</p>
{% highlight python %}
def predict(self,theta,X,**kwargs):
        """ Do a forward pass through the PyTorch model.
        Must convert back to numpy array before returning 

        :param theta: model weights
        :type theta: numpy ndarray

        :param X: model features
        :type X: numpy ndarray

        :return pred_numpy: model predictions 
        :rtype pred_numpy: numpy ndarray same shape as labels
        """
        return pytorch_predict(theta,X,self)
{% endhighlight python %}

<p>
    <code class="highlight">predict()</code> is a required method of all Seldonian models (base class <code class="highlight">seldonian.models.models.SeldonianModel</code>, of which any Seldonian PyTorch model is a child). <code class="highlight">predict()</code> is called when the engine needs the result of the model's forward pass. Here, this method is just a wrapper that calls the function, <code class="highlight">pytorch_predict()</code>. That function is not a method of the base class, but rather a function contained within the same module. Here is that function:
</p>

{% highlight python %}
@primitive
def pytorch_predict(theta,X,model,**kwargs):
    """ Do a forward pass through the PyTorch model.
    Must convert back to numpy array before returning 

    :param theta: model weights
    :type theta: numpy ndarray

    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
        SupervisedPytorchBaseModel 

    :return pred_numpy: model predictions 
    :rtype pred_numpy: numpy ndarray same shape as labels
    """
    # First update model weights
    
    model.update_model_params(theta,**kwargs)
    # Do the forward pass
    pred = model.forward_pass(X,**kwargs)
    # set the predictions attribute of the model
    model.predictions = pred
    # Convert predictions into a numpy array
    pred_numpy = pred.cpu().detach().numpy()
    return pred_numpy
{% endhighlight python %}

<p>
    <code class="highlight">pytorch_predict()</code> takes as arguments <code class="highlight">theta</code>, the weights from the current step of gradient descent, <code class="highlight">X</code>, the features of the dataset, and <code class="highlight">model</code>, which is an instance of your Seldonian model. This function is wrapped by the decorator <code class="highlight">@primitive</code>, which tells autograd not to look inside of it. By default, autograd will try to compute the gradient of this function, and it will fail if it does. The <code class="highlight">@primitive</code> decorator is what allows us to use PyTorch tensors (and other non-NumPy objects) in the model. There is another function in this same module called <code class="highlight">pytorch_predict_vjp()</code> which defines the gradient of this function, which we will cover later. <code class="highlight">pytorch_predict()</code> calls several model methods (which we will explain shortly), obtains the outputs of the forward pass of the model, and finally converts them back to a numpy array. The first model method this function calls is <code class="highlight">model.update_model_params()</code>, which we reproduce here: 

</p>
{% highlight python %}
def update_model_params(self,theta,**kwargs):
    """ Update all model parameters using theta,
    which must be reshaped

    :param theta: model weights
    :type theta: numpy ndarray
    """
    # Update model parameters using flattened array
    with torch.no_grad():
        i = 0
        startindex = 0
        for param in self.pytorch_model.parameters():
            if param.requires_grad:
                nparams = self.param_sizes[i]
                param_shape = param.shape
                theta_numpy = theta[startindex:startindex+nparams]
                theta_torch = torch.from_numpy(theta_numpy).view(param_shape)
                param.copy_(theta_torch)
                i+=1
                startindex+=nparams
    return
{% endhighlight python %}

<p>
    This method uses the flattened theta array passed in as input and updates the model weights. Each model parameter is stored as a separate tensor, so we can now see it was a good idea to keep track of <code class="highlight">self.param_sizes</code> in <code class="highlight">__init__</code>. This allows us to quickly partition <code class="highlight">theta</code>, convert it from a NumPy array to a PyTorch tensor, and reshape it so that has the correct dimensionality of the PyTorch model parameter tensor. The next method that <code class="highlight">pytorch_predict()</code> calls is <code class="highlight">model.forward_pass()</code>.
</p>
{% highlight python %}
def forward_pass(self,X,**kwargs):
    """ Do a forward pass through the PyTorch model and return the 
    model outputs (predicted labels). The outputs should be the same shape 
    as the true labels

    :param X: model features
    :type X: numpy ndarray

    :return: predictions
    :rtype: torch.Tensor
    """
    X_torch = torch.tensor(X,requires_grad=True).float().to(self.device)
    predictions = self.pytorch_model(X_torch)
    return predictions
{% endhighlight python %}

<p>
    In PyTorch, all you have to do to perform a forward pass is call the PyTorch model. The only inputs are the features, which must be tensors, hence the conversion from NumPy array to PyTorch tensor. That is the extent of the forward pass! Now, we can move on to the backward pass. 
</p>
<p>
We mentioned earlier that we have to explicitly specify the gradient of the <code class="highlight">pytorch_predict()</code> method because autograd is not capable of computing it for us. We do that with a function called <code class="highlight">pytorch_predict_vjp()</code>. We also have to include a line that links the two functions together, instructing autograd where to look to find the gradient, given that it isn't calculating it itself. 
</p>
{% highlight python %}
def pytorch_predict_vjp(ans,theta,X,model):
    """ Do a backward pass through the PyTorch model,
    obtaining the Jacobian d pred / dtheta. 
    Must convert back to numpy array before returning 

    :param ans: The result from the forward pass
    :type ans: numpy ndarray
    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
        SupervisedPytorchBaseModel 

    :return fn: A function representing the vector Jacobian operator
    """
    def fn(v):
        # v is a vector of shape ans, the return value of mypredict()
        # return a 1D array [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2]],
        # where i is the data row index
        model.zero_gradients()
        external_grad = torch.from_numpy(v).float().to(model.device)
        dpred_dtheta = model.backward_pass(external_grad)
        return np.array(dpred_dtheta)
    return fn

# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(pytorch_predict,pytorch_predict_vjp)
{% endhighlight python %}

<p>
    The first thing you may notice is that this function returns another function, <code class="highlight">fn</code>. The reason for this is beyond the scope of this tutorial, but if you are curious, the explanation is provided here: <a href="https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#extend-autograd-by-defining-your-own-primitives">https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#extend-autograd-by-defining-your-own-primitives</a>. 
</p>

<p>
    The function <code class="highlight">pytorch_predict_vjp()</code> takes four arguments, <code class="highlight">ans</code>, the result from calling the <code class="highlight">pytorch_predict()</code> function given the other inputs: <code class="highlight">theta, X, model</code>, which you will notice are the three arguments to <code class="highlight">pytorch_predict()</code>. In the function that is returned, <code class="highlight">fn</code>, we zero the gradients of the model, perform the backward pass to get the gradient, then convert the gradient to a NumPy array. Let's look at the model method <code class="highlight">model.zero_gradients()</code>.
</p>
{% highlight python %}
def zero_gradients(self):
    """ Zero out gradients of all model parameters """
    for param in self.pytorch_model.parameters():
        if param.requires_grad:
            if param.grad is not None:
                param.grad.zero_()
    return
{% endhighlight python %}
<p>
    This method is pretty straightforward. It sets the gradients of the model with respect to each model parameter to zero. This is necessary so that the gradients from previous backward passes (from previous steps of gradient descent) do not accumulate and influence the result of the current backward pass. Finally, let's look at the model method: <code class="highlight">model.backward_pass()</code>.
</p>
{% highlight python %}
def backward_pass(self,external_grad):
    """ Do a backward pass through the PyTorch model and return the
    (vector) gradient of the model with respect to theta as a numpy ndarray

    :param external_grad: The gradient of the model with respect to itself
        see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
        for more details
    :type external_grad: torch.Tensor 
    """
    self.predictions.backward(gradient=external_grad,retain_graph=True)
    grad_params_list = []
    for param in self.pytorch_model.parameters():
        if param.requires_grad:
            grad_numpy = param.grad.cpu().numpy()
            grad_params_list.append(grad_numpy.flatten())
    return np.concatenate(grad_params_list)
{% endhighlight python %}

<p>
    This method takes a single argument, <code class="highlight">external_grad</code>, which is the gradient of the output of the forward pass with respect to itself. If you're new to PyTorch and this doesn't make sense, read this: <a href="https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd">https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd</a>, but note that the "autograd" they are referring to is internal to PyTorch and is not the same as the autograd library we use in the toolkit for automatic differentiation. Backpropagation is done in the step: <code class="highlight">self.predictions.backward(gradient=external_grad,retain_graph=True)</code>, which is the standard PyTorch syntax. In the rest of the method, we collect the gradients of each of the model parameters and then store them in a flattened numpy array. The flattening is done so that the gradient array has the same shape as the theta array. This makes it easy to do the gradient update step in gradient descent. 
</p>

<p>There are two more methods of the base class, and it is helpful to understand what they do. </p>
{% highlight python %}
def get_initial_weights(self,*args):
    """ Return initial weights as a flattened 1D array
    Also return the number of elements in each model parameter """
    layer_params_list = []
    for param in self.pytorch_model.parameters():
        if param.requires_grad:
            param_numpy = param.cpu().detach().numpy()
            layer_params_list.append(param_numpy.flatten())
    return np.concatenate(layer_params_list)

def get_param_sizes(self):
    """ Get the sizes (shapes) of each of the model parameters
    """
    param_sizes = []
    for param in self.pytorch_model.parameters():
        if param.requires_grad:
            param_sizes.append(param.numel())
    return param_sizes
{% endhighlight python %}

<p>
    <code class="highlight">get_initial_weights()</code> is useful because you may want to initialize your model weights using PyTorch's initializers. When the model is created, PyTorch automatically initializes the weights, so all we have to do is extract them. This method allows you to do that and to get the result in a flattened numpy array that is ready to be used in the toolkit's gradient descent module. 
</p>

<p>
    <code class="highlight">get_param_sizes()</code> is a method we have already seen in action. All it does is store the number of parameters in each of the model's parameter sets (each layer can have more than one set of parameters) in a list. 
</p>

<p>We have now covered the entire base class. Any of these methods can be overridden in a child class, although we strongly advise caution when overriding <code class="highlight">predict()</code>, unless you really understand what is going on under the hood with autograd primitives. At this point, we are ready to build a usable Seldonian model with PyTorch. </p>

<h3>Summary</h3>
<p>In this tutorial, we demonstrated how to implement a deep learning model using Pytorch and integrate it with the Seldonian toolkit. The base class we presented contains a lot of the basic functionality that is generally needed for any PyTorch model. We hope by providing an example implementation of a convolutional neural network, it will be easier for you to implement your own PyTorch models.  </p>

</div> 