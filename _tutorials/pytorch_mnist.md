---
layout: tutorial
permalink: /tutorials/pytorch_mnist/
prev_url: /tutorials/gridworld_RL_tutorial/
prev_page_name: (G) Reinforcement learning first tutorial
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Tutorial H: Creating your first Seldonian PyTorch model </h2>
    <hr class="my-4">
    <h3>Introduction</h3>
    <p>
        Deep learning is a cornerstone of modern machine learning. It is being used in nearly every industry, and it is increasingly affecting our lives. Due to the complexity of some deep learning models (e.g., <a href="https://deepai.org/machine-learning-glossary-and-terms/hidden-layer-machine-learning">hidden layers</a>), they can be opaque to the people applying them. This makes it especially important to be able to put constraints on their behavior. 
    </p>

    <p> 
        <a href="https://pytorch.org/">PyTorch</a> is an open-source machine learning framework written in Python that is extremely popular for deep learning. PyTorch uses objects called tensors (<code class="highlight">torch.Tensor</code>) to represent multi-dimensional arrays of data. The Seldonian toolkit uses NumPy arrays to represent data and as a whole is not written in PyTorch. Fortunately, PyTorch tensors can be converted to NumPy arrays with relative ease, and vice versa. The toolkit takes care of this conversion, allowing you to focus on implementing your PyTorch model. For now, only the Seldonian model can be written in PyTorch. The dataset must still be provided to the engine as numpy arrays, not PyTorch tensors. Currently, PyTorch is <b>only supported for supervised learning models</b>. We plan to support deep reinforcement learning policies in PyTorch in the near future. 
    </p>

    <p>
        In this tutorial, you will learn how to build and run a GPU-accelerated convolutional neural network (CNN) on the MNIST database of handwritten digits with a simple safety constraint.
    </p>

    <p>
        <b>Note:</b> To enable GPU acceleration, you will need to have the proper setup for your machine. For example, CUDA needs to be enabled if you have an NVIDIA graphics card. The good news is that even if you don't have access to a GPU, you will still be able to run the model on your CPU. To learn more about GPU-acceleration for PyTorch, see the "Install Pytorch" section of this page: <a href="https://pytorch.org/">https://pytorch.org/</a>
  
    </p> 

<h3> Implementing a convolutional neural network with PyTorch </h3>

<p>
    In this section, we will build a CNN in PyTorch that is compatible with the Seldonian Toolkit. First, make sure you have the latest version of the engine installed. </p>    
{% highlight javascript %}
$ pip install --upgrade seldonian-engine
{% endhighlight javascript %}
    <p>
        You will also need the <code>torchvision</code> Python package if you want to run the example below, and this package is not included in the library:
    </p>
{% highlight python %}
$ pip install torchvision
{% endhighlight python %}
<p>
    It is important to make a clear distinction when referring to "models" throughout this tutorial. We will use the term "Seldonian model" to refer to the highest level model abstraction in the toolkit. The Seldonian model is the thing that communicates with the rest of the toolkit. The Seldonian model we will build in this tutorial consists of a "PyTorch model," a term which we will use to refer to the actual PyTorch implementation of the neural network. The PyTorch model <i>does not</i> communicate with the other pieces of the Seldonian Toolkit, whereas the Seldonian model does. 
</p>
<p>
    Seldonian models are implemented as Python classes. There are three requirements for creating a new Seldonian model class with PyTorch:
    <ol>
        <li>The class must inherit from this base class: <code class="highlight">seldonian.models.pytorch_model.SupervisedPytorchBaseModel</code>. </li>
        <li>The class must take as input a <code class="highlight">device</code> string, which specifies the hardware (e.g. CPU vs. GPU) on which to run the model, and pass that device string to the base class's <code class="highlight">__init__</code> method.</li>
        <li>The class must have a <code class="highlight">create_model()</code> method in which it defines the PyTorch model and returns it as an instance of <code class="highlight">torch.nn.Module</code> or <code class="highlight">torch.nn.Sequential</code>. The reason for this is that the PyTorch model must have a <code class="highlight">forward()</code> method, and these are two common model classes that have that method. </li>
    </ol> 
    We will name our new model <code class="highlight">PyTorchCNN</code> and use the network architecture described in this article: <a href="https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118">https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118</a>. This is an overview of the flow of the network: 
</p>
{% highlight python %}
Sequential(
  (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (4): ReLU()
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Flatten(start_dim=1, end_dim=-1)
  (7): Linear(in_features=1568, out_features=10, bias=True)
  (8): Softmax(dim=1)
)
{% endhighlight python %}

<p>
    This model has around 29,000 parameters. We consider the <code class="highlight">Conv2d</code>, <code class="highlight">ReLU</code>, and <code class="highlight">MaxPool2d</code> to be a single hidden layer, such that this network comprises two hidden layers, followed by a single output <code class="highlight">Linear</code>+<code class="highlight">Softmax</code> layer.
</p>

<p>
      Here is the implementation of the Seldonian model that implements this PyTorch model, meeting the three requirements specified above. 
</p>
{% highlight python %}
from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
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
    It is very important that the <code class="highlight">create_model()</code> method returns the PyTorch model object. In this implementation, we used PyTorch's <a href="https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html">sequential</a> container to hold a sequence of Pytorch <a href="https://pytorch.org/docs/stable/generated/torch.nn.Module.html">nn.Modules</a>. This is just one pattern that PyTorch provides, and it is not required in the <code class="highlight">create_model()</code> method. As long as this method returns a PyTorch model object that has a <code class="highlight">forward()</code> method, it should be compatible with the toolkit. Also note that we applied a softmax layer at the end of the model so that the outputs of the model are probabilities instead of logits. We did this because the primary objective function we will use when we run the model in the following example expects probabilities. This might not always be the case for your use case, so be aware of what your objective function expects from your model. 
</p> 

<p>
    At this point, this model is ready to use in the toolkit.
</p>
     
<h3> MNIST with a safety constraint </h3>

<p> The Modified National Institute of Standards and Technology <a href="https://en.wikipedia.org/wiki/MNIST_database">(MNIST)</a> database is a commonly used dataset in machine learning research. It contains 60,000 training images and 10,000 testing images of the handwritten digits 0-9. Yes, there are models that are superior to the simple CNN we built in terms of accuracy (equivalently, error rate), but objective in this tutorial is to show how to create a Seldonian version of a PyTorch deep learning model, not to achieve maximum accuracy. 
</p>

<h5>Formulate the Seldonian ML problem</h5>
<p>We first need to define the standard machine learning problem in the absence of constraints. We have a multiclass classification problem with 10 output classes. We could train the two-hidden-layer convolutional neural network we defined in the previous section to minimize some cost function. Specifically, we could use gradient descent to minimize the cross entropy (also called logistic loss for multiclass classification problems).  </p>

<p>
    Now let's suppose we want to add a safety constraint. One safety constraint we could enforce is that the accuracy (the fraction of correctly labeled digits across all 10 classes) of our trained model must be at least 0.95, and we want that constraint to hold with 95% confidence. The problem can now be fully formulated as a Seldonian machine learning problem:
</p>

<p>
    Using gradient descent on the CNN we defined in the previous section, minimize the cross entropy of the model, subject to the safety constraint:
    <ul>
        <li> $g_1$ : $\text{ACC} - 0.95$ (equivalent to $\text{ACC} <= 0.95$), and $\delta=0.05$, where $\text{ACC}$ is the measure function for accuracy. </li>
    </ul>    
    Note that if this safety constraint is fulfilled, we can have high confidence ([$1-\delta$]-confidence) that the accuracy of the model, when applied to unseen data, is at least 0.95. This is <i>not</i> a property of even the most sophisticated models that can achieve accuracies of $\gt0.999$ on MNIST. 
</p>

<h5>Running the Seldonian algorithm</h5>
<p>
    If you are reading this tutorial, chances are you have already used the engine to run Seldonian algorithms. If not, please review the <a href="{{ "/tutorials/fair_loans_tutorial" | relative_url}}">Fair loans tutorial</a>. We need to create a <code class="highlight">SupervisedSpec</code> object, consisting of everything we will need to run the Seldonian algorithm. We will write a script that does this and then runs the algorithm. First, let's define the imports that we will need. The <code class="highlight">PytorchCNN</code> model that we defined above is already part of the library, living in a module called <code>seldonian.models.pytorch_cnn</code>, so we can import the model from that module. 
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

<p>Next, set a random seed so that we can reproduce the result each time we run the script, set the regime and subregime of the problem, and then fetch the data. We obtain the data using PyTorch's <code class="highlight">torchvision</code> API, which provides a copy of the MNIST dataset. We will retrieve the training and test sets and then combine them into a single dataset. If you want, you can set <code class="highlight">download=True</code> to download the data to disk so that you can run the script in the future when you are offline. If you choose to do that, change the <code class="highlight">data_folder</code> to where you want to download the data. We extract the features and labels from the combined data. The Seldonian algorithm will partition these into candidate and safety data according to the value of <code class="highlight">frac_data_in_safety</code> that we specify in the spec object. In this example, we will use <code class="highlight">frac_data_in_safety=0.5</code> </p>

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

<p>Notice that we reshaped the features, changed their data type, divided all of the pixel values by 255.0. This is all done so that the features are compliant with the input layer of the model. We also converted them to NumPy arrays, a necessary step even though they will get converted back to Tensors before they are fed into the model. The reason why we have to do this is beyond the scope of this tutorial and will be explained in a future tutorial. Next, let's create the <code class="highlight">SupervisedDataset</code> object. </p>
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
<p>There are no sensitive attributes in this dataset. Recall that the constraint we defined is that the accuracy should be at least $0.95$ with a confidence level of $\delta=0.05$. We create this constraint as follows: </p>
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
    In my case, I am running the script on an M1 Macbook Air that has a GPU. To specify that I want to do the model computations using this GPU, I use the device string <code class="highlight">"mps"</code>. If you have an NVIDIA chip (common on modern Windows machines), you can try the device string <code class="highlight">"cuda"</code>. The worst that will happen is you'll get an error message of the sort:
</p>
{% highlight python %}
AssertionError: Torch not compiled with CUDA enabled
{% endhighlight python %}
    <p> If you do not have a GPU enabled, you can run on the CPU by using the device string <code class="highlight">"cpu"</code>. If you are not sure, you can check if cuda is available by doing:
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
    Notice in <code class="highlight">optimization_hyperparams</code> that we are specifying <code class="highlight">'use_batches' : True</code>, indicating that we want to use batches in gradient descent. The batch size we request is <code class="highlight">150</code>, and we want to run for 5 epochs. Using batches of around this size will make gradient descent run significantly faster. If we don't use batches, we will be running every image in the candidate dataset (35,000 in this example) through the forward and backward pass of the model on every single step of gradient descent, which will be extremely slow. We should note that we had to play around with batch size, number of epochs, and the theta learning rate, <code class="highlight">alpha_theta</code>, to get gradient descent to converge appropriately. We did not perform a proper hyperparameter optimization process, which we recommend for a real problem.
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
    The whole script takes about 90 seconds to run on my M1 Macbook Air using the GPU, and about 5 minutes to run on the CPU. The location of the candidate selection log info file will be in a <code>logs/</code> subfolder of wherever you ran the script. The safety test should also pass for you, though the value of the primary objective (cross entropy over all 10 classes) on the safety test might differ slightly because your machine's random number generator may differ from mine. The important thing is that the gradient descent curve is similar. Plot it using the following Python code, replacing the path to the <code class="highlight">logfile</code> with the location where that file was saved on your machine.
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
        <figcaption align="left"> <b>Figure 1</b> - How the parameters of the Lagrangian optimization problem changed during gradient descent on the MNIST task. The panels show the values of the (left) primary objective $\hat{f}(\theta,D_\mathrm{cand})$ (in this case the cross entropy), (middle left) single Lagrange multiplier, ${\lambda_1}$, (middle right) predicted high-confidence upper bound (HCUB) on the  constraint function, $\hat{g}_1(\theta,D_\mathrm{cand}) = \text{ACC} - 0.95$, and (right) the Lagrangian $\mathcal{L}(\theta,\lambda)$. The dotted lines indicate where the optimum was found. The optimum is defined as the feasible solution with the lowest value of the primary objective. A feasible solution is one where $\mathrm{HCUB}(\hat{g}_i(\theta,D_\mathrm{cand})) \leq 0, i \in \{1 ... n\}$.  In this example, we only have one constraint. The infeasible region is shown in red in the middle right plot. The feasible region is shown in white in the same plot. The noise in the curves for $\hat{f}$ and $\mathcal{L}$ is due to the fact that we batched the candidate data in gradient descent.</figcaption>
    </figure>
</div>

<h3>Summary</h3>
<p>In this tutorial, we demonstrated how to implement a deep learning model using Pytorch and integrate it with the Seldonian toolkit. We created a convolutional neural network and enforced a simple safety constraint on it. The safety constraint was simple only for the purposes of creating a simple example. The constraints that one can apply to this PyTorch model are no more restricted than to any other supervised learning model. We hope by providing an example implementation of a convolutional neural network, it will be easier for you to implement your own safe and fair PyTorch models.  </p>

</div> 