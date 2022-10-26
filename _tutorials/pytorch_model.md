---
layout: tutorial
permalink: /tutorials/pytorch_model/
prev_url: /tutorials/gridworld_RL_tutorial/
prev_page_name: (G) Reinforcement learning first tutorial
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Tutorial H: Using a PyTorch model</h2>
    <hr class="my-4">
    <h3>Outline</h3>
    <p>In this tutorial, you will learn how to:
    <ul>
        <li>create a simple PyTorch model that can be integrated with the toolkit</li>
        <li>integrate more complex PyTorch models into the toolkit </li>
        <li>(optional) override autograd to insert any external code into the toolkit </li>
    </ul>

    </p> 
    <h3> Background: Autograd, PyTorch and this toolkit </h3>
    <p>
        To use gradient-based optimization during <a href="/tutorials/alg_details_tutorial/#candidate_selection">candidate selection</a>, we need to be able to compute the gradients of the primary objective function and the upper bounds on the constraint functions. The constraint functions, and perhaps also the primary objective function, are arbitrary functions that the user has defined. Is is often impractical or impossible to obtain closed forms for their gradients (or their upper bounds), so we rely on automatic differentiation (autodiff) to obtain them. The package we use within the toolkit for autodiff is called <a href="https://github.com/HIPS/autograd">Autograd</a>. When asked to compute a gradient of a function, autograd traces through the entire function, creating a graph of operations so that it can apply the <a href="https://en.wikipedia.org/wiki/Chain_rule">chain rule</a>. If autograd detects any objects or operations that do not belong to numpy (actually, autograd's version of numpy) or Python's standard library, it will raise an error. 
    </p>

    <p>
        <a href="https://pytorch.org/">PyTorch</a> is an open-source machine learning framework that is particularly popular for deep learning. PyTorch uses objects called tensors (<code class="highlight">torch.Tensor</code>) to represent multi-dimensional arrays of data. PyTorch tensors are examples of objects that autograd cannot differentiate. Fortunately, there is a way to override autograd's default behavior and tell it not to look inside of a specific function when figuring out that function's gradient. When doing this, one must provide the gradient (the vector Jacobian product, actually) of that function as another function and then link the original function with its gradient function so that autograd knows not to try to compute the gradient. This is the method we will use to allow PyTorch models to be used within in the toolkit. There are some requirements when taking this approach, which we will explore in the next few sections.
    </p>

     <p><b>Note 2:</b> this approach overrides a single forward and backward pass through the PyTorch model. PyTorch is not being used to perform the entire optimization process (although it could, in principle). At each step in gradient descent, autograd computes gradients of every other function in the call stack until it gets to the call of the Seldonian model's predict function (forward pass through the PyTorch model) or a call for its gradient (backward pass). </p>

     <p><b>Note 3:</b> this approach can in principle work for any other autograd-incompatible library, such as TensorFlow or even Python code that makes calls to libraries not written in Python. </p>

<h3>How do we introduce a PyTorch model into the toolkit?</h3>
<p>
    When we wrap a function with the <code class='highlight'>@autograd.extend.primitive</code> decorator, we are telling autograd to ignore the code within that function and its gradient function (for more details see the section below: <a href="#details">(Optional) How to keep autograd from looking inside of a function</a>. This opens the door to allow PyTorch code (and code from other deep learning libraries, possibly) into our functions and gradient functions. There are some requirements that must be met (see open <a href="https://github.com/HIPS/autograd/issues/584">GitHub issue</a> discussing these requirements and whether they can be removed):
    <ol>
        <li id='requirement1'>The function and its gradient function must return something that autograd can differentiate </li>
        <li id='requirement2'>The function and its gradient must strictly be functions, not class methods</li>
        <li id='requirement3'>Perhaps obvious, but you need to be able to write down the gradient function</li>
    </ol>
    The first requirement is the strictest and somewhat counterintuitive, but we can build around it. The second is an inconvenience but easy enough to overcome. The third point is not a big deal because PyTorch also has its own autodiff functionality that we can harness to compute the VJP of any PyTorch model. This is nice because it drastically simplifies what we have to write inside the VJP function.
</p>

<p>
    We created a PyTorch model base class for supervised learning in the module <code>seldonian.models.models</code> called <code class='highlight'>SupervisedPytorchBaseModel</code> which can be used as a parent class for implementing your own PyTorch model classes. We also implemented a generalized predict function and its gradient function and linked them so that autograd knows not to look inside of them. Here is a peek at the base class, the predict function and its gradient function.

</p>
{% highlight python %}
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd.extend import primitive, defvjp
from seldonian.models.models import SupervisedModel

import torch

class SupervisedPytorchBaseModel(SupervisedModel):
    def __init__(self,input_dim,output_dim,**kwargs):
        """ Base class for Supervised learning Seldonian
        models implemented in Pytorch

        :param input_dim: Number of features
        :param output_dim: Size of output layer (number of label columns)
        """
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.pytorch_model = self.create_model(**kwargs)
        

    def predict(self,theta,X,**kwargs):
        """ Predict label using the linear model

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predicted labels
        :rtype: numpy ndarray
        """
        return pytorch_predict(theta,X,self,**kwargs)

    def create_model(self,**kwargs):
        """ Create the pytorch model and return it
        """
        raise NotImplementedError("Implement this method in child class")

    def update_model_params(self,theta,**kwargs):
        """ Update weights of PyTorch model using theta,
        the weights from the previous step of gradient descent

        :param theta: model weights
        :type theta: numpy ndarray
        """
        raise NotImplementedError("Implement this method in child class")

    def zero_gradients(self,*kwargs):
        """ Zero out the gradients of all model parameters """
        raise NotImplementedError("Implement this method in child class")

    def forward_pass(self,X,**kwargs):
        """ Do a forward pass through the PyTorch model and return the 
        model outputs (predicted labels). The outputs should be the same shape 
        as the true labels
    
        :param X: model features
        :type X: numpy ndarray
        """
        raise NotImplementedError("Implement this method in child class")

    def backward_pass(self,external_grad,**kwargs):
        """ Do a backward pass through the PyTorch model and return the
        (vector) gradient of the model with respect to theta as a numpy ndarray

        :param external_grad: The gradient of the model with respect to itself
            see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
            for more details
        :type external_grad: torch.Tensor 
        """
        raise NotImplementedError("Implement this method in child class")
@primitive
def pytorch_predict(theta,X,model,**kwargs):
    """ Do a forward pass through the PyTorch model.
    Must convert back to numpy array before returning 

    :param theta: model weights
    :type theta: numpy ndarray

    :param X: model features
    :type X: numpy ndarray

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
    pred_numpy = pred.detach().numpy()
    return pred_numpy


def pytorch_predict_vjp(ans,theta,X,model):
    """ Do a backward pass through the PyTorch model,
    obtaining the Jacobian d pred / dtheta. 
    Must convert back to numpy array before returning 

    :param theta: model weights
    :type theta: numpy ndarray

    :param X: model features
    :type X: numpy ndarray

    :return fn: A function representing the vector Jacobian operator
    """
    def fn(v):
        # v is a vector of shape ans, the return value of mypredict()
        # return a 1D array [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2]],
        # where i is the data row index
        model.zero_gradients()
        external_grad = torch.tensor(v)
        dpred_dtheta = model.backward_pass(external_grad)
        return np.array(dpred_dtheta)
    return fn

# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(pytorch_predict,pytorch_predict_vjp)
{% endhighlight python %}

<p>
The first thing to notice is that <code class='highlight'>__init__()</code>
only takes two required parameters: <code class='highlight'>input_dim,output_dim</code>, which are the size of the inputs (number of features) and the dimension of the output layer (number of label classes, e.g.). All <code class='highlight'>__init__()</code> does is create the PyTorch model by calling the <code class='highlight'>create_model()</code> method, which the user must override in the child class. The <code class='highlight'>predict()</code> method is the only method that does not need to be overridden. It calls the function <code class='highlight'>pytorch_predict()</code>, which does the forward pass through the model. The user should not have to reimplement this function (in most cases). In order to do the forward pass, the model weights need to be set to the weights calculated in the previous step of gradient descent, which are contained in the <code class='highlight'>theta</code> parameter input to this method. How this is done is up to the user to implement in a model method they must override called <code class='highlight'>update_model_params()</code>. Note that the model parameters are encoded in <code class='highlight'>theta</code> as a numpy array, whereas the PyTorch model weights are stored as <a href="https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html">PyTorch parameters</a>, which are stored in the PyTorch model object, <code class='highlight'>self.pytorch_model</code>. Once the model weights are updated, <code class='highlight'>pytorch_predict()</code> calls <code class='highlight'>forward_pass()</code>, a model method that the user must override. This method calculates the predictions of the pytorch model (the outputs) and stores them as an attribute of the Seldonian model object. This is useful because the predictions need to be accessed when doing the backward pass. <code class='highlight'>pytorch_predict()</code> returns the model outputs after converting them to a numpy array in order to meet <a href="#requirement1">autograd requirement 1/3</a>. 
</p>

<p>
    The function <code class='highlight'>pytorch_predict_vjp()</code> specifies how to obtain the Jacobian matrix of the <code class='highlight'>pytorch_predict()</code> function, $J_{ij} = \frac{\partial F_i}{\partial \theta_j}$, where $F_i$ is the prediction for the $i$th data point and $\theta_j$ are the model parameters. The user should not have to reimplement this function in most cases. <code class='highlight'>pytorch_predict_vjp()</code> first calls <code class='highlight'>zero_gradients()</code>, a model method that the user must override. All this method must do is zero the gradients of all parameters of the pytorch model. <code class='highlight'>pytorch_predict_vjp()</code> then does a backward pass through the PyTorch model by calling <code class='highlight'>backward_pass()</code>, a model method that the user must override. <code class='highlight'>pytorch_predict_vjp()</code> is written in such a way that when autograd asks for the gradient of our <code class='highlight'>model.predict()</code> method, it will obtain the vector gradient $\frac{\partial F}{\partial \theta}$ of this function, which is exactly what we need to proceed in gradient descent. 
</p>

<h5> An example implementation</h5>
<p>

    To make this more concrete, we will create a child class that implements  univariate linear regression using a single Pytorch layer. We will use it to solve the two-constraint problem discussed in the <a href="/tutorials/simple_engine_tutorial">Getting started with the Seldonian Engine tutorial</a>. This model can be written in any file as long as it is importable at runtime. Let's write it in a file called <code>pytorch_model.py</code>:
</p>

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">
{% highlight python %}
# pytorch_model.py
### A simple single layer Pytorch model implementing linear regression 

from seldonian.models.models import SupervisedPytorchBaseModel
import torch

class PytorchLRTestModel(SupervisedPytorchBaseModel):
    def __init__(self,input_dim,output_dim):
        """ Implements linear regression using a single Pytorch linear layer

        :param input_dim: Number of features
        :param output_dim: Size of output layer (number of label columns)
        """
        super().__init__(input_dim,output_dim)
        self.has_intercept=True

    def create_model(self,**kwargs):
        """ Create the pytorch model and return it
        """
        return torch.nn.Linear(self.input_dim, self.output_dim)

    def update_model_params(self,theta,**kwargs):
        """ Update bias and weight parameters using theta

        :param theta: model weights
        :type theta: numpy ndarray
        """
        with torch.no_grad():
            self.pytorch_model.bias[0] = theta[0]
            self.pytorch_model.weight[:] = torch.tensor(theta[1:])
        return

    def zero_gradients(self):
        """ Zero out gradients of bias and weight parameters """
        if self.pytorch_model.bias.grad is not None:
            self.pytorch_model.bias.grad.zero_()
        if self.pytorch_model.weight.grad is not None:
            self.pytorch_model.weight.grad.zero_()
        return

    def forward_pass(self,X,**kwargs):
        """ Do a forward pass through the PyTorch model and return the 
        model outputs (predicted labels). The outputs should be the same shape 
        as the true labels
    
        :param X: model features
        :type X: numpy ndarray

        :return: predictions
        :rtype: torch.Tensor
        """
        X_torch = torch.tensor(X,requires_grad=True)
        predictions = self.pytorch_model(X_torch.float()).view(-1)
        return predictions

    def backward_pass(self,external_grad):
        """ Do a backward pass through the PyTorch model and return the
        (vector) gradient of the model with respect to theta as a numpy ndarray

        :param external_grad: The gradient of the model with respect to itself
            see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
            for more details
        :type external_grad: torch.Tensor 
        """
        self.predictions.backward(gradient=external_grad,retain_graph=True)
        grad = torch.cat((self.pytorch_model.bias.grad,self.pytorch_model.weight.grad.view(-1)))
        return grad
{% endhighlight python %}
</div>
<p>
    Notice that the class, <code class='highlight'>PytorchLRTestModel(SupervisedPytorchBaseModel)</code> inherits from the parent Seldonian model class, <code class='highlight'>SupervisedPytorchBaseModel</code> that we defined above. This class also overrides all of the methods of the base class except <code class='highlight'>predict</code>. These methods provide examples of how to create the model (<code class='highlight'>create_model()</code>) update the model parameters (<code class='highlight'>update_model_params()</code>), zero out the gradients (<code class='highlight'>zero_gradients()</code>), do the forward pass (<code class='highlight'>forward_pass()</code>), and do the backward pass (<code class='highlight'>backward_pass()</code>). Note that when zeroing out the gradients, it is important to zero out the gradients of <b>all</b> model parameters. If this is not done, gradients can accumulate (for example, see <a href="https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html">https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html</a>).
</p>

</p>

<p>We can now use this model to run the Seldonian algorithm presented in the <a href="/tutorials/simple_engine_tutorial">Getting started with the Seldonian Engine tutorial</a>. Let's create a file called <code>pytorch_example.py</code> which will be identical to the script used in that tutorial, except that we will replace the model with our new Seldonian PyTorch model. You may have to change the line where the <code class='highlight'>PytorchLRTestModel</code> is imported to point it to the file we wrote above. Note that we use <code class='highlight'>input_dim=1</code> and <code class='highlight'>output_dim=1</code> when we instantiate our model because we have a single feature and and single output label. </p>
{% highlight python %}
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd.extend import defvjp
from seldonian.models.pytorch_model import PytorchLRTestModel
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
    constraint_strs = ['Mean_Squared_Error >= 1.25','Mean_Squared_Error <= 2.0',]
    # confidence levels: 
    deltas = [0.1,0.1] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas)

    # 3. Define the underlying machine learning model - the Pytorch model
    model = PytorchLRTestModel(input_dim=1,output_dim=1)

    """4. Create a spec object, using some
    hidden defaults we won't worry about here
    """
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='regression',
    )
    spec.optimization_hyperparams['lambda_init'] = np.array([0.5,0.5])

    # 5. Run seldonian algorithm using the spec object
    SA = SeldonianAlgorithm(spec)
    # passed_safety,solution = SA.run(write_cs_logfile=True,debug=True)
    passed_safety,solution = SA.run()
    print(passed_safety,solution)

{% endhighlight python %}

<p>If we run this code, we see that it outputs an identical result to the one we saw when running the <a href="/tutorials/simple_engine_tutorial">Getting started with the Seldonian Engine tutorial</a>:</p>
{% highlight python %}
Iteration 0
Iteration 10
Iteration 20
Iteration 30
Iteration 40
Iteration 50
Iteration 60
Iteration 70
Iteration 80
Iteration 90
Iteration 100
Iteration 110
Iteration 120
Iteration 130
Iteration 140
Iteration 150
Iteration 160
Iteration 170
Iteration 180
Iteration 190
True [0.16911355 0.1738146 ]
{% endhighlight python %}
<p>
    The "True" indicates that it passed the safety test and the printed vector is the solution found (intercept and slope of the line). The exact solution you see may differ depending on your machine's random number generator, but the result should be very similar to the one you see when running the <a href="/tutorials/simple_engine_tutorial">Getting started with the Seldonian Engine tutorial</a>. 
</p>

<p> We hope this gives you a good place to start for defining your own Seldonian models with PyTorch. </p>

<h3 id='details'> (Optional) How to keep autograd from looking inside of a function</h3>
<p>This section gets into the nitty gritty details of how we bypass autograd's default behavior to allow PyTorch models in the library. It is not necessary to understand this section, but for those who are curious or interested in adding models from TensorFlor or other deep learning libraries, proceed at your own risk. </p>
<h5>Simple example</h5>
    <p>Let us first consider an example that does not use PyTorch. Autograd has an <a href="https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#extend-autograd-by-defining-your-own-primitives">example</a> in its tutorial that you may find helpful as well. Let's say we have a function $Q(x)$ that squares its input. To tell autograd not to look inside our function, we need to wrap the function with the <code class='highlight'>@primitive</code> decorator.
    </p> 
{% highlight python %}
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
from autograd.extend import primitive, defvjp

@primitive
def Q(x):
    return x**2
{% endhighlight python %}
    
    <p>
    Next, we write a function that defines the gradient $\frac{\partial Q(x)}{\partial x}$. 
    </p>
{% highlight python %}
def Q_vjp(ans,x):
    def fn(v):
        return 2*x
    return fn
{% endhighlight python %}
<p>
For now let's ignore why this function looks this way and tell autograd to link the function Q with its gradient function Q_vjp. This can be done with a single line:
</p>  
{% highlight python %}
defvjp(Q,Q_vjp)
{% endhighlight python %}

<p>Now let's define a function <code class='highlight'>example_func(x)</code> that calls Q and ask autograd to compute the gradient of this new function at <code class='highlight'>x=4.0</code>: 
</p>
{% highlight python %}
def example_func(x):
    return Q(x)

grad_example = grad(example_func)
x=4.0
grad_example(x)
{% endhighlight python %}

<p>If we run the above code, we get the expected result $2*x=8$. Now let's return to why <code class='highlight'>Q_vjp</code> was written in such an awkward way. It turns out that this function must return a function, the vector jacobian operator. Because our input and output were both scalars, we could kind of ignore this fact and return a function that returns the derivative, <code class='highlight'>dQ/dx = 2x</code>. When the inputs and outputs of Q are vector-valued, then we need to think more about this function and what it is actually doing. 
</p>
<h5>More realistic example</h5>
<p> Let us consider an example that is more relevant to the toolkit. Let's say we want to calculate the gradient with respect to theta of a new function, <code class='highlight'>P(theta,x)</code>, which is designed to mimic the predict function for linear regression. Here is how we might implement that:
</p>
{% highlight python %}
from autograd import jacobian
@primitive
def P(theta,x):
    return theta[0] + x @ theta[1:]

def P_vjp(ans,theta,x):
    J = np.ones((len(x),len(theta)))
    J[:,1:] = x
    def fn(v):
        return v.T @ J
    return fn

def example_func(theta,x):
    return P(theta,x)

defvjp(P,P_vjp)

grad_example = jacobian(example_func)
theta=np.array([0.0,1.0,2.0])
x = np.array([[1.0,2.0],[3.0,4.0],[5.0,6.0],[7.0,8.0]])

grad_example(theta,x)
{% endhighlight python %}

<p>
    First, notice that $P(\theta,x)$ has two inputs, both of which are vectors. The gradient of $P(\theta,x)$ with respect to theta, $\frac{\partial P_i}{\partial \theta_j}$ is actually a matrix (the Jacobian matrix) where $\frac{\partial P_i}{\partial \theta_0} = 1.0$ and $\frac{\partial P_i}{\partial \theta_{j}} = x_i$ when $j>0$. When we write: <code class='highlight'>grad_example = jacobian(example_func)</code>, there is an implicit default parameter value <code class='highlight'>argnums=0</code>, which means that we want the gradient with respect to the first argument of <code class='highlight'>example_func</code>. For the gradient with respect to the second  argument, e.g., explicitly change this parameter like: <code class='highlight'>grad_example = jacobian(example_func,argnums=1)</code>. Autograd requires the VJP function (<code class='highlight'>P_vjp</code> in our case) to return a function. Whenever we ask autograd for the gradient of $P(\theta,X)$, it calls the function that <code class='highlight'>P_vjp</code> returns, which we are calling <code class='highlight'>fn</code>. This function has a single argument, which we call <code class='highlight'>v</code>, which is a vector of length N, where N is length of <code class='highlight'>ans</code>, the return value of the original function. <code class='highlight'>fn</code> is evaluated by autograd N times, where the value of its argument, <code class='highlight'>v</code>, and the result of the computation, <code class='highlight'>v.T @ J</code> each time is:
</p> 

{% highlight python %}
i=0, v= [1. 0. 0. 0.], v.T @ J = [1. 1. 2.]
i=1, v= [0. 1. 0. 0.], v.T @ J = [1. 3. 4.]
i=2, v= [0. 0. 1. 0.], v.T @ J = [1. 5. 6.]
i=3, v= [0. 0. 0. 1.], v.T @ J = [1. 7. 8.]
{% endhighlight python %}

<p> The reason autograd does this is so that you can compute the vector Jacobian product on the fly, rather than computing the entire Jacobian matrix. This can be a lot more efficient in some cases. In our example here, we actually want the entire Jacobian matrix, so we compute it once at the top of the function <code class='highlight'>P_vjp</code> and then left multiply it by the v vector each time <code class='highlight'>fn</code> is evaluated. After autograd is done evaluating <code class='highlight'>fn</code> N times, it combines its N outputs into a single matrix. Running the above code produces the matrix:
</p>
{% highlight python %}
array([[1., 1., 2.],
       [1., 3., 4.],
       [1., 5., 6.],
       [1., 7., 8.]])
{% endhighlight python %}

<p>This is the full Jacobian matrix $\frac{\partial P_i}{\partial \theta_j}$. One last thing to notice is the signature of the VJP function: <code class='highlight'>P_vjp(ans,theta,x)</code>. <code class='highlight'>ans</code> is the return value of <code class='highlight'>P(theta,x)</code>, and it must always be the first parameter. The parameters that follow it must be the parameters of <code class='highlight'>P</code>, in the same order that they apear in <code class='highlight'>P</code>. We don't use <code class='highlight'>ans</code> in our VJP function here, but it can be useful in some cases.</p>
