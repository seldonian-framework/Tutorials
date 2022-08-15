---
layout: tutorial
permalink: /tutorials/gridworld_RL_tutorial/
prev_url: /tutorials/custom_base_node_tutorial/
prev_page_name: Custom base variables tutorial
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    
<h2 align="center" class="mb-3">Tutorial: Introduction to reinforcement learning with the Seldonian Toolkit: gridworld</h2>

<hr class="my-4">

<h3>Introduction</h3>
<p>
The Seldonian Toolkit supports offline (batch) reinforcement learning (RL) Seldonian algorithms. These are handled in much the same way as supervised learning. In the RL setting, the user must provide data, a model, a primary objective function that they want optimized, and behavioral constraints that they want enforced. The RL model comprises the agent and the environment, but both are treated as separate objects. The Seldonian Engine searches for a new policy model of the agent that simultaneously optimizes the primary objective function and satisfies the behavioral constraints with high confidence. 
</p>

<h3>Outline</h3>

<p>In this tutorial, you will learn how to:</p>

<ul>
    <li>Define an RL Seldonian algorithm using a simple gridworld environment with a tabular softmax agent </li>
    <li>Run the algorithm using the Seldonian Engine </li>
    <li>Run an RL Seldonian Experiment</li>
</ul>

<h3 id="dataset_prep"> Define the environment and agent </h3>
<p> 
The first step in setting up an RL Seldonian algorithm is to identify the environment and agent. 

<h5> Defining the environment </h5>
In this tutorial, we will consider a 3x3 gridworld environment as shown in the figure below:

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gridworld_img.png" | relative_url}}" class="img-fluid my-2" style="width: 20%"  alt="Candidate selection"> 
        <figcaption align="left"> <b>Figure 1</b> - 3x3 gridworld where the initial state is the upper left cell and the terminal state is in the bottom right cell. The possible actions are up, down, left, right. Hitting a wall, e.g. action=left in the initial state, is a valid action and returns the agent to the same state. The reward is 0 in all cells except a -1 reward in the bottom middle cell and a +1 reward when reaching the terminal state. We will use a discount factor of $gamma=0.9$ for this environment.  </figcaption>
    </figure>
</div>
</p>

<h5> Defining the agent </h5>
<p>
The agent will employ a parametrized softmax policy:
$$ \pi(s,a) = \frac{e^{p(s,a)}}{\sum_{a'}{e^{p(s,a')}}}$$
where $p(s,a)$ is a matrix of of transition probabilities for a given state, $s$, and action, $a$. At each state, the agent chooses an action by drawing from a discrete probability distribution where the probabilities of each action are given by the softmax function.
</p>

<h3>Formulate the Seldonian ML problem</h3>
<p>
Consider the offline RL problem where we want to find an optimal policy for the 3x3 gridworld environment using the agent. We could generate some episodes of data using a behavioral policy and then search for new policies using an optimizer and the importance sampling (IS) estimate as our primary objective function for evaluating the relative performance of new policies compared to the behavioral policy. In principle, this will give us a better policy, but there is no guarantee that it will. 
</p>
<p>
Let's suppose we want to enforce a constraint such that the performance of the new policy we obtain via importance sampling is at least as good as the behavioral policy, with a probability of at least 0.95. Before we can write the constraint out formally, we need to define the behavioral policy and calculate what its performance is on this environment. In this tutorial, we will use a uniform random policy as the behavioral policy. We ran 10000 episodes using the softmax agent with equal transition probabilities (i.e uniform random action probabilities) on the 3x3 gridworld environment and found an mean discounted ($gamma=0.9$) sum of rewards of $J(pi_b)=-0.25$.

  We can now write out the Seldonian ML problem:
</p>    
<p>
    Find a new policy using gradient descent with a primary objective of the unweighted IS estimate subject to the constraint:
    <ul>
        <li>$g1 = -0.25 - J\_pi\_new$, and $delta=0.05$</li>
    </ul>
    $J_pi_new$ is a new <a href="{{ "/glossary/#measure_function" | relative_url }}">measure function</a> which, when interpreted by the Engine, results in the calculation of confidence bounds on the importance sampling estimate for the new policy relative to the behavioral policy.
</p>


<h3>Creating the specification object</h3>
<p>
Our goal is to create an <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.RLSpec.html?highlight=rlspec#seldonian.spec.RLSpec">RLSpec</a> object, which will consist of everything we will need to run a Seldonian algorithm using the Engine. Creating the RLSpec object involves defining the RL environment, agent and then creating an <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.dataset.RLDataSet.html?highlight=rldataset#seldonian.dataset.RLDataSet">RLDataset</a> object, which differs in important ways from the dataset object used in the supervised learning regime. First, a dictionary called the <a href="">hyperparameter and settings dictionary</a> must be created, which contains the names of the environment and agent and details on how to generate the data. 
</p>

<h5>Defining the gridworld environment</h5>
<p>
We provide a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.environments.gridworld.Gridworld.html#seldonian.RL.environments.gridworld.Gridworld">Gridworld</a> environment as part of the Engine library, which defines a square gridworld of arbitrary size. This environment has a single parameter, <code class='highlight'>size</code>, which determines the number of grid cells on each side of the square. The default size is 3, and the reward function is already programmed to match the description in Figure 1. Any environment that is used in the Seldonian Toolkit must be written as a class and inherit from the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.environments.Environment.Environment.html#seldonian.RL.environments.Environment.Environment">Environment</a> base class we provide. The environment is set in a dictionary called the hyperparameter and settings dictionary.  
</p>

<h5>Defining the softmax agent</h5>
<p>
Like environments, agents used in the Seldonian Toolkit must be written as classes and inherit from the <a href="#">Agent</a> base class. For this tutorial, we will use the <a href="">Softmax</a> agent, which we have written as part of the library. This agent takes two parameters, an environment description (a required attribute of all Environment objects) and the hyperparameter and settings dictionary. Like the environment, the agent is set in the hyperpameter and settings dictionary.
</p>

<h5>Creating the behavioral data </h5>
The <code class='highlight'>RLdataset</code> object consists of an array of<a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.dataset.Episode.html#seldonian.dataset.Episode">Episode</a> objects generated by the behavioral policy. We will  generate these episodes using the uniform random softmax policy. 

<h3> Running the Seldonian Engine </h3>

<h3> Running a Seldonian Experiment </h3>

<h3>Summary</h3>
<p>
In this tutorial, we demonstrated ...
</p>

</div>