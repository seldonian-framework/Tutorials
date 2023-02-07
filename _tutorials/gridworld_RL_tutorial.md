---
layout: tutorial
permalink: /tutorials/gridworld_RL_tutorial/
prev_url: /tutorials/pytorch_mnist/
prev_page_name: (G) Creating your first Seldonian PyTorch model
next_url: /tutorials/custom_base_variable_tutorial/
next_page_name: (I) Creating custom base variables in behavioral constraints
title: Seldonian \| Tutorial H
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    
<h2 align="center" class="mb-3">Tutorial H: Introduction to reinforcement learning with the Seldonian Toolkit</h2>

<hr class="my-4">

<h3> Contents </h3>
    <ul>
        <li> <a href="#intro">Introduction</a> </li>
        <li> <a href="#outline">Outline</a> </li>
        <li> <a href="#background">RL background</a> </li>
        <li> <a href="#env_and_policy">Define the environment and policy</a> </li>
            <ul>
                <li><a href="#environment">Defining the environment</a></li>
                <li><a href="#policy">Defining the policy</a></li>
            </ul>
        <li> <a href="#formulate">Formulate the Seldonian ML problem</a> </li>
            <ul> <li> <a href="#supervised_to_rl">From supervised learning to reinforcement learning</a> </li>
            </ul>
        <li> <a href="#spec_object">Creating the specification object</a> </li>
        <li> <a href="#running_the_engine">Running the Seldonian Engine</a> </li>
        <li> <a href="#experiments">Running a Seldonian Experiment</a> </li>
        <li> <a href="#summary">Summary</a> </li>
    </ul>
    <hr class="my-4">

<h3 id="intro">Introduction</h3>
<p>
The Seldonian Toolkit supports offline (batch) <i>reinforcement learning</i> (RL) Seldonian algorithms. In the RL setting, the user must provide data (the observations, actions, and rewards from past episodes), a policy parameterization (similar to a <i>model</i> in the supervised learning regime), and the desired behavioral constraints. Seldonian algorithms that are implemented via the engine search for a new policy that simultaneously optimizes a primary objective function (e.g., expected discounted return) and satisfies the behavioral constraints with high confidence. This tutorial builds on the previous supervised learning tutorials, so we suggest familiarizing yourself with those, in particular the <a href="{{ "/tutorials/simple_engine_tutorial" | relative_url }}">Getting started with the Seldonian Engine tutorial</a>. Note that due to the choice of confidence-bound method used in this tutorial (Student's $t$-test), the algorithms in this tutorial are technically quasi-Seldonian algorithms (QSAs).
</p>

<h3 id="outline">Outline</h3>

<p>In this tutorial, after reviewing RL notation and fundamentals, you will learn how to:</p>

<ul>
    <li>Define a RL Seldonian algorithm using data generated from a gridworld environment and a tabular softmax policy </li>
    <li>Run the algorithm using the Seldonian Engine to find a policy that satisfies a behavioral constraint</li>
    <li>Run a RL Seldonian Experiment for this Seldonian algorithm</li>
</ul>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="background">RL background</h3>
<p>
    In the RL regime, an <i>agent</i> interacts sequentially with an <i>environment</i>. Time is discretized into integer time steps $t \in \{0,1,2,\dotsc\}$. At each time, the agent makes an observation $O_t$ about the current state $S_t$ of the environment. This observation can be noisy and incomplete. The agent then selects an action $A_t$, which causes the environment to transition to the next state $S_{t+1}$ and emit a scalar (real-valued) reward $R_t$. The agent's goal is to determine which actions will cause it to obtain as much reward as possible (we formalize this statement with math below).
</p>
<p>
    Before continuing, we establish our notation for the RL regime:
    <ul>
        <li>$t$: The current time step, starting at $t=0$.</li>
        <li>$S_t$: The state of the environment at time $t$.</li>
        <li>$O_t$: The agent's observation at time $t$. For many problems, the agent observes the entire state $S_t$. This can be accomplished by simply defining $O_t=S_t$.</li>
        <li>$A_t$: The action chosen by the agent at time $t$ (the agent selects $A_t$ after observing $O_t$).</li>
        <li>$R_t$: The reward received by the agent at time $t$ (the agent receives $R_t$ after the environment transitions to state $S_{t+1}$ due to action $A_t$).</li>
        <li>$\pi$: A <i>policy</i>, where $\pi(o,a)=\Pr(A_t=a|O_t=o)$ for all actions $a$ and observations $o$.</li>
        <li>$\pi_\theta$: A <i>parameterized policy</i>, which is simply a policy with a weight vector $\theta$ that changes the conditional distribution over $A_t$ given $O_t$. That is, for all actions $a$, observations $o$, and weight vectors $\theta$, $\pi_\theta(o,a) = \Pr(A_t=a|O_t=o ; \theta)$, where $;\theta$ indicates "given that the agent uses policy parameters $\theta$". This is not quite the same as a conditional probability because $\theta$ is not an event (one should not apply Bayes' theorem to this expression thinking that $\theta$ is an event).</li>
        <li>$\gamma$: The reward discount parameter. This is a problem-specific constant in $[0,1]$ that is used in the objective function, defined next.</li>
        <li>$J$: The objective function. The default in the toolkit is the expected discounted return: $J(\pi)=\mathbf{E}\left [ \sum_{t=0}^\infty \gamma^t R_t\right ]$. The agent's goal is to find a policy that maximizes this objective function.</li>
        <li>The agent's experiences can be broken into statistically independent <i>episodes</i>. Each episode begins at time $t=0$. When an episode ends, the agent no longer needs to select actions and no longer receives rewards (this can be modeled using a <i>terminal absorbing state</i> as defined on page 18 <a href="https://people.cs.umass.edu/~pthomas/courses/CMPSCI_687_Fall2020/687_F20.pdf">here</a>).</li>
    </ul>
</p>
<p>
    At this time, the Seldonian Toolkit is only compatible with <i>batch</i> or <i>offline</i> RL. This means that some current policy (typically called the <i>behavior policy</i> $\pi_b$) is already in use to select actions. The available data corresponds to the observations, actions, and rewards from previous episodes wherein the current policy was used. To make this more formal, we will define a <i>history</i> $H$ to be the available historical information from one episode:
    $$ H = (O_0, A_0, R_0, O_1, A_1, R_1, \dotsc).$$
    All of the available data can then be expressed as $D = (H_i)_{i=1}^m$, where $m$ is the number of episodes for which data is available. 
</p>
<p>
    For generality, the toolkit does not require there to be just one current policy. Instead, the available data could have been generated by several different (past) policies. However, the default Seldonian RL algorithms in the toolkit require some knowledge about what the past policies were. Specifically, they require each action $A_t$ in the historical data to also come with the probability that the behavior policy (the policy that selected the action) would select that action: $\pi_b(O_t,A_t)$. Hence, we amend the definition of a history $H$ to include this information:
    $$ H = (O_0, A_0, \pi_b(O_0,A_0), R_0, O_1, A_1, \pi_b(O_1,A_1) R_1, \dotsc).$$
    Notice that in this batch setting, Seldonian RL algorithms can be used to improve upon current and past policies. However, they can't be used to construct the first policy for an application. 
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="env_and_policy"> Define the environment and policy </h3>
<p> 
The first steps in setting up an RL Seldonian algorithm are to select an environment of interest and then to specify the policy parameterization the agent should use. That is, how does $\theta$ change the policy $\pi_\theta$? For example, when using a neural network, this corresponds to determining the network architecture and how the network's outputs specify the probability of each possible action.
</p>

<h5 id="environment"> Defining the environment </h5>
<p>
In this tutorial, we will consider a 3x3 gridworld environment as shown in the figure below.

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gridworld_img.png" | relative_url }}" class="img-fluid my-2" style="width: 20%" alt="Gridworld Sketch" /> 
        <figcaption align="left"> <b>Figure 1</b> – A 3x3 gridworld where the initial state ($S_0$) is the upper left cell. Episodes end when the agent reaches the bottom right cell, which we refer to as the "terminal state." This problem is fully observable, meaning that the agent observes the entire state, $O_t=S_t$, always. The possible actions are up, down, left, and right, which cause the agent to move one cell in the specified direction. Hitting a wall, e.g., action=left in the initial state, is a valid action and returns the agent to the same state. Each cell is labeled with the reward that the agent receives when that state is $S_{t+1}$. For example, if the agent transitions from the middle state ($S_t$) to the bottom middle state ($S_{t+1}$) then the reward would be $R_t=-1$. Notice that the reward is $0$ in all cells except for a $-1$ reward in the bottom middle cell and a $+1$ reward when reaching the terminal state. We will use a discount factor of $\gamma=0.9$ for this environment when calculating the expected return of a policy.  </figcaption>
    </figure>
</div>
</p>

<h5 id="policy"> Defining the policy </h5>
<p>
The toolkit is compatible with a wide range of possible policy parameterizations, and even allows you to introduce your own environment-specific policy representation. For this example, we will use a tabular softmax policy. This policy stores one policy parameter (weight) for every possible observation-action pair. Let $\theta_{o,a}$ be the parameter for observation $o$ and action $a$. A tabular softmax policy can then be expressed as:
$$ \begin{equation}
\pi_\theta(o,a) = \frac{e^{\theta_{o,a}}}{\sum_{a'}{e^{\theta_{o,a'}}}}
\label{policy_parameterization}
\end{equation}
$$
So, given the current observation $O_t$, the agent chooses an action by drawing from a discrete probability distribution where the probability of each action $a$ is $\pi(O_t,a)$.
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="formulate">Formulate the Seldonian ML problem</h3>
<p>
    Consider the offline RL problem of finding a policy for the 3x3 gridworld that has the largest expected return possible (primary objective) subject to the safety constraint that the expected return is at least $-0.25$ (this might be the performance of the current policy). In this tutorial, we simulate this process, generating many episodes of data using a behavior policy (current policy) and then feeding this data to our Seldonian algorithm with the tabular softmax policy parameterization. We include a behavioral constraint that requires the performance of the new policy to be at least $-0.25$ with probability at least $0.95$. In later tutorials, we show how safety constraints can be defined in terms of additional reward functions (as in constrained Markov decision processes [MDPs]).
</p>
<p>
    For those familiar with <i>off-policy evaluation</i> (OPE), our algorithms use off-policy estimates of the expected return based on the per-decision importance sampling estimator. These estimates are used both in the primary objective (maximize the expected discounted return) and sometimes in safety constraints (like in this example, where the safety constraint requires the performance of the new policy to be at least $-0.25$). The per-decision importance sampling estimator provides unbiased estimates of $J(\pi_\theta)$, for any $\theta$, which are used like the unbiased estimates $\hat g$ and $\hat z$ described in the tutorials for the supervised learning regime.
</p>
<p>
    In summary, we will apply a Seldonian RL algorithm to ensure that the performance of the new policy is at least the performance of the behavior policy $(-0.25)$, with probability at least $0.95$. The algorithm will use 1,000 episodes of data generated using the behavior policy (which selects each action with equal probability). We selected the performance threshold of $-0.25$ by running 10,000 additional episodes using the behavior policy and computing the average discounted return, giving $J(\pi_b)\approx -0.25$. We can now write out the Seldonian ML problem:
</p>    
<p>
    Find a new policy subject to the constraint:
    <ul>
        <li>$g1: J\_{\text{pi_new}} \geq -0.25$, and $\delta=0.05$</li>
    </ul>
    where $J\_{\text{pi_new}}$ is an RL-specific <a href="/Tutorials/glossary/#measure_function">measure function</a>, which means that the engine is programmed to interpret $J\_{\text{pi_new}}$ as the performance of the new policy. The performance of the new policy is calculated using per-decision importance sampling with data generated by the behavior policy. 
</p>

<h5 id="supervised_to_rl"> From supervised learning to reinforcement learning</h5>
<p>
    From the toolkit's perspective, there are few differences between the supervised learning and reinforcement learning regimes. The algorithm overview remains unchanged: the Seldonian algorithm still has candidate selection and safety test modules. It still uses gradient descent using the KKT conditions (or black box optimization) to find the candidate solutions and uses statistical tools like Student's $t$-test to compute confidence intervals. Though there are some differences in the code (as shown in the example below), the core conceptual differences are:
    <ul>
        <li>The parameterized "model" is replaced with a parameterized "policy." However, this is mostly just a terminology change (both take an input and provide a distribution over possible outputs).</li>
        <li>Though both still use a parse tree to represent the behavioral constraints, in the RL regime the base variables (leaf nodes of the parse tree that are not constants) use <i>importance sampling</i> variants to estimate what would happen if a new policy were to be used. More precisely, they provide unbiased estimates of the expected discounted return of the new policy, and this expected discounted return can use additional rewards defined to capture the safety of the new policy. The importance sampling estimates correspond to the $\hat z$ functions described in the <a href="{{ "/tutorials/alg_details_tutorial/#parse_tree" | relative_url }}">Algorithm details tutorial</a>.</li>
        <li>The data consist of $m$ episodes in the RL regime as opposed to $m$ data points in the supervised learning regime. </li>
        <li>In the Experiments library, the default way of generating ground truth data in the RL regime is to run additional episodes using a behavior policy. We will see this played out in the <a href="#experiments">Running a Seldonian Experiment</a> section.</li>
    </ul>
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="spec_object">Creating the specification object</h3>
<p>
Our goal is to create an <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.RLSpec.html?highlight=rlspec#seldonian.spec.RLSpec">RLSpec</a> object, which will consist of everything we will need to run a Seldonian algorithm using the engine. Creating this object involves defining the behavior dataset, policy parameterization, any environment-specific parameters, and the behavioral constraints. 
</p>

<p>
In general, the manner in which the behavior data is generated is not important. In fact, generating data is not something a user will typically do using the engine. However, for this tutorial, we will generate synthetic data for reproducibility purposes. The data file we will create can be found here if you would like to skip this step: <a href="https://github.com/seldonian-toolkit/Engine/blob/main/static/datasets/RL/gridworld/gridworld_1000episodes.pkl">gridworld_1000episodes.pkl</a>. We will use the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.environments.gridworld.Gridworld.html#seldonian.RL.environments.gridworld.Gridworld">Gridworld</a> environment that is part of the Engine library. This environment defines a square gridworld of arbitrary size. The default size is 3 cells on a side, and the reward function is already programmed to match the description in Figure 1, so we can use this environment without modification. We will use an agent that adopts a uniform random <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.Agents.Policies.Softmax.html#module-seldonian.RL.Agents.Policies.Softmax">DiscreteSoftmax</a> policy. The weights of the policy are used to generate the action probabilities as in equation \eqref{policy_parameterization}. Because there are 9 states and 4 possible actions, there are a total of 36 weights. The behavior policy sets all of these weights to zero, such that the probability of each action after any observation is $p=0.25$, i.e., uniform random.
</p>

<p> We will use the <code class='highlight'>run_trial()</code> in the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.RL_runner.html">RL_runner</a> module to generate the data. A trial is a set of episodes. This function takes as input a dictionary where we provide the specification of the trial, such as the number of episodes, the environment, and the agent. Again, this step is only necessary for generating the data, which users will likely do outside of the Engine library. 
</p>
<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">

{% highlight python %}
# generate_data.py
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.io_utils import save_pickle

trial_dict = {}
trial_dict["env"] = "gridworld"
trial_dict["agent"] = "Parameterized_non_learning_softmax_agent"
trial_dict["num_episodes"] = 1000
trial_dict["num_trials"] = 1
trial_dict["vis"] = False

def main():
    episodes, agent = run_trial(trial_dict)
    episodes_file = './gridworld_1000episodes.pkl'
    save_pickle(episodes_file,episodes)

if __name__ == '__main__':
    main()
{% endhighlight python %}
</div>

<p>
Save this script as a file called <code>generate_data.py</code> and run it from the command line like 

{% highlight bash %}
$ python generate_data.py
{% endhighlight bash %}

This saves a file called <code>gridworld_1000episodes.pkl</code> in the current directory which contains the 1000 episodes.
</p>

<p>
Now we can use these episodes to create a <code class='highlight'>RLDataSet</code> object and complete the spec file. We will use the <code class="highlight">createRLSpec()</code> function which takes as input a dataset, policy, constraint strings, deltas, and environment-specific keyword arguments. The dataset will be created from the episodes we just saved in one line. We already specified our constraint strings and deltas above. The policy requires some knowledge of the environment, which we call a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.Env_Description.Env_Description.Env_Description.html#seldonian.RL.Env_Description.Env_Description.Env_Description">Env_Description</a>. This is just a description of the observation and action spaces. We provide these as objects. In our case, the only additional environment-specific piece of information is the discount factor, $\gamma=0.9$, which we specify in the <code class='highlight'>env_kwargs</code> argument. 
</p>

{% highlight python %}
# createSpec.py
from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet
from seldonian.utils.io_utils import load_pickle

def main():
    episodes_file = './gridworld_1000episodes.pkl'
    episodes = load_pickle(episodes_file)
    dataset = RLDataSet(episodes=episodes)

    # Initialize policy
    num_states = 9
    observation_space = Discrete_Space(0, num_states-1)
    action_space = Discrete_Space(0, 3)
    env_description =  Env_Description(observation_space, action_space)
    policy = DiscreteSoftmax(hyperparam_and_setting_dict={},
        env_description=env_description)
    env_kwargs={'gamma':0.9}
    save_dir = '.'
    constraint_strs = ['J_pi_new >= -0.25']
    deltas=[0.05]

    spec = createRLSpec(
        dataset=dataset,
        policy=policy,
        constraint_strs=constraint_strs,
        deltas=deltas,
        env_kwargs=env_kwargs,
        save=True,
        save_dir='.',
        verbose=True)

if __name__ == '__main__':
    main()


{% endhighlight python %}

<p> 
Saving this script to a file called <code>createSpec.py</code> and then running it the command line like:
{% highlight bash %}
$ python createSpec.py
{% endhighlight bash %}
will create a file called <code>spec.pkl</code> in whatever directory you ran the command. Once that file is created, you are ready to run the Seldonian algorithm. 
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="running_the_engine"> Running the Seldonian Engine </h3>
<p>
Now that we have the spec file, running the Seldonian algorithm is simple. You may need to change the path to <code class='highlight'>specfile</code> if your spec file is saved in a location other than the current directory. We will also change some of the defaults of the optimization process; namely, we will set the number of iterations to $10$ and set the learning rates of $\theta$ and $\lambda$ to $0.01$.
{% highlight python %}
# run_gridworld.py 
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
    # load specfile
    specfile = './spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters']=10
    spec.optimization_hyperparams['alpha_theta']=0.01
    spec.optimization_hyperparams['alpha_lamb']=0.01
    # Run Seldonian algorithm 
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run()
    if passed_safety:
        print("Passed safety test!")
        print("The solution found is:")
        print(solution)
    else:
        print("No Solution Found")
{% endhighlight python %}

This should result in the following output, though the exact numbers may differ due to your machine's random number generator:
{% highlight bash %}
Safety dataset has 600 episodes
Candidate dataset has 400 episodes
Iteration 0
Passed safety test!
The solution found is:
[[ 0.19907136  0.20260221 -0.20081473  0.2004336 ]
 [-0.20323448  0.20115298 -0.2023812  -0.19957809]
 [-0.20665602  0.19071499  0.20247835 -0.20167761]
 [ 0.20084656  0.21050098 -0.20152061 -0.20083289]
 [ 0.20156723  0.2012198  -0.20136459  0.20180457]
 [-0.20693545 -0.20328714  0.20207675 -0.20123736]
 [ 0.20047478 -0.20036864 -0.20053801  0.19877662]
 [ 0.20421366  0.20038838 -0.20056068  0.21222363]
 [ 0.          0.          0.          0.        ]]

{% endhighlight bash %}

</p>
As we can see, the solution returned by candidate selection passed the safety test. The solution shows the weights $\theta(s,a)$ of the new policy, where the $j$th column in the $i$th row represents the $j$th action given the $i$th observation (also state, in this case) in the gridworld. The final row is all zeros because no actions are taken from the terminal state. This may not be a particularly good solution, but we have a high-probability guarantee that it is better than the uniform random policy. 
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="experiments"> Running a Seldonian Experiment </h3>
<p>
Now that we have successfully run the Seldonian algorithm once with the engine, we are ready to run a Seldonian Experiment. This will help us better understand the safety and performance of our new policy. It will also help us understand how much data we need to meet the safety and performance requirements of our problem. We recommend reading the <a href="https://seldonian-toolkit.github.io/Experiments/build/html/overview.html">Experiments overview</a> before continuing here. If you have not already installed the Experiments library, follow the instructions <a href="{{ "/tutorials/install_toolkit_tutorial/" | relative_url}}">here</a> to do so.
</p>

<p>
    Here is an outline of the experiment we will run: 
    <ul>
        <li>The performance metric we will use is the expected discounted return.</li>
        <li>Create an array of data fractions, which will determine how much of the data to use as input to the Seldonian algorithm in each trial. We will use 10 different data fractions, which will be log-spaced between approximately 0.005 and 1.0. This array times the number of episodes in the original dataset (1,000) will make up the horizontal axis of the three plots, i.e., the "number of training samples."</li>
        <li>Create 20 datasets (one for each trial) by rerunning the behavior data generation process with a different random seed. Each regenerated dataset will have the same number of episodes (1,000) as the original dataset. We use 20 trials so that we can compute uncertainties on the plotted quantities at each data fraction. <b>We will use a dataset of 1,000 episodes generated using the new policy as the ground truth dataset</b> when calculating the performance and safety metrics for each trial. </li>
        <li>
        For each <code class='highlight'>data_frac</code> in the array of data fractions, run 20 trials. In each trial, use only the first <code class='highlight'>data_frac</code> fraction of episodes to run the Seldonian algorithm using the Seldonian Engine. We will use the same spec file we used above for each run of the engine, where only the <code class='highlight'>dataset</code> parameter will be modified for each trial. This will generate 15x20=300 total runs of the Seldonian algorithm. Each run will consist of a different set of new policy parameters, or "NSF" if no solution was found. </li>
        <li>For each <code class='highlight'>data_frac</code>, calculate the mean and standard error on the performance (expected discounted return) across the 20 trials at this <code class='highlight'>data_frac</code>. This will be the data used for the first of the three plots. Also record how often a solution was returned and passed the safety test across the 20 trials. This fraction, referred to as the "solution rate," will be used to make the second of the three plots. Finally, for the trials that returned solutions that passed the safety test, calculate the fraction of trials for which the constraint was violated on the ground truth episodes. The fraction violated will be referred to as the "failure rate" and will make up the third and final plot. </li>
    </ul>
</p>
<p>
Now, we will show how to implement the described experiment using the Experiments library. At the center of the Experiments library is the PlotGenerator class, and in our particular example, the RLPlotGenerator child class. The goal of the following script is to set up this object, use its methods to run our experiments, and then to make the three plots.
</p>

<p> 
First, the necessary imports:
{% highlight python %}
import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.generate_plots import RLPlotGenerator

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import (create_env,
    create_agent,run_trial_given_agent_and_env)
{% endhighlight python %}
</p>

<p>
Now, we will set up the parameters for the experiments, such as the data fractions we want to use and how many trials at each data fraction we want to run. Each trial in an experiment is independent of all other trials, so parallelization can speed experiments up enormously. Set <code class='highlight'>n_workers</code> to however many CPUs you want to use. The results for each experiment we run will be saved in subdirectories of <code class='highlight'>results_dir</code>. Change this variable as desired. <code class='highlight'>n_episodes_for_eval</code> determines how many episodes are used for evaluating the performance and failure rate. 

{% highlight python %}
if __name__ == "__main__":
    # Parameter setup
    run_experiments = True
    make_plots = True
    save_plot = False
    performance_metric = 'J(pi_new)'
    n_trials = 20
    data_fracs = np.logspace(-2.3,0,10)
    n_workers = 8
    verbose=True
    results_dir = f'results/gridworld_2022Sep09_{n_trials}trials'
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
    n_episodes_for_eval = 1000
{% endhighlight python %}
</p>

<p>
We will need to load the same spec object that we created for running the engine. As before, change the path to where you saved this file. We will modify some of the hyperparameters of gradient descent. This is not necessary, but we found that using these parameters resulted in better performance than the default parameters that we used to run the engine a single time. This also illustrates how to modify the spec object before running an experiment, which can be useful.   
{% highlight python %}
    # Load spec
    specfile = './spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters'] = 40
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
{% endhighlight python %}
</p>

<p>
One of the remaining parameters of the plot generator is <code class="highlight">perf_eval_fn</code>, a function that will be used to evaluate the performance in each trial. In general, this function is completely up to the user to define. During each trial, this function is called after the Seldonian algorithm has been run and only if the safety test passed for that trial. The <code class='highlight'>kwargs</code> passed to this function will always consist of the <code class="highlight">model</code> object from the current trial and the <code class='highlight'>hyperparameter_and_setting_dict</code>. There is also the option to pass additional keyword arguments to this function via the <code class='highlight'>perf_eval_kwargs</code> parameter. In our case, we will define this function to generate 1000 episodes using the policy obtained by running the Seldonian algorithm and calculate the expected discounted return ($\gamma = 0.9$). The only additional argument we need to provide is how many episodes to run. This function uses the same <code class='highlight'>run_trial</code> function that we used when generating the original dataset. We set the agent's weights to the weights that were trained by the Seldonian algorithm so that the data are generated using the new policy obtained by the Seldonian algorithm. 

{% highlight python %}
    def generate_episodes_and_calc_J(**kwargs):
        """ Calculate the expected discounted return 
        by generating episodes

        :return: episodes, J, where episodes is the list
            of generated ground truth episodes and J is
            the expected discounted return
        :rtype: (List(Episode),float)
        """
        # Get trained model weights from running the Seldonian algo
        model = kwargs['model']
        new_params = model.policy.get_params()
       
        # create env and agent
        hyperparameter_and_setting_dict = kwargs['hyperparameter_and_setting_dict']
        agent = create_agent(hyperparameter_and_setting_dict)
        env = create_env(hyperparameter_and_setting_dict)
       
        # set agent's weights to the trained model weights
        agent.set_new_params(new_params)
        
        # generate episodes
        num_episodes = kwargs['n_episodes_for_eval']
        episodes = run_trial_given_agent_and_env(
            agent=agent,env=env,num_episodes=num_episodes)

        # Calculate J, the discounted sum of rewards
        returns = np.array([weighted_sum_gamma(ep.rewards,env.gamma) for ep in episodes])
        J = np.mean(returns)
        return episodes,J
    perf_eval_kwargs = {'n_episodes_for_eval':n_episodes_for_eval}   
{% endhighlight python %}
</p>

<p>
Next, we need to define the <code class='highlight'>hyperparameter_and_setting_dict</code> as we did in the <code>createSpec.py</code> script above. This is required for generating the datasets that we will use in each of the 20 trials, as well as for generating the data in the function we defined to evaluate the performance. Note that in this case the number of episodes that we use for generating the trial datasets and the number of episodes that we use for evaluating the performance are the same. This does not need to be the case, as we can pass a different value for <code class="highlight">n_episodes_for_eval</code> via <code class="highlight">perf_eval_kwargs</code> if we wanted to. 

{% highlight python %}
    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["env"] = "gridworld"
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparameter_and_setting_dict["num_episodes"] = 1000
    hyperparameter_and_setting_dict["num_trials"] = 1
    hyperparameter_and_setting_dict["vis"] = False
{% endhighlight python %}
</p>

<p>
Now we are ready to make the <code class='highlight'>RLPlotGenerator</code> object.

{% highlight python %}
    plot_generator = RLPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='generate_episodes',
        hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
        perf_eval_fn=perf_eval_fn,
        perf_eval_kwargs=perf_eval_kwargs,
        results_dir=results_dir,
        )
{% endhighlight python %}
</p>

<p>
To run the experiment, we simply run the following code block:
{% highlight python %}
    if run_experiments:
        plot_generator.run_seldonian_experiment(verbose=verbose)
{% endhighlight python %}
</p>

<p>
After the experiment is done running, we can make the three plots, which is done in the following code block:
{% highlight python %}
    if make_plots:
        if save_plot:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,
                savename=plot_savename)
        else:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,)
{% endhighlight python %}
</p>

<p>
Here is the entire script, saved in a file called <code>generate_gridworld_plots.py</code>. We put the function definition for evaluating the performance outside of the main block in the full script for clarity. This is a slightly different place than where we presented it in the step-by-step code blocks above. 

{% highlight python %}
# generate_gridworld_plots.py
import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.generate_plots import RLPlotGenerator

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import (create_env,
    create_agent,run_trial_given_agent_and_env)

def generate_episodes_and_calc_J(**kwargs):
    """ Calculate the expected discounted return 
    by generating episodes

    :return: episodes, J, where episodes is the list
        of generated ground truth episodes and J is
        the expected discounted return
    :rtype: (List(Episode),float)
    """
    # Get trained model weights from running the Seldonian algo
    model = kwargs['model']
    new_params = model.policy.get_params()
   
    # create env and agent
    hyperparameter_and_setting_dict = kwargs['hyperparameter_and_setting_dict']
    agent = create_agent(hyperparameter_and_setting_dict)
    env = create_env(hyperparameter_and_setting_dict)
   
    # set agent's weights to the trained model weights
    agent.set_new_params(new_params)
    
    # generate episodes
    num_episodes = kwargs['n_episodes_for_eval']
    episodes = run_trial_given_agent_and_env(
        agent=agent,env=env,num_episodes=num_episodes)

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,env.gamma) for ep in episodes])
    J = np.mean(returns)
    return episodes,J
    
if __name__ == "__main__":
    # Parameter setup
    run_experiments = True
    make_plots = True
    save_plot = False
    performance_metric = 'J(pi_new)'
    n_trials = 20
    data_fracs = np.logspace(-2.3,0,10)
    n_workers = 8
    verbose=True
    results_dir = f'results/gridworld_2022Sep09_{n_trials}trials'
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
    n_episodes_for_eval = 1000
    # Load spec
    specfile = f'../engine-repo/examples/gridworld_tutorial/spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters'] = 40
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
    spec.optimization_hyperparams['beta_velocity'] = 0.9
    spec.optimization_hyperparams['beta_rmspropr'] = 0.95

    perf_eval_fn = generate_episodes_and_calc_J
    perf_eval_kwargs = {'n_episodes_for_eval':n_episodes_for_eval}

    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["env"] = "gridworld"
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparameter_and_setting_dict["num_episodes"] = 1000
    hyperparameter_and_setting_dict["num_trials"] = 1
    hyperparameter_and_setting_dict["vis"] = False

    plot_generator = RLPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='generate_episodes',
        hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
        perf_eval_fn=perf_eval_fn,
        perf_eval_kwargs=perf_eval_kwargs,
        results_dir=results_dir,
        )
    if run_experiments:
        plot_generator.run_seldonian_experiment(verbose=verbose)

    if make_plots:
        if save_plot:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,
                savename=plot_savename)
        else:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,)

{% endhighlight python %}
</p>

<p>
Running the script should produce a plot that looks very similar to the one below. Once the experiments are run, if you want to save the plot but not rerun the experiments simply set: <code class="highlight">run_experiments = False</code> and <code class="highlight">save_plot = True</code>.

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gridworld_20trials.png" | relative_url}}" class="img-fluid mt-4" style="width: 65%"  alt="Disparate impact log loss"> 
        <figcaption align="left"> <b>Figure 2</b> - The three plots of a Seldonian Experiment shown for the gridworld environment with a softmax agent. A behavioral constraint, $J\_{\text{pi_new}} \geq -0.25$, is enforced with $\delta = 0.05$. Each panel shows the mean (point) and standard error (shaded region) over 20 trials of a quantity for the quasi-Seldonian algorithm (QSA, blue), plotted against the number of training samples as determined from the data fraction array. (Left) the performance of the new policy evaluated on the ground truth dataset. (Middle) the fraction of trials at each data fraction that returned a solution. (Right) the fraction of trials that violated the safety constraint on the ground truth dataset. The black dashed line is set at the $\delta=0.05$ value that we set in our behavioral constraint. </figcaption>
    </figure>
</div>
The performance of the obtained policy increases steadily with increasing number of episodes provided to the QSA. The QSA does not always return a solution for small amounts of data, but at $\sim10^3$ episodes it returns a solution every time it is run. This is desired behavior because for small amounts of data, the uncertainty about whether the solution is safe is too large for the algorithm to guarantee safety. The QSA always produces safe behavior (failure rate = 0). For very small amounts of data ($\lesssim10$ episodes), the algorithm may have a non-zero failure rate in your case. It can even have a failure rate above $\delta$ because the algorithm we are using here is quasi-Seldonian, i.e., the method used to calculate the confidence bound on the behavioral constraint makes assumptions that are only reasonable for relatively large amounts of data. 
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="summary">Summary</h3>
<p>
In this tutorial, we demonstrated how to run a quasi-Seldonian offline reinforcement learning algorithm with the Seldonian Toolkit. We defined the Seldonian machine learning problem in the context of a simple gridworld environment with a softmax policy parameterization. The behavioral constraint we set out to enforce was that the performance of the new policy must be at least as good as a uniform random policy. Running the algorithm using the Seldonian Engine, we found that the solution we obtained passed the safety test and was therefore deemed safe. To explore the behavior of the algorithm in more detail, we ran a Seldonian Experiment. We produced the three plots, performance, solution rate, and failure rate as a function of the number of episodes used to run the algorithm. We found the expected behavior of the algorithm: the performance of the obtained policy parameterization improved as more episodes were used. When small amounts of data were provided to the algorithm, the uncertainty in the safety constraint was large, preventing the algorithm from guaranteeing safety. With enough data, though, the algorithm returned a solution it guaranteed to be safe. Importantly, the algorithm never returned a solution it deemed to be safe that actually was not safe according to a ground truth dataset.
</p>

</div>