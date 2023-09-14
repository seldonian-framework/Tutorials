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
The Seldonian Toolkit supports offline (batch) <i>reinforcement learning</i> (RL) Seldonian algorithms. In the RL setting, the user must provide data (the observations, actions, and rewards from past episodes), a policy parameterization (similar to a <i>model</i> in the supervised learning regime), and the desired behavioral constraints. Seldonian algorithms that are implemented via the engine search for a new policy that simultaneously optimizes a primary objective function (e.g., expected discounted return) and satisfies the behavioral constraints with high confidence. This tutorial builds on the previous supervised learning tutorials, so we suggest familiarizing yourself with those, in particular <a href="{{ "/tutorials/simple_engine_tutorial" | relative_url }}">Tutorial C: Running the Seldonian Engine</a>. Note that due to the choice of confidence-bound method used in this tutorial (Student's $t$-test), the algorithms in this tutorial are technically quasi-Seldonian algorithms (QSAs).
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
    In the RL regime, an <i>agent</i> interacts sequentially with an <i>environment</i>. Time is discretized into integer time steps $t \in \{0,1,2,\dotsc\}$. At each time step, the agent makes an observation $O_t$ about the current state $S_t$ of the environment. This observation can be noisy and incomplete. The agent then selects an action $A_t$, which causes the environment to transition to the next state $S_{t+1}$ and emit a scalar (real-valued) reward $R_t$. The agent's goal is to determine which actions will cause it to obtain as much reward as possible (we formalize this statement with math below).
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
        <li>$\pi_\theta$: A <i>parameterized policy</i>, which is simply a policy with a weight vector $\theta$ that changes the conditional distribution over $A_t$ given $O_t$. That is, for all actions $a$, observations $o$, and weight vectors $\theta$, $\pi_\theta(o,a) = \Pr(A_t=a|O_t=o ; \theta)$, where "$;\theta$" indicates "given that the agent uses policy parameters $\theta$". This is not quite the same as a conditional probability because $\theta$ is not an event (one should not apply Bayes' theorem to this expression thinking that $\theta$ is an event).</li>
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
    <b>Notice that in this batch setting, Seldonian RL algorithms can be used to improve upon current and past policies. However, they can't be used to construct the first policy for an application. </b>
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="env_and_policy"> Define the environment and policy </h3>
<p> 
The first steps in setting up an RL Seldonian algorithm are to select an environment of interest and then to specify the policy parameterization the agent should use. That is, how does $\theta$ change the policy $\pi_\theta$? For example, when using a neural network as the policy, this corresponds to determining the network architecture and how the network's outputs specify the probability of each possible action.
</p>

<h5 id="environment"> Defining the environment </h5>
<p>
In this tutorial, we will consider a 3x3 gridworld environment as shown in the figure below.

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gridworld_img.png" | relative_url }}" class="img-fluid my-2" style="width: 20%" alt="Gridworld Sketch" /> 
        <figcaption align="left"> <b>Figure 1</b> – A 3x3 gridworld where the initial state ($S_0$) is the upper left cell. Episodes end when the agent reaches the bottom right cell, which we refer to as the "terminal state." This problem is fully observable, meaning that the agent observes the entire state, $O_t=S_t$, always. The possible actions are up, down, left, and right, which cause the agent to move one cell in the specified direction. Hitting a wall, e.g., action=left in the initial state, is a valid action and returns the agent to the same state, i.e., $S_{t+1}=S_t$. Each cell is labeled with the reward that the agent receives when that state is $S_{t+1}$. For example, if the agent transitions from the middle state ($S_t$) to the bottom middle state ($S_{t+1}$) then the reward would be $R_t=-1$. Notice that the reward is $0$ in all cells except for a $-1$ reward in the bottom middle cell and a $+1$ reward when reaching the terminal state. We will use a discount factor of $\gamma=0.9$ for this environment when calculating the expected return of a policy.  </figcaption>
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
    For those familiar with <i>off-policy evaluation</i> (OPE), our algorithms use off-policy estimates of the expected return based on various importance sampling estimators. These estimates are used both in the primary objective (to maximize the expected discounted return) and sometimes in safety constraints (like in this example, where the safety constraint requires the performance of the new policy to be at least $-0.25$). In this tutorial, we will use the ordinary importance sampling estimator, which provides unbiased estimates of $J(\pi_\theta)$, for any $\theta$. These unbiased estimates are used like the unbiased estimates $\hat g$ and $\hat z$ described in the tutorials for the supervised learning regime.
</p>
<p>
    In summary, we will apply a Seldonian RL algorithm to ensure that the performance of the new policy is at least the performance of the behavior policy $(-0.25)$, with probability at least $0.95$. The algorithm will use 1,000 episodes of data generated using the behavior policy (which selects each action with equal probability). We selected the performance threshold of $-0.25$ by running 10,000 additional episodes using the behavior policy and computing the average discounted return, giving $J(\pi_b)\approx -0.25$. We can now write out the Seldonian ML problem:
</p>    
<p>
    Find a new policy subject to the constraint:
    <ul>
        <li>$g1: \text{J_pi_new_IS} \geq -0.25$, and $\delta=0.05$</li>
    </ul>
    where $\text{J_pi_new_IS}$ is an RL-specific <a href="{{ "/glossary/#measure_function" | relative_url }}">measure function</a>, which means that the engine is programmed to interpret $\text{J_pi_new_IS}$ as the performance of the new policy, as evaluated using ordinary importance sampling with data generated by the behavior policy. Other importance sampling estimators can be referenced via different suffixes, such as $\text{J_pi_new_PDIS}$ for per-decision importance sampling. 
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
Our goal is to create an <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.RLSpec.html?highlight=rlspec#seldonian.spec.RLSpec">RLSpec</a> object (analogous to the <code class="codesnippet">SupervisedSpec</code> object), which will consist of everything we will need to run a Seldonian algorithm using the engine. Creating this object involves defining the behavior dataset, policy parameterization, any environment-specific parameters, and the behavioral constraints. 
</p>

<p>
In general, the manner in which the behavior data is generated is not important. In fact, generating data is not something a user will typically do using the engine. However, for this tutorial, we will generate synthetic data for reproducibility purposes. The data file we will create can be found here if you would like to skip this step: <a href="https://github.com/seldonian-toolkit/Engine/blob/main/static/datasets/RL/gridworld/gridworld_1000episodes.pkl">gridworld_1000episodes.pkl</a>. We will use the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.environments.gridworld.Gridworld.html#seldonian.RL.environments.gridworld.Gridworld">Gridworld</a> environment that is part of the Engine library. This Python class implements a generalized version of the gridworld environment described in Figure 1. It defines a square gridworld of arbitrary size, where the default size and reward function match the description in Figure 1. Therefore, we can use this environment without modification. We will use an agent that adopts a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.Agents.Parameterized_non_learning_softmax_agent.Parameterized_non_learning_softmax_agent.html#seldonian.RL.Agents.Parameterized_non_learning_softmax_agent.Parameterized_non_learning_softmax_agent">uniform random softmax</a> policy. The weights of the policy are used to generate the action probabilities as in equation \eqref{policy_parameterization}. Because there are 9 states and 4 possible actions, there are a total of 36 weights. The behavior policy sets all of these weights to zero, such that the probability of each action after any observation is $p=0.25$, i.e., uniform random.
</p>

<p> We will use the <code class='codesnippet'>run_trial()</code> in the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.RL_runner.html">RL_runner</a> module to generate the data. A trial is a set of episodes. This function takes as input a dictionary where we provide the specification of the trial, such as the number of episodes, a function to create the environment, and a function to create the agent. The reason we provide functions to create the environment and agent instead of those objects themselves is so that we can create new instances of those objects in parallel processes to speed up the data generation process. Again, this step is only necessary for generating the data, which users will either not need to do or will potentially do outside of the Engine library. 
</p>
<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">

{% highlight python %}
# generate_data.py
from functools import partial
import autograd.numpy as np
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.io_utils import save_pickle
from seldonian.RL.environments.gridworld import Gridworld
from seldonian.RL.Agents.Parameterized_non_learning_softmax_agent import Parameterized_non_learning_softmax_agent

def create_env_func():
    return Gridworld(size=3)

def create_agent_func(new_params):   
    dummy_env = Gridworld(size=3)
    env_description = dummy_env.get_env_description()
    agent = Parameterized_non_learning_softmax_agent(
        env_description=env_description,
        hyperparam_and_setting_dict={},
    )
    agent.set_new_params(new_params)
    return agent

def main():
    num_episodes = 1000
    initial_solution = np.zeros((9,4))
    
    hyperparams_and_setting_dict = {}
    hyperparams_and_setting_dict["create_env_func"] = create_env_func
    hyperparams_and_setting_dict["create_agent_func"] = partial(
        create_agent_func,
        new_params=initial_solution
    )
    hyperparams_and_setting_dict["num_episodes"] = num_episodes
    hyperparams_and_setting_dict["num_trials"] = 1
    hyperparams_and_setting_dict["vis"] = False
    episodes = run_trial(hyperparams_and_setting_dict,parallel=True,n_workers=8)

    episodes_file = f'./gridworld_{num_episodes}episodes.pkl'
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
Now we can use these episodes to create a <code class='codesnippet'>RLDataSet</code> object and complete the spec file. We will use the <code class='codesnippet'>createRLSpec()</code> function which takes as input a dataset, policy, constraint strings, deltas, and environment-specific keyword arguments. This function hides some defaults to the <code class="codesnippet">RLSpec</code> object for convenience here. The dataset will be created from the episodes we just saved in one line. We already specified our constraint strings and deltas above. 
</p>

<p>
    We create a custom policy class called <code class="codesnippet">GridworldSoftmax</code>which inherits from the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.Agents.Policies.Softmax.DiscreteSoftmax.html#seldonian.RL.Agents.Policies.Softmax.DiscreteSoftmax">DiscreteSoftmax</a> base class. The "discrete" in the name comes from the fact that the observation space and the action space are both discrete-valued, as opposed to continuous-valued. We use a custom class to i) show how to subclass a policy base class and ii) speed up the computation of action probabilities given (observation,action) pairs. The <code class="codesnippet">DiscreteSoftmax</code> base class does not assume that observations and actions are provided in 0-indexed form, so before it evaluates the action probabilities, it transforms them to 0-indexed form. In our case, we generated behavior data where the observations and actions are 0-indexed, so we can take advantage of that to speed up our code. This will be particularly helpful when we run an experiment below, which calls the engine many times. Note the signature of the method: <code class="codesnippet">get_probs_from_observations_and_actions(self,observations,actions,_)</code>. This is a method that every policy class must have, and it takes three arguments: <code class="codesnippet">observations,actions,behavior_action_probs</code>. In our case, we don't use the behavior action probabilities, so we use the placeholder <code class="codesnippet">_</code> for that parameter.
</p>

<p>
The policy requires some knowledge of the environment, which we call a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.Env_Description.Env_Description.Env_Description.html#seldonian.RL.Env_Description.Env_Description.Env_Description">Env_Description</a>. This is just a description of the observation and action spaces, which we define as objects. In our case, the only additional environment-specific piece of information is the discount factor, $\gamma=0.9$, which we specify in the <code class='codesnippet'>env_kwargs</code> argument. 
</p>


<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">
{% highlight python %}
# createSpec.py
import autograd.numpy as np
from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet,RLMetaData
from seldonian.utils.io_utils import load_pickle

class GridworldSoftmax(DiscreteSoftmax):
    def __init__(self, env_description):
        hyperparam_and_setting_dict = {}
        super().__init__(hyperparam_and_setting_dict, env_description)

    def get_probs_from_observations_and_actions(self,observations,actions,_):
        return self.softmax(self.FA.weights)[observations,actions]

    def softmax(self,x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    episodes_file = './gridworld_1000episodes.pkl'
    episodes = load_pickle(episodes_file)
    meta = RLMetaData(all_col_names=["episode_index", "O", "A", "R", "pi_b"])
    dataset = RLDataSet(episodes=episodes,meta=meta)

    # Initialize policy
    num_states = 9
    observation_space = Discrete_Space(0, num_states-1)
    action_space = Discrete_Space(0, 3)
    env_description =  Env_Description(observation_space, action_space)
    policy = GridworldSoftmax(env_description=env_description)
    env_kwargs={'gamma':0.9}
    save_dir = '.'
    constraint_strs = ['J_pi_new_IS >= -0.25']
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
</div>

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
Now that we have the spec file, running the Seldonian algorithm is simple. You may need to change the path to <code class='codesnippet'>specfile</code> if your spec file is saved in a location other than the current directory. We will also change some of the defaults of the optimization process; namely, we will set the number of iterations to $10$ and set the learning rates of $\theta$ and $\lambda$ to $0.01$.

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">
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
</div>

This should result in the following output, though the exact numbers may differ due to your machine's random number generator:
{% highlight bash %}
Safety dataset has 600 episodes
Candidate dataset has 400 episodes
Have 10 epochs and 1 batches of size 400 for a total of 10 iterations
Epoch: 0, batch iteration 0
Epoch: 1, batch iteration 0
Epoch: 2, batch iteration 0
Epoch: 3, batch iteration 0
Epoch: 4, batch iteration 0
Epoch: 5, batch iteration 0
Epoch: 6, batch iteration 0
Epoch: 7, batch iteration 0
Epoch: 8, batch iteration 0
Epoch: 9, batch iteration 0
Passed safety test!
The solution found is:
[[ 0.20093676  0.20021209 -0.20095315  0.2018718 ]
 [ 0.19958715  0.21193489 -0.21250555 -0.20173669]
 [ 0.20076256 -0.19977755  0.21200753 -0.20160111]
 [ 0.20127782  0.21320971 -0.20111709  0.19984976]
 [ 0.19655251  0.20152484 -0.20126941 -0.19943337]
 [-0.20448104 -0.2002143   0.20163945 -0.20272628]
 [ 0.20233235 -0.20099993  0.20002241  0.20128514]
 [ 0.20062167  0.20107532 -0.20115979 -0.20080426]
 [ 0.          0.          0.          0.        ]]
{% endhighlight bash %}

</p>
As we can see, the solution returned by candidate selection passed the safety test. The solution shows the weights $\theta(s,a)$ of the new policy, where the $j$th column in the $i$th row represents the $j$th action given the $i$th observation (also state, in this case) in the gridworld. The final row is all zeros because no actions are taken from the terminal state. This may not be a particularly good solution, but we have a high-probability guarantee that it is better than the uniform random policy. 
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="experiments"> Running a Seldonian Experiment </h3>
<p>
Now that we have successfully run the Seldonian algorithm once with the engine, we are ready to run a Seldonian Experiment. This will help us better understand the safety and performance of our new policies. It will also help us understand how much data we need to meet the safety and performance requirements of our problem. We recommend reading the <a href="https://seldonian-toolkit.github.io/Experiments/build/html/overview.html">Experiments overview</a> before continuing here. If you have not already installed the Experiments library, follow the instructions <a href="{{ "/tutorials/install_toolkit_tutorial/" | relative_url}}">here</a> to do so.
</p>

<p>
    Here is an outline of the experiment we will run: 
    <ul>
        <li>
            Create an array of data fractions, which will determine how much of the data to use as input to the Seldonian algorithm in each trial. We will use 10 different data fractions, which will be log-spaced between approximately 0.005 and 1.0. This array times the number of episodes in the original dataset (1,000) will make up the horizontal axis of the three plots, i.e., the "amount of data."
        </li>
        <li>
            Create 20 datasets (one for each trial) by rerunning the behavior data generation process with a different random seed. Each regenerated dataset will have the same number of episodes (1,000) as the original dataset. We use 20 trials so that we can compute uncertainties on the plotted quantities at each data fraction. 
        </li>
        <li>
            For each <code class='codesnippet'>data_frac</code> in the array of data fractions, run 20 trials. This will result in 200 total trials. In each trial:
            <ul>
                <li>
                    Use only the first <code class='codesnippet'>data_frac</code> fraction of episodes from the corresponding trial dataset to run the Seldonian algorithm using the Seldonian Engine. 
                </li>
                <li>
                    If the safety test passes, return the new policy parameters obtained via the Seldonian algorithm. If not, return "NSF" for no solution found.
                </li>
                <li>
                     If the safety test passes, generate 1,000 episodes using the new policy parameters. This is data set on which the performance and safety metrics will be calculated for this trial.
                </li>
            </ul>
        </li>
        <li>
            For each <code class='codesnippet'>data_frac</code>, calculate the mean and standard error on the performance (expected discounted return) across the 20 trials at this <code class='codesnippet'>data_frac</code>. This will be the data used for the first of the three plots. Also record how often a solution was returned and passed the safety test across the 20 trials. This fraction, referred to as the "solution rate," will be used to make the second of the three plots. Finally, for the trials that returned solutions that passed the safety test, calculate the fraction of trials for which the constraint was violated on the ground truth episodes. The fraction violated will be referred to as the "failure rate" and will make up the third and final plot. 
        </li>
    </ul>
</p>
<p>
Now, we will show how to implement the described experiment using the Experiments library. At the center of the Experiments library is the PlotGenerator class, and in our particular example, the RLPlotGenerator child class. The goal of the following script is to set up this object, use its methods to run our experiments, and then to make the three plots.
</p>

<p> 
First, the necessary imports:
{% highlight python %}
from functools import partial
import os
os.environ["OMP_NUM_THREADS"] = "1"
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import run_episode,run_episodes_par
from seldonian.RL.environments.gridworld import Gridworld
from seldonian.RL.Agents.Parameterized_non_learning_softmax_agent import Parameterized_non_learning_softmax_agent

from experiments.generate_plots import RLPlotGenerator
{% endhighlight python %}
</p>

<p>
    The line <code class="codesnippet">os.environ["OMP_NUM_THREADS"] = "1"</code> in the imports block above turns off NumPy's implicit parallelization, which we want to do when using <code class="codesnippet">n_workers>1</code> (see <a href="{{ "/tutorials/parallelization_tutorial/" | relative_url}}"> Tutorial M: Efficient parallelization with the toolkit </a> for more details).
</p>
<p>
    Now, we will set up the parameters for the experiments, such as the data fractions we want to use and how many trials at each data fraction we want to run. Each trial in an experiment is independent of all other trials, so parallelization can speed experiments up enormously. Set <code class='codesnippet'>n_workers</code> to however many CPUs you want to use. The results for each experiment we run will be saved in subdirectories of <code class='codesnippet'>results_dir</code>. Change this variable as desired. <code class='codesnippet'>n_episodes_for_eval</code> determines how many episodes are used for evaluating the performance and failure rate. 
</p>

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
    results_dir = f'results/gridworld_tutorial'
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
    n_episodes_for_eval = 1000
{% endhighlight python %}

<p>
    We will need to load the same spec object that we created for running the engine. As before, change the path to where you saved this file. We will modify some of the hyperparameters of gradient descent. This is not necessary, but we found that using these parameters resulted in better performance than the default parameters that we used to run the engine a single time. This also illustrates how to modify the spec object before running an experiment, which can be useful. 
</p>  
{% highlight python %}
    # Load spec
    specfile = './spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters'] = 40
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
{% endhighlight python %}

<p>
One of the remaining parameters of the plot generator is <code class='codesnippet'>perf_eval_fn</code>, a function that will be used to evaluate the performance in each trial. In general, this function is completely up to the user to define. During each trial, this function is called after the Seldonian algorithm has been run and only if the safety test passed for that trial. The <code class='codesnippet'>kwargs</code> passed to this function will always consist of the <code class='codesnippet'>model</code> object from the current trial and the <code class='codesnippet'>hyperparameter_and_setting_dict</code>. There is also the option to pass additional keyword arguments to this function via the <code class='codesnippet'>perf_eval_kwargs</code> parameter. 
</p>

<p>
We will write this function to generate 1000 episodes using the policy obtained by running the Seldonian algorithm and calculate the expected discounted return (with $\gamma = 0.9$). The only additional kwarg we need to provide is how many episodes to run. In this function, we create a fresh gridworld environment and agent so these are independent across trials. We set the agent's weights to the weights that were trained by the Seldonian algorithm so that the data are generated using the new policy obtained by the Seldonian algorithm. 
</p>

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
        agent = create_agent_fromdict(hyperparameter_and_setting_dict)
        env = Gridworld(size=3)
       
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

<p>
Next, we need to define the <code class='codesnippet'>hyperparameter_and_setting_dict</code> as we did in the <code>createSpec.py</code> script above. This is required for generating the datasets that we will use in each of the 20 trials, as well as for generating the data in the function we defined to evaluate the performance. Note that in this case the number of episodes that we use for generating the trial datasets and the number of episodes that we use for evaluating the performance are the same. This does not need to be the case, as we can pass a different value for <code class='codesnippet'>n_episodes_for_eval</code> via <code class='codesnippet'>perf_eval_kwargs</code> if we wanted to. 
</p>

{% highlight python %}
    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["env"] = Gridworld(size=3)
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparameter_and_setting_dict["num_episodes"] = 1000
    hyperparameter_and_setting_dict["num_trials"] = 1
    hyperparameter_and_setting_dict["vis"] = False
{% endhighlight python %}

<p>
Now we are ready to make the <code class='codesnippet'>RLPlotGenerator</code> object.

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
        plot_generator.make_plots(
            fontsize=12,
            legend_fontsize=12,
            performance_label=performance_metric,
            savename=plot_savename if save_plot else None,
            save_format="png")
{% endhighlight python %}
</p>

<p>
Here is the entire script, saved in a file called <code>generate_gridworld_plots.py</code>. We put the function definition for evaluating the performance outside of the main block in the full script for clarity. This is a slightly different place than where we presented it in the step-by-step code blocks above. 

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">
{% highlight python %}
# run_experiment.py
from functools import partial
import os
os.environ["OMP_NUM_THREADS"] = "1"
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from concurrent.futures import ProcessPoolExecutor

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import run_episode
from seldonian.RL.environments.gridworld import Gridworld
from seldonian.RL.Agents.Parameterized_non_learning_softmax_agent import Parameterized_non_learning_softmax_agent

from experiments.generate_plots import RLPlotGenerator

from createSpec import GridworldSoftmax

def create_env_func():
    return Gridworld(size=3)

def create_agent_func(new_params):   
    dummy_env = Gridworld(size=3)
    env_description = dummy_env.get_env_description()
    agent = Parameterized_non_learning_softmax_agent(
        env_description=env_description,
        hyperparam_and_setting_dict={},
    )
    agent.set_new_params(new_params)
    return agent

def generate_episodes_and_calc_J(**kwargs):
    """ Calculate the expected discounted return 
    by generating episodes

    :return: episodes, J, where episodes is the list
        of generated ground truth episodes and J is
        the expected discounted return
    :rtype: (List(Episode),float)
    """
    model = kwargs['model']
    num_episodes = kwargs['n_episodes_for_eval']
    hyperparameter_and_setting_dict = kwargs['hyperparameter_and_setting_dict']

    # Get trained model weights from running the Seldonian algo
    new_params = model.policy.get_params()

    # Create the env and agent (setting the new policy params) 
    # and run the episodes
    episodes = []
    env = create_env_func()
    agent = create_agent_func(new_params)
    for i in range(num_episodes):
        episodes.append(run_episode(agent,env))

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,gamma=0.9) for ep in episodes])
    J = np.mean(returns)
    return episodes,J

if __name__ == "__main__":
    # Parameter setup
    np.random.seed(99)
    run_experiments = True
    make_plots = True
    save_plot = False
    include_legend = True

    num_episodes = 1000 # For making trial datasets and for looking up specfile
    n_episodes_for_eval = 1000
    n_trials = 20
    data_fracs = np.logspace(-3,0,10)
    n_workers_for_episode_generation = 1
    n_workers = 6

    frac_data_in_safety = 0.6
    verbose=True
    results_dir = f'results/gridworld_{num_episodes}episodes_{n_trials}trials'
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'gridworld_experiment.png')
    # Load spec
    specfile = f'./spec.pkl'
    spec = load_pickle(specfile)

    perf_eval_fn = generate_episodes_and_calc_J
    perf_eval_kwargs = {
        'n_episodes_for_eval':n_episodes_for_eval,
        'env_kwargs':spec.model.env_kwargs,
    }
    initial_solution = np.zeros((9,4)) # theta values
    # The setup for generating behavior data for the experiment trials.
    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["create_env_func"] = create_env_func
    hyperparameter_and_setting_dict["create_agent_func"] = partial(
        create_agent_func,
        new_params=initial_solution,
    )
    hyperparameter_and_setting_dict["num_episodes"] = num_episodes 
    hyperparameter_and_setting_dict["n_workers_for_episode_generation"] = n_workers_for_episode_generation
    hyperparameter_and_setting_dict["num_trials"] = 1 # Leave as 1 - it is not the same "trial" as experiment trial.
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
        plot_generator.make_plots(fontsize=18,
            performance_label=r"$J(\pi_{\mathrm{new}})$",
            include_legend=include_legend,
            save_format="png",
            title_fontsize=18,
            legend_fontsize=16,
            custom_title=r"$J(\pi_{\mathrm{new}}) \geq -0.25$ (vanilla gridworld 3x3)",
            savename=plot_savename if save_plot else None)
{% endhighlight python %}
</div>

</p>

<p>
Running the script should produce a plot that looks very similar to the one below. Once the experiments are run, if you want to save the plot but not rerun the experiments simply set: <code class='codesnippet'>run_experiments = False</code> and <code class='codesnippet'>save_plot = True</code>.

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gridworld_20trials.png" | relative_url}}" class="img-fluid mt-4" style="width: 65%"  alt="Disparate impact log loss"> 
        <figcaption align="left"> <b>Figure 2</b> - The three plots of a Seldonian Experiment shown for the gridworld environment with a softmax agent. A behavioral constraint, $\text{J_pi_new_IS} \geq -0.25$, is enforced with $\delta = 0.05$. Each panel shows the mean (point) and standard error (shaded region) over 20 trials of a quantity for the quasi-Seldonian algorithm (QSA, blue), plotted against the number of training samples as determined from the data fraction array. (Left) the performance of the new policy evaluated on the ground truth dataset. (Middle) the fraction of trials at each data fraction that returned a solution. (Right) the fraction of trials that violated the safety constraint on the ground truth dataset. The black dashed line is set at the $\delta=0.05$ value that we set in our behavioral constraint. </figcaption>
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