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
The Seldonian Toolkit supports offline (batch) reinforcement learning (RL) Seldonian algorithms. In the RL setting, the user must provide data, a model, and behavioral constraints that they want enforced. The RL model is composed of the agent and the environment, each of which must be provided by the user. The agent contains a policy which dictates how to choose actions in any given state. Seldonian algorithms implemented via the Engine search for a new policy of the agent that simultaneously optimizes a primary objective function and satisfies the behavioral constraints with high confidence. 
</p>

<h3>Outline</h3>

<p>In this tutorial, you will learn how to:</p>

<ul>
    <li>Define a RL Seldonian algorithm using a simple gridworld environment with a tabular softmax agent </li>
    <li>Run the algorithm using the Seldonian Engine to find a policy that satisfies a behavioral constraint</li>
    <li>Run a RL Seldonian Experiment</li>
</ul>

<h3 id="dataset_prep"> Define the environment and agent </h3>
<p> 
The first step in setting up an RL Seldonian algorithm is to identify the environment and agent. 
</p>

<h5> Defining the environment </h5>
<p>
In this tutorial, we will consider a 3x3 gridworld environment as shown in the figure below:

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gridworld_img.png" | relative_url}}" class="img-fluid my-2" style="width: 20%"  alt="Candidate selection"> 
        <figcaption align="left"> <b>Figure 1</b> - 3x3 gridworld where the initial state is the upper left cell and the terminal state is in the bottom right cell. The possible actions are up, down, left, right. Hitting a wall, e.g., action=left in the initial state, is a valid action and returns the agent to the same state. The reward is 0 in all cells except a -1 reward in the bottom middle cell and a +1 reward when reaching the terminal state. We will use a discount factor of $\gamma=0.9$ for this environment when calculating the expected return of a policy.  </figcaption>
    </figure>
</div>
</p>

<h5> Defining the agent </h5>
<p>
The agent will employ a parametrized softmax policy:
$$ \pi(s,a) = \frac{e^{p(s,a)}}{\sum_{a'}{e^{p(s,a')}}}$$
where $p(s,a)$ is a matrix of transition probabilities for a given state, $s$, and action, $a$. At each state, the agent chooses an action by drawing from a discrete probability distribution where the probabilities of each action are given by the above softmax function.
</p>

<h3>Formulate the Seldonian ML problem</h3>
<p>
Consider the offline RL problem where we want to find an optimal policy for the 3x3 gridworld environment using the agent. We could generate some episodes of data using a behavioral policy and then search for new policies using an optimizer and the importance sampling (IS) estimate as our primary objective function for evaluating the relative performance of new policies compared to the behavioral policy. In principle, this will give us a better policy, but there is no guarantee that it will. 
</p>
<p>
Let's suppose we want to enforce a constraint such that the performance of the new policy we obtain via importance sampling <i>must be</i> at least as good as the behavioral policy, with a probability of at least 0.95. Before we can write the constraint out formally, we need to define the behavioral policy and calculate what its performance is on this environment. In this tutorial, we will use a uniform random policy as the behavioral policy. We ran 10000 episodes using the softmax agent with equal transition probabilities (i.e uniform random action probabilities) on the 3x3 gridworld environment and found a mean discounted ($\gamma=0.9$) sum of rewards of $J(pi_b)=-0.25$.

  We can now write out the Seldonian ML problem:
</p>    
<p>
    Find a new policy using gradient descent with a primary objective of the unweighted IS estimate subject to the constraint:
    <ul>
        <li>$g1: J\_{\text{pi_new}} \geq -0.25$, and $\delta=0.05$</li>
    </ul>
    where $J\_{\text{pi_new}}$ is an RL-specific <a href="{{ "/glossary/#measure_function" | relative_url }}">measure function</a>. The Engine is programmed to know that $J\_{\text{pi_new}}$ means the performance of the new policy. The performance is calculated using the unweighted IS estimate. 
</p>


<h3>Creating the specification object</h3>
<p>
Our goal is to create an <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.RLSpec.html?highlight=rlspec#seldonian.spec.RLSpec">RLSpec</a> object, which will consist of everything we will need to run a Seldonian algorithm using the Engine. Creating this object involves defining the RL environment, agent, the dataset and the behavioral constraints. 
</p>
<p>
We provide a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.environments.gridworld.Gridworld.html#seldonian.RL.environments.gridworld.Gridworld">Gridworld</a> environment as part of the Engine library, which defines a square gridworld of arbitrary size. This environment has a single parameter, <code class='highlight'>size</code>, which determines the number of grid cells on each side of the square. The default size is 3, and the reward function is already programmed to match the description in Figure 1, so we can use it as is. We also provide a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.Agents.Parameterized_non_learning_softmax_agent.Parameterized_non_learning_softmax_agent.html#seldonian.RL.Agents.Parameterized_non_learning_softmax_agent.Parameterized_non_learning_softmax_agent">Parameterized_non_learning_softmax_agent</a>, which inacts the softmax policy described above.
</p>

<p>
All RL problems using the Engine require a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.dataset.RLDataSet.html#seldonian.dataset.RLDataSet">RLdataset</a> object, which consists of an array of <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.dataset.Episode.html#seldonian.dataset.Episode">Episode</a> objects. Episode objects consist of arrays of states, actions, rewards and action probabilities, where each array entry is for one timestep in the episode. Episodes need to be pre-generated by the behavioral policy, which we stated above will be the uniform random softmax policy. When initialized, the softmax agent by default carries out a uniform random policy, i.e., it chooses actions up, right, down, left with equal probability regardless of state. 
</p>
<p>
In the code below, we will use the <code class='highlight'>run_trial()</code> function in the <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.RL.RL_runner.html">RL_runner</a> module to generate 1000 episodes using the behavioral policy and then instantiate an <code class='highlight'>RLDataSet</code> from those episodes. We will then define our constraints and pass in the metadata for this gridworld environment to a wrapper function that creates the RLSpec object called <code class='highlight'>createRLspec()</code>.

<div>

<input type="button" style="float: right" class="btn btn-sm btn-secondary" onclick="copy2Clipboard(this)" value="Copy code snippet">

{% highlight python %}
# createSpec.py
from seldonian.RL.RL_runner import run_trial
from seldonian.spec import createRLspec
from seldonian.dataset import RLDataSet

hyperparams_and_setting_dict = {}
hyperparams_and_setting_dict["env"] = "gridworld"
hyperparams_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
hyperparams_and_setting_dict["num_episodes"] = 1000
hyperparams_and_setting_dict["num_trials"] = 1

def main():
    episodes, agent = run_trial(hyperparameter_and_setting_dict)
    dataset = RLDataSet(episodes=episodes)

    metadata_pth = "../../static/datasets/RL/gridworld/gridworld_metadata.json"
    save_dir = '.'
    constraint_strs = ['-0.25 - J_pi_new']
    deltas=[0.05]

    createRLspec(
        dataset=dataset,
        metadata_pth=metadata_pth,
        agent=agent,
        constraint_strs=constraint_strs,
        deltas=deltas,
        save_dir=save_dir,
        verbose=True)

if __name__ == '__main__':
    main()
{% endhighlight python %}
</div>

Notice that in the <code class='highlight'>hyperparams_and_setting_dict</code> dictionary which we passed to <code class='highlight'>run_trial()</code>, we specified the string names of the environment and agent. <code class='highlight'>run_trial()</code> knows how to map these string names to the objects describing the environment and agent. Here are the contents of the metadata file that we passed to <code class='highlight'>createRLspec()</code>:

{% highlight python %}
# gridworld_metadata.json
{
   "columns":["episode_index","O","A","R","pi"],
    "regime":"RL",
    "RL_module_name":"gridworld",
    "RL_class_name":"Gridworld"
}
{% endhighlight python %}

The <code class='highlight'>"RL_module_name":"gridworld"</code> points to the module  <code>seldonian.RL.environments.gridworld</code>. The <code class='highlight'>"RL_class_name":"Gridworld"</code> points to the class <code class='highlight'>Gridworld()</code> in that module. 
</p>


<p> 
Running that file from the command line like:
{% highlight bash %}
$ python createSpec.py
{% endhighlight bash %}
creates a file called <code>spec.pkl</code> in whatever directory you ran the command. Once that file is created, you are ready to run the Seldonian algorithm. 
</p>

<h3> Running the Seldonian Engine </h3>
<p>
Now that we have the spec file, running the Seldonian algorithm is extremely simple:
{% highlight python %}
# run_gridworld.py 
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
    # load specfile
    specfile = 'spec.pkl'
    spec = load_pickle(specfile)
    # Run Seldonian algorithm 
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run()
    if passed_safety:
        print("Passed safety test")
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
Iteration 10
Iteration 20
Passed safety test!
The solution found is:
[[-0.17718385  0.1704365  -0.17040032  0.17088498]
 [ 0.17059895  0.17041713 -0.17023933 -0.17075099]
 [ 0.17071368 -0.17149423  0.17053143 -0.16999813]
 [ 0.17053054 -0.17239301 -0.1705741   0.18006215]
 [ 0.17048531  0.17024612 -0.17034173 -0.17001209]
 [-0.17303155  0.17141605  0.17056875 -0.17039872]
 [ 0.17040837 -0.1702631   0.17070513  0.16976562]
 [ 0.17046631  0.17002752 -0.16996262 -0.17031659]
 [ 0.          0.          0.          0.        ]]
{% endhighlight bash %}

</p>
As we can see, the solution returned by candidate selection passed the safety test. The solution is the Q-table, $p(s,a)$ of the new policy, where the jth column in the ith row represents the jth action given the ith state in the gridworld. The final row is all zeros because no actions are taken from the terminal state. 
<h3> Running a Seldonian Experiment </h3>
<p>
Now that we have successfully run the Seldonian algorithm once with the Engine, we are ready to run a Seldonian Experiment. This will help us better understand the safety and performance of our new policy. It will also help us understand how much data we need to meet the safety and performance requirements of our problem. The <a href="https://seldonian-toolkit.github.io/Experiments"> Seldonian Experiments library</a> was designed to help implement Seldonian Experiments. We recommend reading the <a href="https://seldonian-toolkit.github.io/Experiments/build/html/overview.html">Experiments overview</a> before continuing here. If you have not already installed the Experiments library, follow the instructions <a href="{{ "/tutorials/install_toolkit_tutorial/" | relative_url}}">here</a> to do so.
</p>

<p>
    Here is an outline of the experiment we will run: 
    <ul>
        <li>Create an array of data fractions, which will determine how much of the data to use as input to the Seldonian algorithm in each trial. We will use 10 different data fractions, which will be log-spaced between aproximately 0.005 and 1.0. This array times the number of epsiodes in the original dataset (1000) will make up the horizontal axis of the three plots.</li>
        <li>Create 20 datasets (one for each trial) by re-running the behavioral data generation process with a different random seed. Each regenerated dataset will have the same number of episodes (1000) as the original dataset. We use 20 trials so that we can compute uncertainties on the plotted quantities at each data fraction. <b>We will use yet another re-generated dataset as the ground truth dataset</b> for calculating the performance and safety metrics.</li>
        <li>
        For each <code class='highlight'>data_frac</code> in the array of data fractions, run 20 trials. In each trial, use only the first <code class='highlight'>data_frac</code> fraction of episodes to run the Seldonian algorithm using the Seldonian Engine. We will use the same spec file we used above for each run of the Engine, where only the <code class='highlight'>dataset</code> parameter will be modified for each trial. This will generate 15x20=300 total runs of the Seldonian algorithm. Each run will consist of a different set of new policy parameters, or "NSF" if no solution was found. </li>
        <li>For each <code class='highlight'>data_frac</code>, if a solution was returned that passed the safety test, calculate the mean and standard error on the performance (expected discounted return) across the 20 trials at this <code class='highlight'>data_frac</code> using the new policy parameters evaluated on the ground truth episodes. This will be the data used for the first of the three plots. Also record how often a solution was returned and passed the safety test across the 20 trials. This fraction, referred to as the "solution rate", will be used to make the second of the three plots. Finally, for the trials that returned solutions that passed the safety test, calculate the fraction of trials for which the constraint was violated on the ground truth episodes. The fraction violated will be referred to as the "failure rate" and will make up the third and final plot. </li>
    </ul>

Here is the code to run this experiment, saved in a file called <code>generate_gridworld_plots.py</code>
{% highlight python %}
# generate_gridworld_plots.py
import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.utils import generate_resampled_datasets
from experiments.generate_plots import RLPlotGenerator

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma

from seldonian.RL.RL_runner import run_trial_given_agent_and_env

def generate_episodes_and_calc_J(model,**kwargs):
    """ Calculate the expected return of the sum 
    of discounted rewards by generating episodes
    
    :param model: The RL_model object containing
        the environment and agent 
    
    :return: episodes, J, where episodes is the list
        of generated ground truth episodes and J is
        the expected discounted return
    :rtype: (List(Episode),float)
    """
    agent = model.agent
    env = model.env
    num_episodes = kwargs['n_episodes']
    # generate episodes
    episodes = run_trial_given_agent_and_env(
        agent=agent,
        env=env,
        num_episodes=num_episodes)

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,env.gamma) for ep in episodes])
    J = np.mean(returns)
    return episodes,J
    

if __name__ == "__main__":
    # Parameter setup
    run_experiments = Ture
    make_plots = True
    save_plot = False
    performance_metric = 'J(pi_new)'
    n_trials = 20
    data_fracs = np.logspace(-2.3,0,10)
    n_workers = 8
    verbose=True
    results_dir = f'results/gridworld_debug_{n_trials}trials'
    plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
    n_episodes_for_eval = 1000
    # Load spec
    specfile = f'../interface_outputs/gridworld_james/spec.pkl'
    spec = load_pickle(specfile)

    spec.use_builtin_primary_gradient_fn = False

    os.makedirs(results_dir,exist_ok=True)
    
    perf_eval_fn = generate_episodes_and_calc_J
    perf_eval_kwargs = {'n_episodes':n_episodes_for_eval}

    spec.optimization_hyperparams['num_iters'] = 40
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
    spec.optimization_hyperparams['beta_velocity'] = 0.9
    spec.optimization_hyperparams['beta_rmspropr'] = 0.95

    hyperparameter_and_setting_dict = {}

    hyperparameter_and_setting_dict["env"] = "gridworld"
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"

    hyperparameter_and_setting_dict["num_episodes"] = n_episodes_for_eval
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
        constraint_eval_fns=[],
        results_dir=results_dir,
        perf_eval_kwargs=perf_eval_kwargs,
        )
    if run_experiments:
        plot_generator.run_seldonian_experiment(verbose=verbose)

    if save_plot:
        plot_generator.make_plots(fontsize=12,legend_fontsize=8,
            performance_label=performance_metric,
            savename=plot_savename)
    else:
        plot_generator.make_plots(fontsize=12,legend_fontsize=8,
            performance_label=performance_metric,)
{% endhighlight python %}
</p>

<h3>Summary</h3>
<p>
In this tutorial, we demonstrated ...
</p>

</div>