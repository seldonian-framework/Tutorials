---
layout: tutorial
permalink: /tutorials/gridworld_RL_tutorial/
prev_url: /tutorials/custom_base_variable_tutorial/
prev_page_name: Custom base variables tutorial
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    
<h2 align="center" class="mb-3">Tutorial: Introduction to reinforcement learning with the Seldonian Toolkit: gridworld</h2>

<hr class="my-4">

<h3>Introduction</h3>
<p>
The Seldonian Toolkit supports offline (batch) <i>reinforcement learning</i> (RL) Seldonian algorithms. In the RL setting, the user must provide data (the observations, actions, and rewards from past episodes), a policy parameterization (similar to a <i>model</i> in the supervised learning regime), and the desired behavioral constraints. Seldonian algorithms implemented via the Engine search for a new policy that simultaneously optimizes a primary objective function (expected discounted return) and satisfies the behavioral constraints with high confidence. In this tutorial, we will use an environment and agent that are built into the Engine library. They serve as examples to help you to build your own RL Seldonian algorithms. Note that due to the choice of confidence bound method used in this tutorial (Student's $t$-test), the algorithms in this tutorial are technically quasi-Seldonian algorithms (QSAs).
</p>

<h3>Outline</h3>

<p>In this tutorial, after reviewing RL notation and fundamentals, you will learn how to:</p>

<ul>
    <li>Define a RL Seldonian algorithm using a simple gridworld environment with a tabular softmax agent </li>
    <li>Run the algorithm using the Seldonian Engine to find a policy that satisfies a behavioral constraint</li>
    <li>Run a RL Seldonian Experiment</li>
</ul>

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
        <li>$\pi_\theta$: A <i>parameterized policy</i>, which is simply a policy with a weight vector $\theta$ that changes the conditional distribution over $A_t$ given $O_t$. That is, for all actions $a$, observations $o$, and weight vectors $\theta$, $\pi_\theta(o,a) = \Pr(A_t=a|O_t=o ; \theta)$, where $;\theta$ indicates "given that the agent uses policy parameters $\theta$". This is not quite the same as a conditional probability because $\theta$ is not an event (one should not apply Bayes' Theorem to this expression thinking that $\theta$ is an event).</li>
        <li>$\gamma$: The reward discount parameter. This is a problem-specific contant in $[0,1]$ that is used in the objective function, defined next.</li>
        <li>$J$: The objective function. The default in the toolkit is the expected discounted return: $J(\pi)=\mathbf{E}\left [ \sum_{t=0}^\infty \gamma^t R_t\right ]$. The agent's goal is to find a policy that maximizes this objective function.</li>
        <li>The agent's experiences can be broken into statistically independent <i>episodes</i>. Each episode begins at time $t=0$. When an episode ends, the agent no longer needs to select actions and no longer receives rewards (this can be modeled using a <i>terminal absorbing state</i> as defined on page 18 <a href="https://people.cs.umass.edu/~pthomas/courses/CMPSCI_687_Fall2020/687_F20.pdf">here</a>.</li>
    </ul>
</p>
<p>
    At this time, this toolkit is only compatible with <i>batch</i> or <i>offline</i> RL. This means that some current policy (typically called the <i>behavior policy</i> $\pi_b$) is already in use to select actions. The available data corresponds to the observations, actions, and rewards from previous episodes wherein the current policy was used. To make this more formal, we will define a <i>history</i> $H$ to be the available historical information from one episode:
    $$ H = (O_0, A_0, R_0, O_1, A_1, R_1, \dotsc).$$
    All of the available data can then be expressed as $D = (H_i)_{i=1}^m$, where $m$ is the number of episodes for which data is available. 
</p>
<p>
    For generality, the toolkit does not require there to be just one current policy. Instead, the available data could have been generated by several different (past) policies. However, the default Seldonian RL algorithms in the toolkit require some knowledge about what the past policies were. Specifically, they require each action $A_t$ in the historical data to also come with the probability that the behavior policy (the policy that selected the action) would select that action: $\pi_b(O_t,A_t)$. Hence, we ammend the definition of a history $H$ to include this information:
    $$ H = (O_0, A_0, \pi_b(O_0,A_0), R_0, O_1, A_1, \pi_b(O_1,A_1) R_1, \dotsc).$$
    Notice that in this batch setting, Seldonian RL algorithms can be used to improve upon current and past policies. However, they can't be used to construct the first policy for an application. 
</p>

<h3 id="dataset_prep"> Define the environment and policy </h3>
<p> 
The first steps in setting up an RL Seldonian algorithm are the select an environment of interest and then to specify the policy parameterization the agent should use. That is, how does $\theta$ change the policy $\pi_\theta$? For example, when using a neural network, this corresponds to determining the network architecture and how the network's outputs specify the probability of each possible action.
</p>

<h5> Defining the environment </h5>
<p>
In this tutorial, we will consider a 3x3 gridworld environment as shown in the figure below.

<div align="center">
    <figure>
        <img src="/Tutorials/assets/img/gridworld_img.png" class="img-fluid my-2" style="width: 20%" alt="Gridworld Sketch" /> 
        <figcaption align="left"> <b>Figure 1</b> - 3x3 gridworld where the initial state ($S_0$) is the upper left cell. Episodes end when the agent reaches the bottom right cell, which we refer to as the "terminal state". This problem is fully observable, meaning that the agent observes the entire state: $O_t=S_t$ always. The possible actions are up, down, left, right, which cause the agent to move one cell in the specified direction. Hitting a wall, e.g., action=left in the initial state, is a valid action and returns the agent to the same state. Each cell is labeled with the reward that the agent receives when that state is $S_{t+1}$. For example, if the agent transitions from the middle state ($S_t$) to the bottom middle state ($S_{t+1}$) then the reward would be $R_t=-1$. Notice that the reward is $0$ in all cells except for a $-1$ reward in the bottom middle cell and a $+1$ reward when reaching the terminal state. We will use a discount factor of $\gamma=0.9$ for this environment when calculating the expected return of a policy.  </figcaption>
    </figure>
</div>
</p>

<h5> Defining the policy </h5>
<p>
The toolkit is compatible with a wide range of possible policy parameterizations, and even allows you to introduce your own environment-specific policy representation. For this example, we will use a tabular softmax policy. This policy stores one policy parameter (weight) for every possible observation-action pair. Let $\theta_{o,a}$ be the parameter for observation $o$ and action $a$. A tabular softmax policy can then be expressed as:
$$ \pi_\theta(o,a) = \frac{e^{\theta_{o,a}}}{\sum_{a'}{e^{\theta_{o,a'}}}}.$$
So, given the current observation $O_t$, the agent chooses an action by drawing from a discrete probability distribution where the probability of each action $a$ is $\pi(O_t,a)$.
</p>

<h3>Formulate the Seldonian ML problem</h3>
<p>
    Consider the offline RL problem of finding a policy for the 3x3 gridworld that has the largest expected return possible (primary objective) subject to the safety constraint that the expected return is at least $-0.25$ (this might be the performance of the current policy). In this tutorial we simulate this process, generating many episodes of data using a behavior policy (current policy) and then feeding this data to our Seldonian algorithm with the tabular softmax policy parameterization. We include a behavioral constraint that requires the performance of the new policy to be at least $-0.25$ with probability at least $0.95$. In later tutorials we show how safety constraints can be defined in terms of additional reward functions (as in constrained MDPs).
</p>
<p>
    For those familiar with <i>off-policy evaluation</i> (OPE), our algorithms use off-policy estimates of the expected return based on the per-decision importance sampling estimator. These estimates are used both in the primary objective (maximize the expected discounted return) and sometimes in safety constraints (like in this example, where the safety constraint requires the performance of the new policy to be at least $-0.25$). The per-decision importance sampling estimator provides unbiased estimates of $J(\pi_\theta)$, for any $\theta$, which are used like the unbiased estimates $\hat g$ and $\hat z$ described in the tutorials for the supervised learning regime.
</p>
<p>
    In summary, we will apply a Seldonian RL algorithm to ensure that the performance of the new policy is at least the performance of the behavior policy $(-0.25)$, with probability at least $0.95$. The algorithm will use 10,000 epsiodes of data generated using the behavior policy (which selects each action with equal probability). We selected the performance threshold of $-0.25$ by running 10,000 additional episodes using the behavior policy and computing the average discounted return, giving $J(\pi_b)\approx -0.25$. We can now write out the Seldonian ML problem:
</p>    
<p>
    Find a new policy subject to the constraint:
    <ul>
        <li>$g1: J\_{\text{pi_new}} \geq -0.25$, and $\delta=0.05$</li>
    </ul>
    where $J\_{\text{pi_new}}$ is an RL-specific <a href="/Tutorials/glossary/#measure_function">measure function</a>. The Engine is programmed to know that $J\_{\text{pi_new}}$ means the performance of the new policy. The performance is calculated using an importance sampling estimate. 
</p>

<h5> From supervised to reinforcemnt learning</h5>
<p>
    From the toolkit's perspective, there are few differences between the supervised learning and reinforcement learning regimes. The algorithm overview remains unchanged: the Seldonian algorithm still has candidate selection and safety test modules. It still uses gradient descent using the KKT conditions to find the candidate solutions, and uses statistical tools like Student's $t$-test to compute confidence intervals. Though there are some differences in the code (as shown in the example below), the core conceptual differences are:
    <ul>
        <li>The parameterized "model" is replaced with a parameterized "policy". However, this is mostly just a terminology change (both take an input and provide a distribution over possible outputs).</li>
        <li>Though both still use a parse tree to represent the behavioral constraints, the base variables (leaf nodes of the parse tree that are not constants) use <i>importance sampling</i> variants to estimate what would happen if a new policy were to be used. More precisely, they provides unbiased estimates of the expected discounted return of the new policy, and this expected discounted return can use additional rewards defined to capture the safety of the new policy. The importance sampling estimates correspond to the $\hat z$ functions described in the first tutorial.</li>
    </ul>
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
    constraint_strs = ['J_pi_new >= -0.25']
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
Now that we have the spec file, running the Seldonian algorithm is extremely simple. You may need to change the path to <code class='highlight'>specfile</code> if your spec file is saved in a location other than the current directory.:
{% highlight python %}
# run_gridworld.py 
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
    # load specfile
    specfile = './spec.pkl'
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
</p>
<p>
Now we will show how to implement the described experiment using the Experiments library. At the center of the Experiments library is the PlotGenerator class, and in our particular example the RLPlotGenerator child class. The goal of the following script is to setup this object, use its methods to run our experiments, and then to make the three plots.
</p>

<p> 
First, the necessary imports:
{% highlight python %}
import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.generate_plots import RLPlotGenerator

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import run_trial_given_agent_and_env
{% endhighlight python %}
</p>

<p>
Now we will set up the parameters for the experiments, such as the data fractions we want to use and how many trials at each data fraction we want to run. Each trial in an experiment is independent of all other trials, so parallelization can speed experiments up enormously. Set <code class='highlight'>n_workers</code> to however many CPUs you want to use. The results for each experiment we run will be saved in subdirectories of results_dir. <code class='highlight'>n_episodes_for_eval</code> determines how many episodes are used for evaluating the performance and failure rate.

{% highlight python %}
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
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
    n_episodes_for_eval = 1000
{% endhighlight python %}
</p>

<p>
Now we will need to load the same spec object that we created for running the Engine. As before, change the path to the where you saved this file. We will modify some of the hyperparameters of gradient descent. This is not necessary, but we found that using these parameters resulted in better performance than the default parameters that we used to run the Engine a single time above. This also illustrates one way to modify the spec object before running an experiment, which can be useful.   
{% highlight python %}
    # Load spec
    specfile = f'./spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters'] = 40
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
{% endhighlight python %}
</p>

<p>
We need to define the function we will use to evaluate the performance. This function will generate 1000 new episodes using the new policy and evaluate the performance by calculating the expected discounted return ($\gamma = 0.9$). We also define the keyword arguments to pass to this function in each trial. 

{% highlight python %}
    def perf_eval_fn(model,**kwargs):
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
    perf_eval_kwargs = {'n_episodes':n_episodes_for_eval}   
{% endhighlight python %}
</p>

<p>
We also need to define a <code class='highlight'>hyperparameter_and_setting_dict</code> as we did in the <code>createSpec.py</code> script above. This is required for generating the datasets that we will use in each of the 20 trials. Each of these needs to have 1000 episodes.

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
Here is the entire script, saved in a file called <code>generate_gridworld_plots.py</code>
{% highlight python %}
# generate_gridworld_plots.py
import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

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
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
    n_episodes_for_eval = 1000
    # Load spec
    specfile = f'../interface_outputs/gridworld_james/spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters'] = 40
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
    spec.optimization_hyperparams['beta_velocity'] = 0.9
    spec.optimization_hyperparams['beta_rmspropr'] = 0.95

    perf_eval_fn = generate_episodes_and_calc_J
    perf_eval_kwargs = {'n_episodes':n_episodes_for_eval}

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
Running the script should produce a plot that looks very similar to this:

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gridworld_20trials.png" | relative_url}}" class="img-fluid mt-4" style="width: 65%"  alt="Disparate impact log loss"> 
        <figcaption align="left"> <b>Figure 2</b> - The Three Plots of a Seldonian Experiment shown for the gridworld environment with a softmax agent. A behavioral constraint, $J\_{\text{pi_new}} \geq -0.25$, is enforced with $\delta = 0.05$. Each panel shows the mean (point) and standard error (shaded region) over 20 trials of a quantity for the Quasi-Seldonian model (QSA, blue), plotted against the number of training samples as determined from the data fraction array. (Left) the performance of the new policy evaluated on the ground truth dataset. (Middle) the fraction of trials at each data fraction that returned a solution. (Right) the fraction of trials that violated the safety constraint on the ground truth dataset. The black dashed line is set at the $\delta=0.05$ value that we set in our behavioral constraint. </figcaption>
    </figure>
</div>
The performance of the obtained policy increases steadily with increasing number of episodes. The QSA does not always return a solution for small amounts of data, but at $\sim10^3$ episodes it returns a solution every time it is run. This is desired behavior because for small amounts of data, the uncertainty about whether the solution is safe is too large for the algorithm to guarantee safety. The QSA fails only for very small amounts of data ($\sim5$ episodes), which can happen because it is quasi-Seldonian, i.e., the method used to calculate the confidence bound on the behavioral constraint is not appropriate for very small amounts of data. Given more than 5 episodes, the QSA never violates the behavioral constraint on the ground truth dataset.
</p>
<h3>Summary</h3>
<p>
In this tutorial, we demonstrated how to run a quasi-Seldonian reinforcement learning algorithm with the Seldonian Toolkit. We defined the Seldonian machine learning problem for an simple gridworld problem with a softmax agent. Running the algorithm using the Seldonian Engine, we found that the solution we obtained passed the safety test and was therefore deemed safe. To explore the behavior of the algorithm in more detail, we ran a Seldonian Experiment. We produced the Three Plots: performance, solution rate, and failure rate as a function of number of episodes used to run the algorithm. The softmax agent's performance is improved as more episodes are used. The algorithm needs close to 1000 episodes before it can return a solution every time, 
</p>

</div>