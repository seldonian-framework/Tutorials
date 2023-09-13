---
layout: tutorial
permalink: /tutorials/parallelization_tutorial/
prev_url: /tutorials/dtree_tutorial/
prev_page_name: (L) Creating Fair Decision Trees and Random Forests (for binary classification)
title: Seldonian \| Tutorial B
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h2 align="center" class="mb-3">Tutorial M: Efficient parallelization with the toolkit</h2>
<hr class="my-4">
<h3> Contents </h3>
    <ul>
        <li> <a href="#overview">Why this tutorial?</a> </li>
        <li><a href="#scenario1">Scenario #1: Supervised learning using CPUs</a></li>
        <li><a href="#scenario2">Scenario #2: Supervised learning using the GPU</a></li>
        <li><a href="#scenario3">Scenario #3: Reinforcement learning using CPUs</a></li>
    </ul>
    <hr class="my-4">

<h3 id="overview"> Why this tutorial? </h3>

<p>
    A frequently asked question is how to efficiently parallelize code when running the Seldonian Engine and Experiments libraries. The answer is it depends on the application. In this short tutorial, we discuss the different scenarios where you might want to parallelize and what strategy is best for each scenario.
</p>


<h3 id="scenarios"> Parallelization scenarios  </h3>  

<h5 id="scenario1"> Scenario #1: Supervised learning using CPUs</h5>
<p> 
    This scenario is where we have a supervised learning problem that we can solve without using GPUs. This is typically the case when our dataset and model are relatively small (<1 GB), or we simply do not have access to GPUs. The Engine library is explicitly set up to run on a single CPU. However, if you monitor CPU usage on a multi-CPU machine while the engine is running, you may often see that multiple cores are in use. This is due to the implicit parallelization that NumPy performs when doing vector and matrix arithmetic and our heavy use of NumPy throughout the engine. This is fine for single engine runs (although in practice it does not actually seem to speed up runs of the engine), but it massively impedes performance when running Seldonian Experiments. <b>When running the Experiments library in this scenario, it is very important to disable NumPy's implicit parallelization </b> by including the following lines of code in your top-level script that runs the experiments before you import numpy. 
</p>

{% highlight python %}
import os
os.environ["OMP_NUM_THREADS"] = "1"
# import numpy AFTER the above line!
{% endhighlight python %}

<p>
    This will turn off NumPy's internal parallelization and dramatically speed up your experiment code, assuming you have access to multiple CPUs. We suspect the reason for this is that NumPy's internal parallelization hogs threads that would otherwise be available for parallelization in the experiment, causing extra overhead for Python's explicit multiprocessing pools that we use in the Experiments library. 
</p>

<p>
    Setting the <code class="codesnippet">n_workers</code> parameter of the <a href="https://seldonian-toolkit.github.io/Experiments/build/html/_autosummary/experiments.generate_plots.SupervisedPlotGenerator.html#experiments.generate_plots.SupervisedPlotGenerator">SupervisedPlotGenerator</a> object to a value equal to the number of CPU cores you want to use in your experiment will then allow you to run your experiment trials in parallel.
</p>

<h5 id="scenario2"> Scenario #2: Supervised learning using the GPU</h5>

<p>
    The toolkit is set up for running the engine and experiments library using a single GPU via PyTorch and Tensorflow (see <a href="{{ "/tutorials/pytorch_mnist" | relative_url}}">Tutorial G: Creating your first Seldonian PyTorch model</a>). In this case, you <b>do not</b> want to parallelize your experiments across CPUs, so keep <code class="codesnippet">n_workers=1</code>. This parameter refers to the number of CPU workers, and because the compute-intensive work is done on the GPU, sending more work to the GPU from a second CPU is just going to induce more overhead in your experiments. Whether you use NumPy's implicit parallelization (see previous scenario) does not matter significantly in this scenario. We are considering <a href="https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html">data parallelism</a> as a way to run the toolkit on multiple GPUs, but that is still a work in progress. If you are especially interested in that scenario, please open an issue or submit a pull request.  
</p>

<h5 id="scenario3"> Scenario #3: Reinforcement learning using CPUs </h5>  

<p> 
    Currently, reinforcement learning (RL) policies are only supported on the CPU. Deep RL policy base classes that run on the GPU are technically possible with PyTorch or Tensorflow, but we have not implemented them yet. If you are especially interested in a deep RL application, please open an issue or submit a pull request. As in <a href="#scenario1">Scenario #1: Supervised learning using CPUs only</a>, distributing experiments across multiple CPUs can dramatically speed up your RL experiments. As in Scenario #1, turn off NumPy's implicit parallelization to get the most efficiency when parallelizing experiments:
</p>

{% highlight python %}
import os
os.environ["OMP_NUM_THREADS"] = "1"
# import numpy AFTER the above line!
{% endhighlight python %}

<p>
    In RL experiments, the computational bottleneck may not necessarily be in running the Seldonian algorithm in each trial. Instead, it can sometimes be in the episode generation, both for creating the initial trial datasets and when evaluating a new policy's performance and safety. For this reason, we introduced <code class="codesnippet">n_workers_for_episode_generation</code>, a new key to the <code class="codesnippet">hyperparameter_and_setting_dict</code> dictionary that is an input to <a href="https://seldonian-toolkit.github.io/Experiments/build/html/_autosummary/experiments.generate_plots.RLPlotGenerator.html#experiments.generate_plots.RLPlotGenerator">RLPlotGenerator</a> class. This key controls how many CPUs are used for the episode generation steps. Note that if <code class="codesnippet">n_workers_for_episode_generation > 1</code>, then <code class="codesnippet">n_workers</code> should be kept at 1, and vice versa. Otherwise, you could have child processes forking more child processes, which massively slows down experiments. If the bottleneck in your experiments is running the Seldonian algorithm in each trial and not the episode generation, then set <code class="codesnippet">n_workers_for_episode_generation = 1</code> and set <code class="codesnippet">n_workers</code> to the number of CPUs you have available. 
</p>

</div>