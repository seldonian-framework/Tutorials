---
layout: home
permalink: /coming_soon/
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h5 class="mb-3"><b>Coming soon</b></h5>
    <hr class="my-4">
   
    <h5>Major features planned for Spring 2023 release, in order of priority:</h5>
    <ul>
        <li>
        <a href="https://en.wikipedia.org/wiki/Hoeffding%27s_inequality">Hoeffding's</a> concentration inequality bounding method. This will enable running true Seldonian algorithms (as opposed to quasi-Seldonian algorithms) with the Seldonian Engine.
        </li>
         <li>
        Multi-class classification - This was implemented in the Engine in <a href="https://pypi.org/project/seldonian-experiments/0.0.8/">version 0.0.8</a>, but is not yet integrated into the Experiments library.
        </li>

        <li>
        Multiple label columns in a dataset (supervised learning)
        </li>

        <li>
        Importance sampling variants, such as weighted and per-decision importance sampling. In version alpha, the standard importance sampling estimator is the only primary objective function avaiable for Seldonian reinforcement algorithms. 
        </li>

        <li>
        An automated method for determining optimal data split between candidate data and safety data
        </li>

        <li>
        Automatic differentiation using <a href="https://github.com/google/jax">JAX</a>. JAX will be a big upgrade to autograd and should provide significant improvements for many problems. Autograd will remain an option for users who do not wish to install or use JAX. 
        </li>
        
    </ul>

</div>