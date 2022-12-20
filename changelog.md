---
layout: home
permalink: /changelog/
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h5 class="mb-3"><b>Changelog</b></h5>
    <p>Here we list noteworthy changes to the Engine and Experiments APIs. Some minor versions are skipped here as they did not include noteworthy changes. For a full release version history see <a href="https://pypi.org/project/seldonian-engine/#history">https://pypi.org/project/seldonian-engine/#history</a>. </p>
    <hr class="my-4">
   
    <h5>Engine (since alpha version release):</h5>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.3/">0.7.6</a></h5>
    <ul>
        <li> Allow sensitive attributes in RL datasets. </li>
        <li> Better logging for gradient descent. </li>
        <li> Add option to plot running average of primary objective function and Lagrangian in gradient descent plots. </li>
        <li> Fixes bug where mini-batches were incorrectly calculated when constraints included conditional columns.</li>
        <li> Add PyTorch facial recognition example. </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.3/">0.7.5</a></h5>
    <ul>
        <li> PyTorch forward/backward passes made more efficient. </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.3/">0.7.4</a></h5>
    <ul>
        <li> Add support for Tensorflow models. </li>
        <li> Add option to batch safety data for the safety test. </li>
    </ul>


    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.3/">0.7.3</a></h5>
    <ul>
        <li> RL environment specified as object in hyperparam dict </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.3/">0.7.1</a></h5>
    <ul>
        <li> Add support for PyTorch models. </li>
        <li> Create example demonstrating PyTorch CNN on MNIST. </li>
        <li> Pandas no longer used internally. It is currently only for data loading for CSV data files. </li>
        <li> Supervised learning features can be list of arrays or a single NumPy array. </li>
        <li> Add option to mini-batch in gradient descent. </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.3/">0.6.3</a></h5>
    <ul>
        <li> Add housing prices example - no tutorial for it yet.</li>
        <li> Add accuracy as a measure function for binary and multi-class classification. </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.2/">0.6.2</a></h5>
    <ul>
    <li>
        Remove "include_intercept_term" parameter in dataset, so user no longer can specify if they want a column of ones added in a supervised dataset. The intercept is now handled entirely by the model. 
    </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.1/">0.6.1</a></h5>
    <ul>
    <li>
        Add multi-class classification. 
    </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.0/">0.6.0 (alpha version)</a></h5>
    
    <hr class="my-4">

    <h5>Experiments (since alpha version release):</h5>
        
    <h5><a href="https://pypi.org/project/seldonian-experiments/0.0.9/">0.0.10</a></h5>
    <ul>
        <li> Add support for mini-batching. </li>
        <li> RL environment specified as object in hyperparam dict. </li>
        <li> Fairlearn made optional import. </li>
        <li> Option to choose log scale for vertical axis on the performance plot. </li>
        <li> Fix bug where parse tree base node dict was not being reset before calculating g on ground truth dataset for prob(violated constraint) plot. </li>
        <li> Add option to provide batch_epoch_dict to specify the batch size and number of epochs to use for each data_frac. </li>
        
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-experiments/0.0.9/">0.0.9</a></h5>
    <ul>
        <li> Update to reflect changes to binary classification in the Engine version>=0.6.1. </li>
        <li> Remove "include_intercept_term" functionality to conform with changes to the Engine  version>=0.6.2 </li>
        <li> Add linear regression baseline. </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-experiments/0.0.8/">0.0.8</a></h5>
    <ul>
        <li> Turn off parallel processing for generating the episodes for running experiments.  </li>
        <li> Remove "include_intercept_term" functionality to conform with changes to the Engine  version>=0.6.2. </li>
        <li> Add linear regression baseline. </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-experiments/0.0.7/">0.0.7 (alpha version)</a></h5>
</div>