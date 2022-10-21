---
layout: home
permalink: /changelog/
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h5 class="mb-3"><b>Changelog</b></h5>
    <p>Here we list noteworthy changes to the Engine and Experiments APIs</p>
    <hr class="my-4">
   
    <h5>Engine (since alpha version release):</h5>
    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.3/">0.6.3</a></h5>
    <ul>
        <li> Add housing prices example - no tutorial for it yet</li>
        <li> Add accuracy as a measure function for binary and multi-class classification</li>
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
        Add multi-class classification 
    </li>
    </ul>

    <h5><a href="https://pypi.org/project/seldonian-engine/0.6.0/">0.6.0 (alpha version)</a></h5>
    <hr class="my-4">

    <h5>Experiments (since alpha version release):</h5>
    <h5><a href="https://pypi.org/project/seldonian-experiments/0.0.9/">0.0.9</a></h5>
    <ul>
        <li> Update to reflect changes to binary classification in the Engine version>=0.6.1 </li>
        <li> Remove "include_intercept_term" functionality to conform with changes to the Engine  version>=0.6.2 </li>
        <li> Add linear regression baseline </li>
    </ul>
    <h5><a href="https://pypi.org/project/seldonian-experiments/0.0.8/">0.0.8</a></h5>
    <ul>
        <li> Turn off parallel processing for generating the episodes for running experiments  </li>
        <li> Remove "include_intercept_term" functionality to conform with changes to the Engine  version>=0.6.2 </li>
        <li> Add linear regression baseline </li>
    </ul>
    <h5><a href="https://pypi.org/project/seldonian-experiments/0.0.7/">0.0.7 (alpha version)</a></h5>
</div>