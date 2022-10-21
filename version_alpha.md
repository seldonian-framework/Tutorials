---
layout: home
permalink: /v_alpha/
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h5 class="mb-3"><b>What is in version alpha?</b></h5>
    <hr class="my-4">
    <p>We released version alpha of the Seldonian Toolkit on September 29, 2022. The alpha version of the Engine and Experiments library can be installed via: 
    {% highlight python %}
    pip install seldonian-engine==0.6.0
    pip install seldonian-experiments==0.0.7
    {% endhighlight python %}
    For the latest bug fixes and features, we recommend installing the latest development version of the library:
    {% highlight bash %}
    pip install --upgrade seldonian-engine
    pip install --upgrade seldonian-experiments
    {% endhighlight bash %}
    </p>
    <p>
    In this document, we cover what is included and excluded in the alpha version.
    </p>

    <h5>What is included in the version alpha API</h5>
    <h7><u>Packages/libraries</u></h7>
    <ul>
        <li>
        <a href="https://seldonian-toolkit.github.io/Engine/build/html/index.html">Seldonian Engine API</a>, a library that implements the Seldonian algorithm described <a href="{{ "/tutorials/alg_details_tutorial/#overview" | relative_url}}">here</a>.
        </li>

        <li>
        <a href="https://seldonian-toolkit.github.io/Experiments/build/html/index.html">Seldonian Experiments API</a>, a library for evaluating the safety and performance of Seldonian algorithms run with the Engine
        </li>

        <li>
        <a href="https://seldonian-toolkit.github.io/GUI/build/html/index.html">Seldonian Interface GUI</a>, an interactive GUI for creating behavioral constraints via drag and drop. One example of a Seldonian interface.
        </li>
    </ul>

    <h7><u>Engine features</u></h7>
    <ul>

        <li> 
        A command line Seldonian interface.
        </li>

        <li> 
        Student's $t$-test for the confidence bound calculation 
        </li>
        
        <li>
        Parse tree capable of handling wide range of user-provided behavioral constraints. Constraints can consist of basic mathematical operations (+,-,/,*) operations and use any combination of (min,max,abs,exp) functions.
        </li>

        <li>
        Parse tree visualizer
        </li>

        <li>
        Efficient bound propagation in parse tree by limiting the number of confidence intervals that need to be calculated
        </li>

        <li>
        User can specify an arbitrary number of behavioral constraints for a single Seldonian algorithm
        </li>

        <li>
        User can specify split fraction between candidate selection and safety test    
        </li>

        <li>
        Dataset loaders for CSV-formatted datasets
        </li>

        <li>
        Gradient descent with Adam optimizer module option for candidate selection
        </li>

        <li>
        Black box optimization using SciPy with barrier function module option for candidate selection
        </li>

        <li>
        Gradient descent visualizer
        </li>

        <li>
        Automatic differentiation using the "autograd" Python library for gradient descent. 
        </li>

        <li>
        User can provide gradient function for their custom primary objective. We provide several built-in gradient functions which are often faster than using autograd.
        </li>

        <li>
        Support for parametric supervised learning algorithms (binary classification and regression) as well as offline ("batch") reinforcement learning algorithms
        </li>

        <li>
        Example reinforcement learning policies supporting discrete and continuous observation spaces, such as softmax 
        </li>

        <li>
        Modular design to make implementing user-defined models and constraints seamless for developers. Tutorials to help guide design.
        </li>

    </ul>
    <h7><u>Experiments features</u></h7>
    <ul>
        <li>
        Three plot generator (performance, solution rate, failure rate) for supervised learning and reinforcement learning Seldonian algorithms
        </li>

        <li>
        Logistic regression and random classifier baseline models for comparing against Seldonian classification algorithms.
        </li>

        <li>
        Fairlearn experiment runner for several types of fairness constraints for comparing against Seldonian classification algorithms.
        </li>

        <li>
        Generate resampled datasets that approximate ground truth using no additional data (supervised learning)
        </li>

        <li>
        Generate new episodes to use as ground truth from existing policy parameterizations
        </li>

        <li>
        Modular design to make implementing new baselines seamless for developers. 
        </li>
        
    </ul>

    <h7><u>GUI features</u></h7>
    <ul>
        <li>
        Flask application that users run locally
        </li>

        <li>
        Upload locally stored datasets 
        </li>

        <li>
        Drag-and-drop to build wide array of behavioral constraints
        </li>

        <li>
        Five definitions of fairness hardcoded for quick reference
        </li>
    </ul>


    <h5> Limitations of the version alpha API</h5>
    <p> Many of the features below are in development. Check the <a href="{{"/coming_soon" | relative_url}}">Coming soon</a> page to learn more. Feel free to raise an <a href="https://github.com/seldonian-toolkit/Engine/issues">issue on github</a> requesting new features.</p>
    <ul>
        
        <li>
        The $t$-test confidence bound used when calculating the upper bound on the behavioral constraint relies on reasonable but possibly false assumptions about the distribution of the data. As a result, the algorithms implemented in version alpha are quasi-Seldonian. <a href="https://en.wikipedia.org/wiki/Hoeffding%27s_inequality">Hoeffding's</a> concentration inequality does not rely on such assumptions and, once incorporated into the Engine, will enable running true Seldonian algorithms.
        </li>

        <li>
        Multi-class classification is not yet supported
        </li>

        <li>
        Multiple label columns in a dataset are not supported (supervised learning). Currently, only a single label column is allowed. 
        </li>

        <li>
        Non-parameteric machine learning models (e.g., random forest), are not yet supported.
        </li>

    </ul>

</div>