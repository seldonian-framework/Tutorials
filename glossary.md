---
layout: home
permalink: /glossary/
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
<h3 class="mb-3">Glossary</h3>
<hr class="my-4" />
<p>Here you will find brief definitions of commonly used terms in the Seldonian Framework:</p>
<h5> Behavioral constraint </h5>
<p>  Criteria for fairness or safety provided by the user. Consists of a constraint function and a confidence level. The Seldonian algorithm ensures that the behavioral constraints are met probabilistically. </p> 

<h5>Candidate Selection</h5>
<p>One of the three major components of a Seldonian algorithm. It is the component that, using a fraction of the dataset (called the candidate dataset), searches for a solution that simultaneously optimizes the primary objective (i.e. loss function) and is predicted to satisfy the behavioral constraints on the Safety dataset, the remaining fraction of the dataset. Analogous to the training set in the standard supervised machine learning paradigm. </p>

<h5>Confidence level</h5>
<p>Often simply called ${\delta}$. Provided by the user, the confidence level is used to define the maximum probability that is acceptible for the Seldonian algorithm to violate a behavioral constraint.  </p>

<h5>Interface</h5>
<p>The system with which the user interacts to provide the behavioral constraints and other inputs to the Seldonian algorithm. Examples include simple command line interfaces, scripts, or more complicated graphical user interfaces (GUIs).</p>

<h5 id="measure_function">Measure function</h5>
<p>Components of behavioral that, if appearing in a constraint string, will be recognized by the engine as statistical functions with special meaning. Examples are "Mean_Squared_Error", "FPR" (standing for false positive rate), and "J_pi_new" (standing for performance of new policy in the reinforcement learning regime). </p>

<h5>Primary objective function</h5>
<p>The objective function (also called loss function) that, in the absence of behavioral constraints, wouuld be solely optimized by the machine learning model. The Seldonian machine learning model seeks to simultaneously optimize the primary objective function while satisfying the behavioral constraints. Performance on the objective function is sometimes sacrified to satisfy the behavioral constraints, depending on the problem.</p>

<h5>Regime</h5>
<p>The broad category of machine learning problem, e.g. supervised learning or reinforcement learning. </p>

<h5>Safety test</h5>
<p>One of the three major components of a Seldonian algorithm. It is the component that, given a solution determined during candidate selection, tests whether that solution satisfies the behavioral constraints on the held out safety dataset that were not used to find the solution. Analogous to the test set in the standard supervised machine learning paradigm. </p>

<h5>Seldonian algorithm</h5>
<p>An algorithm designed to enforce high probability constraints in a machine learning problem</p>

<h5>Sensitive attribute</h5>
<p>In a fairness constraint, a sensitive attribute is one against which the model should not discriminate. Gender and race are common examples. Also sometimes called the protected attribute. It only pertains to supervised learning. </p>

<h5>Sub-regime</h5>
<p>Within supervised learning, the sub-regimes supported by this library are classification and regression. Reinforcement learning does not currently have sub-regimes in this library.</p>


</div>


