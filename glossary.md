---
layout: home
permalink: /glossary/
title: Seldonian \| Glossary
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
<h3>Glossary</h3>
<hr class="my-2" />
<p>Here you will find brief definitions of commonly used terms in the Seldonian framework:</p>
<h5 id="behavioral_constraint"> Behavioral constraints </h5>
<p>  Criteria for fairness or safety provided by the user. Each behavioral constraint consists of a constraint function and a confidence level. In many cases, the constraint function can be constructed from a constraint string provided by the user. The Seldonian algorithm ensures that the behavioral constraints are met with probability of at least $1-\delta$, where $\delta$ is the confidence level provided by the user. </p> 

<h5 id="candidate_selection">Candidate selection</h5>
<p>One of the major components of a Seldonian algorithm. It is the component that, using a fraction of the dataset (called the candidate dataset), searches for a solution that simultaneously optimizes the primary objective (i.e., loss function) and is predicted to satisfy the behavioral constraints on the safety dataset, the remaining fraction of the dataset. The candidate dataset is analogous to the training set in the standard supervised machine learning paradigm. </p>

<h5 id="confidence_level">Confidence level</h5>
<p>Often called ${\delta}$. Provided by the user, the confidence level is used to define the maximum acceptable probability for the Seldonian algorithm to violate a behavioral constraint.  </p>

<h5 id="interface">Interface</h5>
<p>The system the user interacts with to provide the behavioral constraints and other inputs to the Seldonian algorithm. Examples include simple command line interfaces, scripts, or more complicated graphical user interfaces (GUIs).</p>

<h5 id="measure_function">Measure function</h5>
<p>Components of a behavioral constraint that, if appearing in a constraint string, will be recognized by the engine as statistical functions with special meaning. Examples are "Mean_Squared_Error", used in regression problems, "FPR", standing for false positive rate and used in classification problems, and "J_pi_new_IS", which stands for the performance of the new policy for reinforcement learning problems, as evaluated by ordinary importance sampling. </p>

<h5 id="primary_objective">Primary objective function</h5>
<p>The objective function (also called loss function) that, in the absence of behavioral constraints, would be solely optimized by the machine learning model. The Seldonian machine learning model seeks to simultaneously optimize the primary objective function while satisfying the behavioral constraints. Performance on the objective function is sometimes sacrificed to satisfy the behavioral constraints, depending on the problem.</p>

<h5 id="regime">Regime</h5>
<p>The broad category of machine learning problem, e.g., supervised learning or reinforcement learning. </p>

<h5 id="safety_test">Safety test</h5>
<p>One of the three major components of a Seldonian algorithm. It is the component that, given a solution determined during candidate selection, tests whether that solution satisfies the behavioral constraints on the held-out safety dataset that was not used to find the solution. The safety dataset is analogous to the test set in the standard supervised machine learning paradigm. </p>

<h5 id="seldonian_algorithm">Seldonian algorithm</h5>
<p>An algorithm designed to enforce high-probability constraints in a machine learning problem</p>

<h5 id="sensitive_attributes">Sensitive attributes</h5>
<p>In a fairness constraint, a sensitive attribute is one against which the model should not discriminate. Gender and race are common examples. Sensitive attributes are also sometimes called protected attributes. </p>

<h5 id="sub_regime">Subregime</h5>
<p>Within supervised learning, the subregimes supported by this library are classification (binary and multiclass) and regression. Reinforcement learning does not currently have subregimes in this library.</p>


</div>


