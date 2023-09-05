---
layout: example
permalink: /examples/diabetes/
title: Seldonian \| Insulin pump controller with reinforcement learning example
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Example: Seldonian reinforcement learning for safe diabetes treatment</h2>
    <hr class="my-4" />
    <h3> Contents </h3>
    <ul>
        <li> <a href="#intro">Introduction</a> </li>
        <li> <a href="#dataset_prep">Dataset preparation</a> </li>
        <li> <a href="#formulate">Formulate the Seldonian ML problem</a> </li>
        <li> <a href="#spec_object">Creating the specification object</a></li>
        <li> <a href="#experiments">Running a Seldonian experiment</a></li>
        <li> <a href="#discussion">Discussion</a></li>
        <li> <a href="#summary">Summary</a></li>
        
    </ul>
    <hr class="my-4">
    
    <h3 id="background">Background: Type 1 Diabetes Treatment as an RL Problem</h3>
    <p>
        Type 1 diabetes describes a condition where the body produces insufficient amounts of insulin. Natural insulin controls the delivery of glucose in the bloodstream to cells throughout the body. In doing so, it lowers the blood glucose concentration. People with untreated type 1 diabetes tend to have high blood glucose concentrations, a condition called hyperglycemia, which can have significant negative health consequences. One treatment for type 1 diabetes is the subcutaneous injection of synthetic insulin using injections with a syringe or an insulin pen.
    </p>

    <p>
        If too much synthetic insulin is injected, blood glucose levels can become dangerously low, a condition called hypoglycemia. Controlling hyperglycemia is important to prevent the long-term consequences of diabetes, and hypoglycemia is a common severe unintended consequence. The symptoms of hypoglycemia are often more acute and range from palpitations, sweating, and hunger, to altered mental status, confusion, coma, and even death. 
    </p>

    <p>
        In treating type 1 diabetes, it is critical to correctly estimate how much insulin a person should inject to mitigate hyperglycemia without inducing hypoglycemia. Insulin is usually delivered through "basal" injections, which regulate blood glucose between meals, and "bolus" injections, which are given just before mealtime to counteract the increase in blood glucose that results from eating a meal. Often, a bolus calculator is used to determine how much bolus insulin one should inject prior to eating a meal. This calculator is often personalized by a physician based on patient data. In this example, we show how to use the Seldonian toolkit to create an RL algorithm that personalizes the parameters of a bolus calculator to keep blood glucose concentrations within safe levels, while ensuring that dangerously low blood glucose levels are avoided.
    </p>
<p>
    The determination of the parameters of a bolus calculator for Type 1 diabetes treatment can be framed as a reinforcement learning (RL) problem. We can consider an episode to be a single day of patient data during  treatment with the bolus calculator. On each day, there is a single action, which is the selection of the parameters of the bolus calculator at the beginning of that day. The state is a complete description of the person throughout the day, though the only part of the state that we observe is the patient's blood glucose levels. At each point throughout the day when the blood glucose levels are observed, an instantaneous reward is assigned based on the difference between the measured blood glucose level and a target blood glucose level, typically determined by a physician. We model our reward function off of Figure 1.6 from Bastani (2014). This reward function penalizes departures from healthy blood glucose concentrations, with worse penalties for hypoglycemia compared to hyperglycemia. At the end of the day, the mean of the instantaneous rewards is calculated and is considered to be the single reward for the day. Our reward function, which is entirely negative-valued, is shown below:
</p>

<div align="center" class="my-4">
    <figure>
        <img src="{{ "/assets/img/diabetes_tutorial/simglucose_primary_reward.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 50%"  alt="Primary reward function">
    </figure> 
    <figcaption><b>Figure 1</b>: The primary reward function (blue). All rewards are negative. Rewards are larger in the target range, and quickly drop off as blood glucose drops beyond this range. At blood glucose lower than 36 mg/dL or above 350 mg/dL, we stop the episode, as blood glucose values outside of this range are unphysical. </figcaption>
</div>

<p>
    This reward function, $R_0$ is defined as:
</p>

$$
\begin{equation} 

\begin{cases} 
      -1.5*10^{-8}*|\text{BG} - \text{target}| & \text{BG} < \text{target} \\
      -1.6*10^{-4}*(\text{BG} - \text{target})^2 & \text{BG} \geq \text{target} \\
   \end{cases},
\label{primary_reward}
\end{equation}
$$

<p>
    where BG is the current blood glucose value and target is the target blood glucose value.
</p>
<p>
    A policy is parameterized by a bolus calculator, which we define in the following equation:
</p>

$$
\begin{equation}
\text{injection} = \frac{\text{carbohydrate content of meal}}{CR} + \frac{\text{blood glucose} - \text{target blood glucose}}{CF},
\label{crcf_equation}
\end{equation}
$$

<p>
    where blood glucose is an estimate of the person’s current blood glucose (measured from a blood sample and using milligrams per deciliter of blood, i.e., mg/dL), target blood glucose is the desired blood glucose (also measured using mg/dL), which is typically specified by a physician, the carbohydrate content of meal is an estimate of the size of the meal to be eaten (measured in grams of carbohydrates), and $CR$ and $CF$ are two real-valued parameters that, for each person, must be tuned to make the treatment policy effective. $CR$ is the carbohydrate-to-insulin ratio, and is measured in grams of carbohydrates per insulin unit, while $CF$ is called the correction factor or insulin sensitivity, and is measured in insulin units per mg/dL. 
</p>
    
</div>


<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="problem" align="center" class="mb-3">Formulating the Seldonian RL problem</h3>

<p>
    In a realistic treatment scenario, a physician might propose an initial range of $(CR,CF)$ values and want to know what is the best sub-range for a particular patient. Therefore, we seek to find a distribution over policies, i.e., a subset of the initially-proposed $(CR,CF)$ parameter space, that optimizes the expected return. Using a standard RL algorithm, one could maximize the expected return of the primary reward (Equation \ref{primary_reward}). However, there would be no guarantee that the optimal policy distribution was actually safe when deployed for that patient. In particular, a reasonable concern is that even though the reward function we chose disincentivizes hypoglycemia, there is still a chance of it occurring, and we do not know what the chance is. What we desire is some degree of confidence that the deployed bolus controller (with $(CR,CF)$ values taken from the obtained distribution over policies) will have lower instances of hypoglycemia than the initially-proposed controller that uses the larger range of $(CR,CF)$ values. 
</p>

<p>
    Our Seldonian optimization problem is therefore to optimize the primary expected return given the primary reward, $R_0$ from Equation \ref{primary_reward}, while simultaneously enforcing the safety constraint that the incidence of hypoglycemia is lower than the that of the initial distribution over policies. To quantify the incidence of hypoglycemica, we define an auxiliary reward function which is identical to Equation \ref{primary_reward} when 

</p>     
    The return function should penalize deviations of blood glucose from optimum levels, with larger penalties for blood glucose levels that are too low. This means that the return function must be selected to trade-off between hyperglycemia and hypoglycemia, a problem described in detail by Bastani (2014).
Here the user of our quasi-Seldonian RL algorithm (Fig. S20) would have to make a decision: what return function quantifies their personal view of the trade-off between hypoglycemia and hyperglycemia, and what auxiliary return functions capture their view of undesirable behavior? Because our aim is to evaluate our Seldonian approach, not to advocate a particular answer to these questions, we adopt the return function used in previous work [30]. To this end, we let the history from each day contain a record of blood glucose levels at each minute, i.e., h = (g0,g1,g2,...,g1440), where gt denotes the blood glucose measurement t minutes after midnight, in mg/dL. The return function is then given by:
</p>

 Here we consider the case in which a physician initially proposes ranges of possible values for CR and CF for a particular individual. The parameter values for any insulin dosage policy considered should lie within these ranges. We refer to E[r1(H)] as the prevalence of low blood glucose hereafter. Below we also discuss the mean-time-hypoglycemic-per-day, a different possible measure of the prevalence of low blood glucose.
Our aim is to evaluate how a batch RL algorithm can take in a series of previously applied policies (settings of CR and CF) for a patient, collected across m ∈ N>0 days, and output a new distribution over policies that, for this particular patient, increases the expected return as defined by Eq. S38. Furthermore, this algorithm must ensure that the safety constraint is satisfied; that is, it must ensure that with high probability the distribution over policies will not be changed in a way that increases the prevalence of low blood glucose. As an additional safety measure, we might also require the hard constraint that the RL algorithm
51
will never deploy values of CR and CF outside the range specified by the physician. This application therefore combines aspects of a multiobjective problem (the return function trades-off hypoglycemia and hyperglycemia), a problem with hard constraints (that the values for CR and CF will always be within the window specified by the physician), and the high-probability behavioral constraints allowed by the Seldonian framework (that the prevalence of low blood glucose will not be increased).

Bastani (2014) and Thomas et al. (2019) used the FDA-approved type 1 diabetes metabolic simulator (T1DMS). T1DMS is written in MATLAB and is closed source, making it a poor fit for the toolkit. Instead, we used <a href="https://github.com/jxx123/simglucose">simglucose</a>, an open source Python implementation of the <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/">FDA-approved UVa/Padova Simulator (2008 version)</a>. We quickly noticed that the Python simulator differed significantly from the MATLAB T1DMS simulator used by Thomas et al. (2019). In particular, the 


One strength of our approach is that it provides per-subject guarantees. That is, standard ML approaches to constructing and evaluating adaptive bolus calculators often measure performance across a population of individuals and argue that new bolus calculators work better for the population (with arguments of statistical significance over the entire population). These arguments of statistical significance provide a form of safety guarantee about the
49
performance of the new adaptive bolus calculator for the population. We instead focus on providing a personalized guarantee—that the controller for an individual subject will not be changed in a way that is worse for that one individual. We therefore perform multiple simulations of a particular individual, with each simulation using different random meal times and random samples within the learning algorithm. Although initially we focus on the individual adult#003 within T1DMS, later we show that this individualized approach is effective for personalizing treatments for all ten simulated adults within T1DMS.
We study the problem of finding a distribution over RL policies, rather than a single policy. Here, a single policy describes the parameters $CR$ and $CF$ of a simulated insulin pump, which delivers bolus injections of insulin to a simulated diabetic patient approximately every eight hours over a period of 24 hours. The size of the bolus injection is defined as:


</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="dataset_prep" align="center" class="mb-3">Dataset preparation</h3>
<p>
    The <a href="https://github.com/mhyeh/Fairness-for-Lie-Detection/blob/main/preprocess.ipynb">preprocess.ipynb</a> notebook implements the steps described in this section. If you would like to skip this section, you can find the correctly reformatted dataset at <a href="https://github.com/mhyeh/Fairness-for-Lie-Detection/blob/main/data/proc_data/data.pkl">data/proc_data/data.pkl</a>, which is the end product of the notebook.
</p>

<p>
The dataset we used can be downloaded <a href="https://www.cs.ucsb.edu/~william/data/liar_dataset.zip">here</a>. The zip file contains three TSV files: train set, validation set, and test set. We merge them together into a single dataframe. The label column (column 2) has six values: "pants-fire", "false", "barely-true", "half-true", "mostly-true", and "true." We discard data points labeled with "half-true", and treat data points labeled with "pants-fire", "false", and "barely-true" as lies, and those with "mostly-true", and "true" as truth. We discard the columns of ID (column 1), speaker's name (column 5), and subject(s) (column 4) since they are either meaningless or hard to be encoded. We also discard the column of the total credit history count (columns 9-13) because of the i.i.d. assumption of Seldonian Algorithms. For the columns of the speaker's job (column 6), party (column 8), state (column 7), and the statement's context (column 14), we one-hot encode each of them into a maximum of 20 categories. 
</p>

<p>
In order to feed the statement into a logistic regression model rather than only using the categorical features described above, we represent each statement with LIWC (Linguistic Inquiry and Word Count), which is a psycholinguistic dictionary that groups words into 75 categories, and has been used to detect lies in multiple deception studies. Specifically, for each statement $s$, we encode it as $[u_1,...,u_{75}]$, where $$u_i=\frac{1}{|s|}\sum_j^{|C_i|}v(s, i, j)$$ denotes the coverage rate of the $i$th word category with respect to the statement. The function $v(\cdot,i,j)$ measures the occurrence count of word $w_{i,j}\in C_i$, and $|s|$ represents the number of tokens in $s$. We also calculate the "number of words per sentence", "number of words with 6+ letters", and "word count" as features. 
</p>

<p>
We create two new binary-value columns, "democrat" and "republican", to denote whether the speaker is a Democrat or Republican as sensitive features. These two columns are not used for model training.
</p>

<p>
In total, we have 159 features per statement. We save the dataframe as a pickle (`.pkl`) file.
</p>
</div>
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="formulate" align="center" class="mb-3">Formulate the Seldonian ML problem</h3>
<p>
	We consider four different definitions of fairness to apply to the problem of predicting whether statements are lies or not. The four definitions and their constraint strings are:
</p>
<ol>
<li> Disparate impact: $\min((\mathrm{PR} | [\mathrm{democrat}])/(\mathrm{PR} | [\mathrm{republican}]),(\mathrm{PR} | [\mathrm{republican}])/(\mathrm{PR} | [\mathrm{democrat}])) \geq \epsilon$ </li>
<li> Equalized odds: $\min((\mathrm{FNR} | [\mathrm{democrat}])/(\mathrm{FNR} | [\mathrm{republican}]),(\mathrm{FNR} | [\mathrm{republican}])/(\mathrm{FNR} | [\mathrm{democrat}])) \geq \epsilon$ </li>
<li> Predictive equality: $\min((\mathrm{FPR} | [\mathrm{democrat}])/(\mathrm{FPR} | [\mathrm{republican}]),(\mathrm{FPR} | [\mathrm{republican}])/(\mathrm{FPR} | [\mathrm{democrat}])) \geq \epsilon$ </li>
<li>Overall accuracy equality: $\min((\mathrm{ACC} | [\mathrm{democrat}])/(\mathrm{ACC} | [\mathrm{republican}]),(\mathrm{ACC} | [\mathrm{republican}])/(\mathrm{ACC} | [\mathrm{democrat}])) \geq \epsilon$ </li>
</ol>


<p>We try 0.8, 0.9, 0.95 as different $\epsilon$, and apply each of these constraints independently, each with $\delta = 0.05$.</p>
<p>
<b>Note</b>: For equalized odds, predictive equality, and overall accuracy equality, there are two ways of encoding $\epsilon$ in the formulas to quantify the amount of violation: ratio and difference. In this example, we use the ratio between two probabilities as the quantifier, i.e., we consider $\min(\frac{p_A}{p_B},\frac{p_B}{p_A})\geq \epsilon$. However, one can also use the absolute difference between two probabilities as the quantifier. I.e., $|p_A-p_B|\leq \epsilon$.
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="spec_object" align="center" class="mb-3">Creating the specification object</h3>

In <a href="https://github.com/mhyeh/Fairness-for-Lie-Detection/blob/main/create_spec.py">createSpec.py</a>, we create a different spec object for each constraint to run 12 experiments (4 fairness definitions times 3 epsilon values). We make these spec objects using a for loop and save them in <a href="https://github.com/mhyeh/Fairness-for-Lie-Detection/tree/main/data/spec">data/spec</a>. 

Running this code should print out that the 12 spec files have been created.
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="experiments" name="experiments" align="center" class="mb-3">Running a Seldonian Experiment</h3>

<p>
	In <a href="https://github.com/mhyeh/Fairness-for-Lie-Detection/blob/main/generate_experiment_plots.py">generate_experiment_plot.py</a>, we run a Seldonian Experiment with a quasi-Seldonian model, a baseline logistic regression model, and a random classifier baseline model. The performance metric is accuracy. The code takes two inputs:
	<ol> 
		<li>Constraint name, chosen from <code class='codesnippet'>['disparate_impact', 'predictive_equality', 'equal_opportunity', 'overall_accuracy_equality']</code></li>
		<li> $\epsilon$, chosen from <code class='codesnippet'>[0.8, 0.9, 0.95]</code></li>
	</ol>
</p>

<p>
	Running the script for each constraint will produce the following plots:
</p>
<div align="center">
    <figure class='mt-4'>
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/disparate_impact_0.2_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Disparate impact 0.8">
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/disparate_impact_0.1_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Disparate impact 0.9"> 
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/disparate_impact_0.05_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Disparate impact 0.95">
    </figure> 
    <figcaption><b>Figure 1</b>: Seldonian Experiments using disparate impact as the definition of fairness. Each row is the result with a different $\epsilon$. From top to bottom: $\epsilon=0.8,0.9,0.95$. </figcaption>
</div>


<div align="center">
    <figure class='mt-4'>
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/predictive_equality_0.2_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Predictive equality 0.8">
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/predictive_equality_0.1_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Predictive equality 0.9"> 
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/predictive_equality_0.05_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Predictive equality 0.95">
    </figure> 
    <figcaption><b>Figure 2</b>: Same as Figure 1, but using predictive equality as the definition of fairness. </figcaption>
</div>

<div align="center">
    <figure class='mt-4'>
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/equal_opportunity_0.2_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.8">
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/equal_opportunity_0.1_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.9"> 
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/equal_opportunity_0.05_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.95">
    </figure> 
    <figcaption><b>Figure 3</b>: Same as Figure 1, but using equal opportunity as the definition of fairness. </figcaption>
</div>

<div align="center mb-4">
    <figure class='mt-4'>
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/overall_accuracy_equality_0.2_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Overall accuracy equality 0.8">
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/overall_accuracy_equality_0.1_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Overall accuracy equality 0.9"> 
        <img src="https://raw.githubusercontent.com/mhyeh/Fairness-for-Lie-Detection/main/images/overall_accuracy_equality_0.05_accuracy.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Overall accuracy equality 0.95">
    </figure> 
    <figcaption><b>Figure 4</b>: Same as Figure 1, but using overall accuracy equality as the definition of fairness. </figcaption>
</div>

<p class='mt-4'>
	For each $\epsilon$ and definition of fairness, a Seldonian Experiment creates three plots, accuracy (left), solution rate (middle), and failure rate (right). The colored points and bands in each panel show the mean (points) and standard error (uncertainty bands) over 50 trials. We compare a logistic model learned with a quasi-Seldonian algorithm (qsa, blue) to a logistic regression baseline (orange) and a random classifier (green).
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="discussion" align="center" class="mb-3">Discussion</h3>

<p>
All the plots show that although the QSA requires the most samples to return a solution and achieve optimal accuracy, it is the only model that always satisfies the fairness constraints regardless of the number of samples. In addition, except for the experiment using overall accuracy equality as the definition of fairness, the logistic regression model violates the fairness constraint most of the time. Moreover, the violation rate increases when we increase the number of samples for training. Because increasing the number of samples improves model accuracy, these findings suggest that the logistic regression model's high accuracy may not be due to its ability to differentiate lies and truth based on statements. Instead, it may pay too much attention to the party the speakers belong to. The QSA, on the other hand, is fair but also has a high accuracy, implying that its decision may be more dependent on the statement itself.
</p>

<p>
A special case occurs when overall accuracy equality is used as the definition of fairness. In this case, both the logistic regression model and the QSA satisfy the fairness constraint. Moreover, the QSA outperforms the logistic regression model at the end. This result suggests that a Seldonian algorithm may sometimes obtain a solution closer to the optimal one.
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="summary" align="center" class="mb-3">Summary</h3>

<p>
In this example, we demonstrated how to use the Seldonian Toolkit to build a predictive model that enforces a variety of fairness constraints on the text-based lie detection dataset. We covered how to construct the dataset and metadata so that they can be used by the Seldonian Experiment. We then ran Seldonian Experiments to evaluate the true performance and safety of the QSA. We found that the QSA can satisfy a range of custom-defined fairness constraints and that the model does not violate the constraints. We compared the QSA to a logistic regression model and found that the performance of the QSA approaches the performance of a logistic regression model that lacks constraints. The logistic regression model frequently violate the constraints for most of the fairness definitions. 
</p>