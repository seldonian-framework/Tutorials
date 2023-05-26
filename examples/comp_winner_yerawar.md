---
layout: example
permalink: /examples/comp_winner_yerawar/
title: Seldonian \| Fairness in student course completion based on student data
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Example: Fairness in student course completion based on student data</h2>
    <p align="center">Contributed by Sahil Yerawar, Pranay Reddy, and Varad Pimpalkhute and selected as one of the winners of the 2023 Seldonian Toolkit student <a href="{{ "/contest/" | relative_url}}">competition</a>. </p>
    <p align="center">
        Minor edits by Austin Hoag.
    </p>
    <p align="center">Based on <a href="https://github.com/pranay-ar/UnfairSlayers_Seldonian_oulad">this Github repository</a></p>
    <hr class="my-4" />
    <h3> Contents </h3>
    <ul>
        <li> <a href="#intro">Introduction</a> </li>
        <li> <a href="#dataset_prep">Dataset preparation</a> </li>
        <li> <a href="#experiments">Seldonian experiments</a></li>
        <li> <a href="#fairness">Fairness constraints</a></li>
        <li> <a href="#education">How does the Seldonian algorithm work in building a fair online education system?</a></li>
        <li> <a href="#summary">Summary</a></li>
        
    </ul>
    <hr class="my-4">
    <h3 id="intro">Introduction</h3>
    <p>
    	With the increasing adoption of Massive Online Open Courses (MOOC), newer educational systems are being built, which can gauge the needs of students to suggest them courses appropriate for them. One of the key factors these systems could consider is the prediction whether if the student given the course would pass or fail the course. Apart from considering the academic factors, these systems could also take into account the personal factors like age, gender, region and disability in their prediction decision, which poses a risk of being unfair while using these attributes. There is a great scope in building fair educational systems, which can be used to provide courses to all in a fair manner.
    </p>

    <p>
    	In this tutorial, we show how Seldonian algorithm can be used in this context to build a fair online education system which is fair across various student demographics. We use the [OULAD dataset](https://analyse.kmi.open.ac.uk/open_dataset) here, which contains information about 32,593 students and their demographic data, used in predicting whether a student is likely to pass the courses offered by Open University. Open University is a public British University that also has the highest number of undergraduate students in the UK. The data presented here is sourced from the Open University's Online Learning platform.
    </p>
	</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="dataset_prep" align="center" class="mb-3">Dataset preparation</h3>
<p>
    The dataset for this tutorial can be found at this <a href="https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad">link</a> and the file <code>preprocess.py</code> converts the dataset into a format suitable for our task. The following is an outline of the dataset preparation process of our pipeline. 
</p>

<ul>
    <li> Firstly, we dropped the columns like student ID which have no importance in the predicition pipeline. </li>
    <li> Secondly, we manipulated the columns like highest education where we grouped divisions like A level or equivalent, post grads, and HE qualification to be a boolean 1 whereas lower than A level and no formal quals to be 0. This is done in order to make <code class="highlight">higher_ed</code> attribute to be binary valued, which is used for further analysis.</li>
    <li> We also converted columns like distinction to binaries. </li>
    <li> The next step is to convert the categorical variables into numerical values. This is done using the LabelEncoder function of the scikit-learn library. The LabelEncoder function assigns a numerical value to each unique categorical value in the column.</li>
    <li> After converting the categorical variables, the next step is to standardize the numerical variables. This is done using the StandardScaler function of the scikit-learn library. The StandardScaler function standardizes the numerical variables by subtracting the mean and dividing by the standard deviation.</li>
</ul>

<p>
    Once the preprocessing steps are complete, we save the dataframe and the meta data which is later used in training and experimentation. The preprocessing step is necessary because we want to work upon the attributes of <code class="highlight">gender</code>, <code class="highlight">disability</code> and <code class="highlight">higher_education</code> to assess whether the predictions are unfair on these attributes.
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="experiments" name="experiments" align="center" class="mb-3">Seldonian Experiments</h3>

<p>
	<code>exp.py</code> takes care of running the seldonian experiments across various spec objects. The file takes two command line inputs: <code class="highlight">attribute</code> which is the name of the attribute to be considered for analysis (<code class="highlight">[gender,higher_ed,disability]</code>) and <code class="highlight">constraint_type</code> which is the nature of the constraint string used to create the spec object (<code class="highlight">[disp,eq]</code>).
</p>

<p>
    Running this script for all the spec objects creates the following experiment plots
</p>


<div align="center mb-4">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/contest_winner_yerawar/disparate_0.9.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Overall accuracy equality 0.2">
        <img src="{{ "/assets/img/contest_winner_yerawar/equalized_0.9.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Overall accuracy equality 0.1">
    </figure> 
    <figcaption><b>Figure 1</b>: Seldonian Experiments using disparate impact and equalized odds as the definition of fairness on the <i>gender</i> attribute.</figcaption>
</div>


<div align="center mb-4">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/contest_winner_yerawar/Constraint2_disability.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.2">
        <img src="{{ "/assets/img/contest_winner_yerawar/Constraint2_disability_eq.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.1">
    </figure> 
    <figure><b>Figure 2</b>: Seldonian Experiments using disparate impact and equalized odds as the definition of fairness on the <i>disability</i> attribute.</figure>
</div>

<div align="center mb-4">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/contest_winner_yerawar/constraint3_higher_ed_orig_disp.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.2">
        <img src="{{ "/assets/img/contest_winner_yerawar/constraint3_higher_ed_eq.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.1">
    </figure> 
    <figure><b>Figure 3</b>: Seldonian Experiments using disparate impact and equalized odds as the definition of fairness on the <i>higher education</i> attribute.</figure>
</div>

<p>
    We compare the performance of our quasi-Seldonian algorithm (QSA)(blue) with the random classifier (green) and logistic_regression (orange) models. The points and bands in each sub-graph denotes the mean value computed across 50 trials.
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="fairness" align="center" class="mb-3">Fairness constraints</h3>

<p>
These are various fairness metrics that can be used to evaluate the fairness of machine learning models. Below we describe several measures a of group fairness, which ensure that the model does not discriminate against any particular group.
</p>

<ol>
    <li>
        Disparate impact: This metric measures the ratio of the rate at which the positive outcome (such as getting a loan or being hired) occurs for one group (e.g., males) to the rate at which it occurs for another group (e.g., females). A value of 1 indicates no disparity, while a value less than 1 indicates that one group is less likely to receive the positive outcome. The disparate impact constraint can be written: $min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M])) >= (1-\epsilon)$, where $\epsilon$ indicates the maximum allowable fractional difference in positive rates.
    </li>

    <li>
        Demographic parity: This metric measures the difference between the rate at which the positive outcome occurs for one group compared to another. A value of 0 indicates no difference, while a value greater than 0 indicates that one group is more likely to receive the positive outcome. The demographic parity constraint can be written: $abs((PR | [M]) - (PR | [F])) <= \epsilon$, where $\epsilon$ indicates the maximum allowable absolute difference in positive rates.     
    </li>

    <li>
        Equalized odds: This metric measures the difference in error rates between the groups. It requires that the false negative rate (FNR) and false positive rate (FPR) are similar across the groups. The equalized odds constraint can be written: $abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) <= \epsilon$, where $\epsilon$ indicates the maximum allowable sum of the absolute false negative rate difference and the absolute false positive rate difference. 
    </li>

    <li>
        Equal opportunity: This metric measures the difference in the FNR between the groups. It requires that the model makes false negatives at a similar rate across the groups. The equal opportunity constraint can be written: $abs((FNR | [M]) - (FNR | [F])) <= \epsilon$, where $\epsilon$ indicates the maximum allowable absolute difference in false negative rates.
    </li>

    <li>
        Predictive equality: This metric measures the difference in the FPR between the groups. It requires that the model makes false positives at a similar rate across the groups. $abs((FPR | [M]) - (FPR | [F])) <= \epsilon$, where $\epsilon$ indicates the maximum allowable absolute difference in false positive rates.
    </li>

    <li>
        Treatment of equality: This metric measures the ratio of the FNR to the FPR between the groups. It requires that the ratio is similar across the groups. $abs((FNR | [M])/(FPR | [M])-(FNR | [F])/(FPR | [F])) <= \epsilon$, where $\epsilon$ indicates the maximum allowable absolute difference between the ratio of the false negative rate and false positive rate for one group and that of the other group.
    </li>

    <li>
        Overall accuracy equality: This metric measures the difference in the overall accuracy of the model between the groups. $abs((TPR | [M])+(TNR | [M])-((TPR | [F])+(TNR | [F]))) <= \epsilon$, where $\epsilon$ indicates the maximum allowable absolute difference.
    </li>
</ol>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="education" align="center" class="mb-3">How does the Seldonian algorithm work in building a fair online education system?</h3>

<p>
    In the context of building a fair online education system, the Seldonian algorithm can be used to ensure that the system does not discriminate against certain student demographics, such as gender, age, disability, etc. The algorithm achieves this by adding constraints to the learning process that ensure that the predictions made by the model are fair across these demographics.
</p>

<p>
    For example, suppose we want to ensure that our online education system is fair with respect to gender. We can add a constraint to the learning process that ensures that the system's predictions are not biased towards one gender. We can formulate this constraint using the concept of "disparate impact," which measures the ratio of the rate at which a positive outcome occurs for one gender to the rate at which it occurs for the other gender. We can set a threshold for this ratio, and then add a constraint to the learning process that ensures that the ratio does not fall below this threshold.
</p>

<p>
    The Seldonian algorithm optimizes the model to minimize the prediction error subject to these constraints. This ensures that the model is not only accurate but also fair across various student demographics.
</p>
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="summary" align="center" class="mb-3">Summary</h3>

<p>
In this tutorial, we implemented the Seldonian Toolkit on the OULAD dataset, building a classifier which predicts the success of a student in a given course. We played around with a variety of constraints, both differing in constraint format and the attribute considered for fairness. We compared the performance of the Seldonian Algorithm with respect to that of a random classifier and a logistic regression model with the help of the <code>experiments</code> library. For the case of <code class="highlight">higher_ed</code>, as the performance of the Seldonian Algorithm approaches that of a logistic regression model without constraints, the logistic regression model violates the fairness constraints very often, while the QSA algorithm always respects the fairness bounds and delivering similar performance.
</p>