---
layout: example
permalink: /examples/comp_winner_lacy/
title: Seldonian \| Breast Cancer Recurrence example
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    <h2 align="center" class="mb-3">Example: Fairness for Breast Cancer Recurrence Prediction</h2>
    <p align="center">An application of the Seldonian Machine Learning Toolkit for breast cancer recurrence prediction that is fair based on menopausal status. Contributed by Derek Lacy, and selected as one of the winners of the 2023 Seldonian Toolkit student <a href="{{ "/contest/" | relative_url}}">competition</a></p>
    <p align="center">
        Minor edits by Austin Hoag.
    </p>
    <p align="center">Based on <a href="https://github.com/d1lacy/Fairness-for-Breast-Cancer-Recurrence-Prediction">this Github repository</a></p>
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
    <h3 id="intro">Introduction</h3>
    <p>
    	In the United States, around 1 in 8 women develop breast cancer. Breast cancer rates are higher in older, post menopausal women, especially women who experienced late menopause. However, women of all ages can be affected by breast cancer. Of women diagnosed with and treated for breast cancer, between 3% and 15% experience a recurrence within 10 years. It is known that people who develop breast cancer before age 35 are more likely to get a recurrence, but otherwise medical experts do not know why some experience recurrence and others do not. With a fair way of predicting recurrence, doctors could provide more effective and continued care to those who are likely to need it, and certain demographics of patients would not be at a disadvantage.
    </p>

    <p>
    	Using the Seldonian Toolkit, I trained a logistic regression model to predict who will and will not experience breast cancer recurrence. I ensured fairness based on menopausal status with two different definitions of fairness. I used a dataset called the <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer">"Breast Cancer Data Set"</a> found in the UCI Machine Learning Repository provided by the Oncology Institute. The unprocessed data (found at <code>./data/breast-cancer.csv</code>) provides 286 instances of 9 breast cancer patient features including menopause satatus which can be used to predict breast cancer recurrence. More information on the data can be found in <code>./data/breast-cancer.txt</code>
    </p>
	</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="dataset_prep" align="center" class="mb-3">Dataset preparation</h3>
<p>
    In <code>dataProcessing.ipynb</code>, I process the data to be used in the Seldonian Toolkit. I first define the label column to be 'recurrence-events' and rename it to recurrence. I then change the menopause column to contain only true or false values. I make 'lt40' and 'ge40' (2 values meaning post menopause) True and change 'premeno' values to False. I also change the values in 'nodes-cap' and 'iradiat' columns to boolean values. I then one hot encode all the columns of the data set (all features are categorical) and I label encode the labels dataframe. I then join the features and labels back to one dataframe, randomize the rows, and rename columns "menopause_False" and "menopause_True" to "premenopause" and "menopause". Finally, I save the processed data as a csv (<code>data/BC_Data_Proc.csv</code>) and define and save the metadata needed for the Seldonian Engine (<code>data/metadata_breast_cancer.json</code>).
</p>


</div>
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="formulate" align="center" class="mb-3">Formulate the Seldonian ML problem</h3>
<p>
    I use two definitions of fairness for my application of the Seldonian Algorithm. Overall accuracy equality ensures that the prediction accuracy is similar for pre and post menopausal patients. Equalized odds ensures that the false negative rate (rate of predicting no recurrence when there will be one) is similar between both groups. Here are their formal definitions:
</p>

<ol>
    <li>Overall accuracy equality: 
    $$\min((\mathrm{ACC} | [\mathrm{premenopause}])/(\mathrm{ACC} | [\mathrm{menopause}]),(\mathrm{ACC} | [\mathrm{menopause}])/(\mathrm{ACC} | [\mathrm{premenopause}])) \geq (1-\epsilon)$$
    </li>
    <li>Equalized odds:
    $$\min((\mathrm{FNR} | [\mathrm{premenopause}])/(\mathrm{FNR} | [\mathrm{menopause}]),(\mathrm{FNR} | [\mathrm{menopause}])/(\mathrm{FNR} | [\mathrm{premenopause}])) \geq (1-\epsilon)$$</li>
</ol>

<p>
I use 0.2, 0.1, 0.05 as $\epsilon$ values which mean ACC or FNR must be within 20%, 10% and 5% for both groups. I set $\delta=0.05$ for all 6 combinations of the 2 fairness definitions and 3 values of $\epsilon$.
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="spec_object" align="center" class="mb-3">Creating the specification object</h3>

In <code>createSpecs.py</code>, I create 6 spec objects, each for a different combination of a fairness definition with a value for $\epsilon$. The spec objects are saved in <code>./specs/</code>.
</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="experiments" name="experiments" align="center" class="mb-3">Running a Seldonian Experiment</h3>

<p>
	In <code>generateAllPlots.py</code>, I run 6 experiments for the 6 spec objects. Each experiment uses a quasi-Seldonian model "qsa" representing one of the 2 fairness definitions with one of the 3 epsilon values. In each experiment, the qsa is compared to a baseline regression model and a random classifier. The models are compared on accuracy, probability of solution, and probability constraint was violated based on different amounts of training data. Each model is compared on 15 different amounts of training data each with 50 trials.
</p>

<p>
    This program saves results and resampled data frames for each experiment in results/ and saves the following plots to images/:
</p>

<p>
    <em>Note: you can also run <code>generateOnePlot.py</code> to take input for a fairness definition and an epsilon value to run a single experiment</em>
</p>

<p>
    Experiment results for overall accuracy equality fairness with $\epsilon$ = 0.2, 0.1, 0.05:
</p>
<div align="center mb-4">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/contest_winner_lacy/overall_accuracy_equality_0.2.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Overall accuracy equality 0.2">
        <img src="{{ "/assets/img/contest_winner_lacy/overall_accuracy_equality_0.1.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Overall accuracy equality 0.1">
        <img src="{{ "/assets/img/contest_winner_lacy/overall_accuracy_equality_0.05.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Overall accuracy equality 0.05">
        
    </figure> 
</div>


<p>
    Experiment results for equal opportunity fairness definition with $\epsilon$ = 0.2, 0.1, 0.05:
</p> 

<div align="center mb-4">
    <figure class='mt-4'>
        <img src="{{ "/assets/img/contest_winner_lacy/equal_opportunity_0.2.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.2">
        <img src="{{ "/assets/img/contest_winner_lacy/equal_opportunity_0.1.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.1">
        <img src="{{ "/assets/img/contest_winner_lacy/equal_opportunity_0.05.png" | relative_url}}" class="img-fluid mx-auto d-block rounded shadow p-3 mb-2 bg-white" style="width: 65%"  alt="Equal opportunity 0.05">
        
    </figure> 
</div>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="discussion" align="center" class="mb-3">Discussion</h3>

<p>
With overall accuracy equality as the fairness constraint and accross all epsilon values, the qsa is less accurate than the logistic regression baseline but more accurate than the random classifier. The qsa also never violates the fairness constraint, while the logistic regression baseline clearly violates the fairness constraint for 
 = 0.1, and 0.05. The logistic regression likely makes a prediction of recurrence with some bias for menopausal status resulting in different accuracies which is an unwanted behavior. The qsa is always fair. The qsa takes more data to find a solution, but has access to only a small amount of data (286 rows). It is clearly improving in accuracy before it runs out of data. With more data the qsa could be much more accurate but still fair.
</p>

<p>
With the equal opportunity fairness constraint, the results are similar. The qsa is always fair but requires more data to find a solution. Once it finds a solution with fairness for equal opportunity (at around 100 rows of data), the accuracy is actually very similar to logistic regression. The logistic regression violates the fairness constraint for all values of epsilon. For either pre-menopausal or post-menopausal patients, it would have a higher rate of false negative predictions meaning it would under predict recurrence events. This behaviour could lead to patients of one of those groups being underprepared for recurrence.
</p>

</div>

<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h3 id="summary" align="center" class="mb-3">Summary</h3>

<p>
The seldonian algorithm seems very promising in this application. A normal logistic regression behaves in unfair ways by predicting recurrence with different accuracies, or with different false negative rates for pre-menopausal or post-menopausal patients. The qsa is always fair and can be nearly as accurate, especially with more data to train on. 
</p>