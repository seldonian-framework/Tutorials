---
layout: tutorial
permalink: /tutorials/fair_loans_tutorial/
prev_url: /tutorials/simple_engine_tutorial/
prev_page_name: Simple Engine tutorial
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    
<h1 class="mb-3">Tutorial: Fairness in loan decision making</h1>

<hr class="my-4">

<h3>Introduction</h3>

<p>This tutorial is intended to provide a more realistic use case of the Seldonian Toolkit compared to the <a href="{{ page.prev_url | relative_url }}">simple engine tutorial</a>. We will be using the <a href="https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)">UCI Statlog (German Credit Data) Data Set</a>, which contains 20 attributes for a set of 1000 people and a single label column for whether they are a high or low credit risk. If someone is a high credit risk, a bank is less likely to give provide them with a loan. Our goal in this tutorial will be to use the Seldonian Toolkit to create a model that makes predictions about credit scores that are fair between males and females. We will use several definitions of fairness, and we stress that these definitions may not be the correct ones to use in reality. They are simply examples to help you understand how to use this library. </p>

<h3>Outline</h3>

<p>In this tutorial, you will learn how to:</p>

<ul>
    <li>Format the UCI German Credit Data dataset so that we use it in the Seldonian Toolkit </li>
    <li>Build a Seldonian Machine Learning model that implements common fairness constraints on this dataset </li>
    <li>Run a Seldonian experiment, assessing the performance and safety of the Seldonian ML model and how it compares to other fairness-aware machine learning models. </li>
</ul>

<h3> Dataset preparation </h3>

<p>
    If you would like to skip this section, you can find the correctly-formatted dataset and metadata file here: <a href="https://github.com/seldonian-toolkit/Engine/tree/main/static/datasets/supervised/german_credit">https://github.com/seldonian-toolkit/Engine/tree/main/static/datasets/supervised/german_credit</a>.
</p>

<p>
    UCI provides two version of the dataset and a file "german.doc" describing the "german.data" file only. We ignored the "german.data-numeric" file because there was no documentation for it. We downloaded the file "german.data" from here: <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/">https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/</a>. We converted it to a CSV file by replacing the space characters with commas. Attribute 9 according to "german.doc" is the personal status/sex of each person. This is a categorical column with 5 possible values, 2 of which describe females and 3 of which describe males. We created a new column that has a value of "F" if female (A92 or A95) and "M" if male (any other value) and dropped the personal status column. We decided to ignore the marriage status of the person for the purpose of this tutorial. 
</p>

<p> 
    Next, we one-hot encoded all thirteen categorical features, including the sex feature that we created in the previous step. We applied a standard scaler to the remaining numerical 7 features. The one-hot encoding step created an additional 39 columns, resulting in 59 total features. The final column in the dataset is the label, which we will refer to as "credit_rating" from here on out. We mapped the values of this column as such: (1,2) -> (0,1) so that they would behave well in  binary classification models. We combined the 59 features and the single label column into a single pandas dataframe and saved the file as a CSV file, which can be found <a href="https://github.com/seldonian-toolkit/Engine/blob/main/static/datasets/supervised/german_credit/german_loan_numeric_forseldonian.csv">here</a>.
</p>

<p> 
    We also prepared a JSON file containing the metadata that we will need to provide to the Seldonian Engine library. The column names beginning with "c__" come from  scikit-learn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html">OneHotEncoder</a> applied to a pandas dataframe of the features with column names we created ourselves by interpreting the column meanings described in "german.doc" file. The columns "M" and "F" are somewhat buried in the middle of the columns list, and correspond to the male and female one-hot encoded columns. The "sensitive_columns" key in this JSON file points to those columns.
    {% highlight python %}
    {
      "regime": "supervised",
      "sub_regime": "classification",
      "columns": [
        "c__account_status_A11",
        "c__account_status_A12",
        "c__account_status_A13",
        "c__account_status_A14",
        "c__credit_history_A30",
        "c__credit_history_A31",
        "c__credit_history_A32",
        "c__credit_history_A33",
        "c__credit_history_A34",
        "c__purpose_A40",
        "c__purpose_A41",
        "c__purpose_A410",
        "c__purpose_A42",
        "c__purpose_A43",
        "c__purpose_A44",
        "c__purpose_A45",
        "c__purpose_A46",
        "c__purpose_A48",
        "c__purpose_A49",
        "c__savings_accounts_A61",
        "c__savings_accounts_A62",
        "c__savings_accounts_A63",
        "c__savings_accounts_A64",
        "c__savings_accounts_A65",
        "c__employment_since_A71",
        "c__employment_since_A72",
        "c__employment_since_A73",
        "c__employment_since_A74",
        "c__employment_since_A75",
        "F",
        "M",
        "c__other_debtors_A101",
        "c__other_debtors_A102",
        "c__other_debtors_A103",
        "c__property_A121",
        "c__property_A122",
        "c__property_A123",
        "c__property_A124",
        "c__other_installment_plans_A141",
        "c__other_installment_plans_A142",
        "c__other_installment_plans_A143",
        "c__housing_A151",
        "c__housing_A152",
        "c__housing_A153",
        "c__job_A171",
        "c__job_A172",
        "c__job_A173",
        "c__job_A174",
        "c__telephone_A191",
        "c__telephone_A192",
        "c__foreign_worker_A201",
        "c__foreign_worker_A202",
        "n__months",
        "n__credit_amount",
        "n__installment_rate",
        "n__present_residence_since",
        "n__age_yrs",
        "n__num_existing_credits",
        "n__num_people_liable",
        "credit_rating"
      ],
      "label_column": "credit_rating",
      "sensitive_columns": [
        "F",
        "M"
      ]
    }
    {% endhighlight python %}

We saved this file as a JSON file, which can be found <a href="https://github.com/seldonian-toolkit/Engine/blob/main/static/datasets/supervised/german_credit/metadata_german_loan.json">here</a>
</p>

<h3>Formulate the Seldonian ML problem</h3>

<p>
    As in the <a href="{{ page.prev_url | relative_url }}">previous tutorial</a>, we  need to define the machine learning problem in the absence of constraints. This is a classification problem, so let's use a logistic regression model to minimize an objective function which is the logistic loss. 
</p>

<p>
    Now let's add the behavioral constraints. The first fairness constraint that we will consider is called predictive equality, which ensures that the false positive rates between sensitive groups are less than some threshold value. In the <a href="{{ page.prev_url | relative_url }}">previous tutorial</a>, you learned how to write fairness constraints for a regression problem using the special measure function "Mean_Squared_Error" in the constraint string. For predictive equality, the measure function we will use is "FPR", which stands for false positive rate. Predictive equality between our two sensitive attribute columns "M" and "F" with a threshold value of 0.2 can be written as: abs( (FPR | [M]) - (FPR | [F]) ) - 0.2.
Let us enforce this constraint function with a probability of $0.95$. 
</p>

<p>
    The problem can now be fully formulated as a Seldonian machine learning problem:
</p>

<p>
    Minimize the logistic loss, subject to the constraint:
<ul>
    <li>
        $g_{1} = \mathrm{abs}( (FPR | [M]) - (FPR | [F]) ) - 0.2$, and ${\delta}_1=0.05$.  
    </li>
</ul>
</p>

<h3>Creating the specification object</h3>

<p>
    To be able to run the Seldonian algorithm using the Engine, we need to create a <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.spec.SupervisedSpec.html#seldonian.spec.SupervisedSpec">SupervisedSpec</a> object. We could create this by following the same set of steps described in the <a href="{{ page.prev_url | relative_url }}">previous tutorial</a>, but in this case we will use the Seldonian Interface GUI to do this for us. 
</p>

<p>
    Start up the GUI and then upload the data file we linked to above in the "Data file" field of the "Data and metadata setup" section. Then select the "supervised" regime and "classification" sub-regime from the drop-downs in that section. Copy and paste the following text string into the "All attributes" field: 
    <code>
    c__account_status_A11,c__account_status_A12,c__account_status_A13,c__account_status_A14,c__credit_history_A30,c__credit_history_A31,c__credit_history_A32,c__credit_history_A33,c__credit_history_A34,c__purpose_A40,c__purpose_A41,c__purpose_A410,c__purpose_A42,c__purpose_A43,c__purpose_A44,c__purpose_A45,c__purpose_A46,c__purpose_A48,c__purpose_A49,c__savings_accounts_A61,c__savings_accounts_A62,c__savings_accounts_A63,c__savings_accounts_A64,c__savings_accounts_A65,c__employment_since_A71,c__employment_since_A72,c__employment_since_A73,c__employment_since_A74,c__employment_since_A75,F,M,c__other_debtors_A101,c__other_debtors_A102,c__other_debtors_A103,c__property_A121,c__property_A122,c__property_A123,c__property_A124,c__other_installment_plans_A141,c__other_installment_plans_A142,c__other_installment_plans_A143,c__housing_A151,c__housing_A152,c__housing_A153,c__job_A171,c__job_A172,c__job_A173,c__job_A174,c__telephone_A191,c__telephone_A192,c__foreign_worker_A201,c__foreign_worker_A202,n__months,n__credit_amount,n__installment_rate,n__present_residence_since,n__age_yrs,n__num_existing_credits,n__num_people_liable,credit_rating
    </code>
    Enter M,F into the "Sensitive attributes" field, and credit_rating into the "Label column" field.
</p>

<p>
    Scroll down to the "Constraint building blocks" area and click the "Predictive equality" button. This will auto-fill "Constraint #1" with a preconfigured constraint for predictive equality, which will have the exact form that we defined above. Type 0.05 into the ${\delta} = $ field below where the constraint function was auto-filled. Then hit the submit button. A dialog box should show up displaying: "Saved ./spec.pkl", which indicates that the specification object has been saved as a pickle file to the directory where you launched the GUI. 
</p>

<p>
    We are now ready to run the Seldonian algorithm. The code below modifies some defaults of the spec object that we created using the GUI and then runs the Seldonian algorihtm using the modified spec object.  

{% highlight python %}
# loan_fairness.py
import os

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
    # loan spec file
    interface_output_dir = os.path.join('/Users/ahoag/beri/code',
        'interface_outputs/loan_predictive_equality')
    specfile = os.path.join(interface_output_dir,'spec.pkl')
    spec = load_pickle(specfile)
    spec.primary_objective = spec.model_class().sample_logistic_loss
    
    spec.use_builtin_primary_gradient_fn = False
    spec.optimization_hyperparams['alpha_theta'] = 0.05
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
    spec.optimization_hyperparams['num_iters'] = 500
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run()
    print(passed_safety,solution)
    if passed_safety:
        print()
        print("Primary objective evaluated on safety test:")
        print(SA.evaluate_primary_objective(branch='safety_test',theta=solution))

{% endhighlight python %}
Create a file called "loan_fairness.py", copy the code above into the file and run it via the command line:
{% highlight bash %}
$ python loan_fairness.py
{% endhighlight bash %}

You should see some output like:
{% highlight python %}
Iteration 0
Iteration 10
Iteration 20
Iteration 30
Iteration 40
...
True [-0.13039981  0.67941344  0.55112    -0.72471417 -1.02456061  0.20229882
  0.47424601 -0.14324495 -0.19447507 -0.99334423  0.03501196 -0.87121189
 -0.72662925 -0.26529318 -0.06931054  0.15878545 -0.00750695  0.72941399
  0.08979682 -0.11101921  0.48035899 -0.04484105 -0.29450799 -0.3227534
 -0.47027917  0.35079107  0.01956879 -0.05637623 -0.91105897 -0.05459372
  0.05005435  0.50653839 -0.94673578 -0.31349763 -0.33209043 -0.1703951
  0.29578712 -0.17005938  0.2094934  -0.43020084  0.46068146  0.00416818
 -0.85611563 -0.10375768 -0.44768108 -0.15209936  0.1817179   0.06881386
 -0.329364   -0.12869479 -0.1216518   0.26806043  0.27829218  0.38621858
  0.0475732   0.16009162  0.37374806  0.32736901]

Primary objective evaluated on safety test:
0.5719906937681168
{% endhighlight %}
The exact numbers you see might differ slightly depending on your machine's random number generator.

</p>
<h3>Summary</h3>
<p>In this tutorial, we demonstrated how to:</p>
<ul>
    <li>Use the Engine to set up a Seldonian machine learning algorithm </li>
    <li>Run the algorithm using the Engine and understand its outputs</li>
</ul>
<p>
</p>

</div>