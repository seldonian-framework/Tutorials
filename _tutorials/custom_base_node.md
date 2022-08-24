---
layout: tutorial
permalink: /tutorials/custom_base_node_tutorial/
prev_url: /tutorials/fair_loans_tutorial/
prev_page_name: Fair loans tutorial
next_url: /tutorials/gridworld_RL_tutorial/
next_page_name: Reinforcement learning first tutorial - gridworld
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
    
<h2 align="center" class="mb-3">Tutorial: Creating your own constraint base nodes</h2>

<hr class="my-4">
<p> The purpose of this tutorial is to help you understand how to extend the functionality of string-based behavioral constraints to suit your custom constraints.
</p>
<h3 class="my-4">Introduction</h3>

<p>
In cases where behavioral constraints can be expressed a mathematical equations or inequalities, which is often the case for fairness constraints (for example, see <a href="https://fairware.cs.umass.edu/papers/Verma.pdf">Verma et al., 2018</a>), the constraints can be passed to the Seldonian Engine as strings with specific syntax. Specifically, these strings can contain five types of things:
<ol>
<li>Mathematical operators <code>(+,-,*,/)</code> </li>
<li>The native Python math functions: <code>min(),max(),abs(),exp()</code> </li>
<li>Constant numbers, such as 0.5</li>
<li>Measure functions, such as "PR" (positive rate) which represent statistical functions that have been specifically implemented in the library </li>
<li>Custom strings that represent a custom base node, which are the subject of this tutorial</li>
</ol> 
For more details on the rules for providing behavioral constraint strings to the Engine, see <a href="https://seldonian-toolkit.github.io/Engine/build/html/overview.html#behavioral-constraints">the Engine docs</a>. 
</p>
<p>
While the Engine supports a wide range of mathematical expressions, it is possible that you will require more custom functionality in your base nodes. First, ask yourself: "can the functionality I need be achieved using a new measure function?" If the answer is "Yes" or "Not sure", then proceed to the next section. If not, then skip to the <a href="#custom_base_node">Adding a custom base node</a> section.

<h3 class="my-4"> Adding a new measure function </h3>
</p>
<p>
     The currently programmed measure functions are listed <a href="https://seldonian-toolkit.github.io/Engine/build/html/_autosummary/seldonian.parse_tree.operators.html#seldonian.parse_tree.operators.measure_functions_dict">here</a> by regime and sub-regime. If your desired constraint involves a statistical function not listed there, for example <a href="https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)">precision</a>, you could add that functionality by doing the following:
<ol>
<li>Fork the Engine repository: https://github.com/seldonian-toolkit/Engine </li>
<li>Define a string to represent the precision operator that you will type into your constraint. For the sake of this example, let's call it "PREC" </li>
<li>Add "PREC" to the list of supervised classification measure functions in the <code>measure_functions_dict</code> here: <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py#L74">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py#L74</a></li>
<li>Add the following block of code to the <code>evaluate_statistic()</code> method of the <code>ClassificationModel()</code> class here: <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/models.py#L251">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/models.py#L251</a>:

{% highlight python %}
    if statistic_name == 'PREC':
            return self.sample_Precision(
                theta,data_dict['features'],data_dict['labels'])
{% endhighlight python %}
This points to a method <code>sample_Precision()</code> which we will implement shortly to calculate a single precision value on a set of points, returning a float.
 </li>

 <li>Add the following block of code to the <code>sample_from_statistic()</code> method of the <code>ClassificationModel()</code> class here: <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/models.py#L251">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/models/models.py#L297</a>:

{% highlight python %}
    if statistic_name == 'PREC':
            return self.vector_Precision(
                theta,data_dict['features'],data_dict['labels'])
{% endhighlight python %}
This points to a method <code>vector_Precision()</code> which we will implement shortly to calculate the precision values on each point in a set of points, returning a vector.
 </li>
 <li>Add a new method <code>sample_Precision()</code> to the <code>ClassificationModel()</code> class that calculates the precision on a set of points and returns a float. Note that we will use the existing methods of the same class <code>sample_True_Positive_Rate()</code> and <code>sample_False_Positive_Rate()</code> in our implementation of the precision.

{% highlight python %}
    def sample_Precision(self,theta,X,Y):
        """ Calculate the precision 
        on whole sample
        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :param Y: The labels
        :type Y: numpy ndarray
        :return: Precision
        :rtype: float
        """
        tpr = self.sample_True_Positive_Rate(theta,X,Y)
        fpr = self.sample_False_Positive_Rate(theta,X,Y)
        res = tpr/(tpr + fpr)
        return res
{% endhighlight python %}
 </li>
 <li>Likewise, add a new method <code>vector_Precision()</code> to the <code>ClassificationModel()</code> class that calculates the precision on each points in a set of points and returns a vector of floats. As before, we will use existing methods of the same class in our implementation of the vector precision.

{% highlight python %}
    def vector_Precision(self,theta,X,Y):
        """ Calculate the probabilistic precision 
        on each point in a sample of points
        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :param Y: The labels
        :type Y: numpy ndarray
        :return: Precision values
        :rtype: List(float)
        """
        tp_values = self.vector_True_Positive_Rate(theta,X,Y)
        fp_values = self.vector_False_Positive_Rate(theta,X,Y)
        res = tp_values/(tp_values + fp_values)
        return res
{% endhighlight python %}
 </li>
</ol>
At this point, you can now use the "PREC" in your constraint string that you provide to the Engine. It will have all of the same abilities as other measure functions, such as filtering by sensitive attributes. Let's say your constraint is that you want the precision of the model to differ by no more than 10% between males and females in the dataset. This constraint could be expressed to the Engine as: 
{% highlight python %}
abs((PREC | [M]) - (PREC | [F])) - 0.1
{% endhighlight python %}
</p>
<h3 class="my-4" id="custom_base_node"> Adding a custom base node </h3>

<p>
If the answer to the question: "Can the functionality I need can be achieved using a new measure function?" is "No", then you may need to implement a new base node class. For example, consider the Seldonian regression algorithm presented by <a href="https://www.science.org/stoken/author-tokens/ST-119/full">Thomas et al. (2019)</a> (see Figure 2), designed to enforce fairness in GPA prediction between male and female applicants based on their scores on nine entrance examinations. The specific fairness constraint enforced in that paper was: the mean prediction error between male and female students should not differ by more 0.05 GPA points. This constraint is actually expressible using existing syntax: <code>abs((Mean_Error | [M]) - (Mean_Error | [F])) - epsilon</code>, where <code>Mean_Error</code> is an existing built-in measure function.
</p>

<p>
However, it turns out that a tighter confidence bound can be obtained using a custom method for calculating the mean error difference. This alternative method pairs up the male and female observations and provides unbiased estimates of the mean error difference all in one node. This cannot be achieved simply by adding a new custom measure function, as we did above for adding the functionality for precision. The reason is that we need to override the function that constructs the dataset and the function that calculates the estimate of the base node, which cannot be done by adding a new measure function. This will become more clear as we implement the new custom base node below.
</p>

<p>
Below, we outline the steps for creating this custom base node, called <code>MEDCustomBaseNode()</code>. The custom base node is already fully implemented in the code for reference <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/nodes.py#L577">here</a>. The steps are written generally to help you implement your own custom base class.
</p>

<ol>

<li>Fork the Engine repository: https://github.com/seldonian-toolkit/Engine </li>

<li> Define a class in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/nodes.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/nodes.py</a> that inherits from the <code>BaseNode</code> class, ideally conforming to <a href="https://en.wikipedia.org/wiki/Camel_case">upper camel case</a> and named something that uniquely identifies your custom base node. The name must not already be an existing class in that file. We chose <code>MEDCustomBaseNode(BaseNode)</code>. The rest of the steps are just writing methods that override the existing methods of the <code>BaseNode</code> class. </li>

<li> Define an <code>__init__()</code> method for the new class which at minimum takes the following arguments and calls <code>super().__init__()</code> to register these arguments to the parent class: 
{% highlight python %}
def __init__(self,
    name,
    lower=float('-inf'),
    upper=float('inf'),
    **kwargs):
    super().__init__(name,lower,upper,**kwargs)
{% endhighlight python %}
</li>

<li> Define a method of the new class called <code>zhat()</code>. <b>Critical: At a minimum, this method needs to take as input a model object, weight vector and data dictionary, and it must return a vector of unbiased estimates (floats) of your base node expression </b>. In our example, we want to calculate the pair-wise mean errors: $(y_i - \hat{y}_i | M) - (y_j - \hat{y}_j | F)$, where $y_i$ is the true label for point $i$ and $\hat{y}_i$ is the prediction for point $i$. Here is our implementation of this method for our example custom base node:
{% highlight python %}
def zhat(self,model,theta,data_dict):
    """
    Pair up male and female columns and compute a vector of:
    (y_i - y_hat_i | M) - (y_j - y_hat_j | F).
    There may not be the same number of male and female rows
    so the number of pairs is min(N_male,N_female)

    :param model: machine learning model
    :type model: models.SeldonianModel object

    :param theta: 
        model weights
    :type theta: numpy ndarray
    
    :param data_dict: 
        contains inputs to model, 
        such as features and labels
    :type data_dict: dict
    """
    X_male = data_dict['X_male'].values
    Y_male = data_dict['Y_male'].values
    X_female = data_dict['X_female'].values
    Y_female = data_dict['Y_female'].values

    prediction_male = model.predict(theta,X_male)
    mean_error_male = prediction_male-Y_male

    prediction_female = model.predict(theta,X_female)
    mean_error_female = prediction_female-Y_female

    return mean_error_male - mean_error_female
{% endhighlight python %}
Notice that we reference several keys <code>'X_male','Y_male','X_female','Y_female'</code> of the <code>data_dict</code>. The <code>data_dict</code> that is passed to the <code>zhat</code> method of the parent <code>BaseNode</code> class only contains the keys: <code>features</code> and <code>labels</code>. <code>data_dict</code> is constructed in the <code>calculate_data_forbound</code> method of <code>BaseNode</code>, so we will need to override that method in our new class as well. <b>The need to override these methods is the main reason why we needed to create a new class here instead of a just creating a new measure function</b>.</li>

<li> <p>Create a <code>calculate_data_forbound()</code> method of your new class which takes only <code> **kwargs </code> as input and returns a data dictionary and the number of observations in the potentially modified data, datasize. This will override this method from the parent class <code>BaseNode</code>. The purpose of this method is to perform preprocessing of the data that you don't need to re-run each time you compute the bound in candidate selection. This method runs a single time at the beginning of candidate selection, and the result gets cached in the parse tree. Each subsequent time the confidence bound is calculated in candidate selection, the cached data is accessed rather than recomputed, potentially speeding up candidate selection enormously.</p>
<p>
In our example, we need to pair up male and female rows in the data set. The feature and label vectors need to be filtered so that they have the same number of female and male observations. Importantly, this filtering does not depend on theta, the current weight vector of the model, so it only needs to be done once in candidate selection. Here is our implementation of this method for our custom base node. In our <code>calculate_data_forbound()</code> method, we call a helper method called <code>precalculate_data()</code> which we also include below:
{% highlight python %}
def calculate_data_forbound(self,**kwargs):
    """ 
    Overrides same method from parent class, :py:class:`.BaseNode`
    """
    dataset = kwargs['dataset']
    dataframe = dataset.df
    
    # set up features and labels 
    label_column = dataset.label_column
    labels = dataframe[label_column]
    features = dataframe.loc[:, dataframe.columns != label_column]
    features.insert(0,'offset',1.0) # inserts a column of 1's
    
    # Do not drop the sensitive columns yet. 
    # They might be needed in precalculate_data()
    data_dict,datasize = self.precalculate_data(
        features,labels,**kwargs)

    if kwargs['branch'] == 'candidate_selection':
        n_safety = kwargs['n_safety']
        frac_masked = datasize/len(dataframe)
        datasize = int(round(frac_masked*n_safety))

    return data_dict,datasize

def precalculate_data(self,X,Y,**kwargs):
    """ 
    Preconfigure dataset for candidate selection or 
    safety test so that it does not need to be 
    recalculated on each iteration through the parse tree
    :param X: features
    :type X: pandas dataframe
    :param Y: labels
    :type Y: pandas dataframe       
    """
    dataset = kwargs['dataset']
    male_mask = X.M == 1

    # drop sensitive column names 
    if dataset.sensitive_column_names:
        X = X.drop(columns=dataset.sensitive_column_names)
    X_male = X[male_mask]
    Y_male = Y[male_mask]
    X_female = X[~male_mask]
    Y_female = Y[~male_mask]
    N_male = len(X_male)
    N_female = len(X_female)
    N_least = min(N_male,N_female)
    
    # sample N_least from both without repeats 
    XY_male = pd.concat([X_male,Y_male],axis=1)
    XY_male = XY_male.sample(N_least,replace=True)
    X_male = XY_male.loc[:,XY_male.columns!= dataset.label_column]
    Y_male = XY_male[dataset.label_column]
    
    XY_female = pd.concat([X_female,Y_female],axis=1)
    XY_female = XY_female.sample(N_least,replace=True)
    X_female = XY_female.loc[:,XY_female.columns!= dataset.label_column]
    Y_female = XY_female[dataset.label_column]
    
    data_dict = {
        'X_male':X_male,
        'Y_male':Y_male,
        'X_female':X_female,
        'Y_female':Y_female}
    datasize=N_least
    return data_dict,datasize
{% endhighlight python %}
Most of the work is done in our helper function. This is out of convenience and is not required for your own implementation. Notice that <code>calculate_data_forbound()</code> returns <code>data_dict,datasize</code>, as we specified above.
</p>  
</li>

<li> Define a string expression for your custom base node that will be referenced in your constraints. It need not be a formula for the constraint, but it does need to uniquely identify this constraint (or set of constraints). The string needs to obey these rules: 
    <ul>
    <li>The string must only consist only of alphabetical characeters (upper and lower case allowed) and the underscore character. No spaces are allowed. </li>
    <li> The string must not already be an existing key of the specific sub-dictionary of the <code>custom_base_node_dict</code> dictionary given the regime and sub-regime of your problem in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py</a>.</li>
    </ul>
    We chose <code>MEDMF</code> for our string expression.
</li>
<li> Add an entry to the dictionary: <code>custom_base_node_dict</code> in <a href="https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py">https://github.com/seldonian-toolkit/Engine/blob/main/seldonian/parse_tree/operators.py</a>, where the key is the string expression of your constraint and the value is the name of the class you defined in step 2. 
</li>
</ol>
At this point, you will be able to use your custom base node in a constraint string that you provide to the Engine. Our original desired constraint was: "ensure that the mean prediction error between male and female students should not differ by more 0.05 GPA points". This can be enforced using a constraint string of <code>MEDMF - 0.05</code> and some value of delta provided as you would normally provide delta.

</div>