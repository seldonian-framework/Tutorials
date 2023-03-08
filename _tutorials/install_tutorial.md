---
layout: tutorial
permalink: /tutorials/install_toolkit_tutorial/
prev_url: /tutorials/alg_details_tutorial/
prev_page_name: (A) Seldonian algorithm details 
next_url: /tutorials/simple_engine_tutorial/
next_page_name: (C) Running the Seldonian Engine
title: Seldonian \| Tutorial B
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h2 align="center" class="mb-3">Tutorial B: Installing the Seldonian Toolkit libraries</h2>
<hr class="my-4">
<h3> Contents </h3>
    <ul>
        <li> <a href="#overview">Toolkit structure</a> </li>
        <li><a href="#engine">Engine installation</a></li>
        <li><a href="#experiments">Experiments installation</a></li>
        <li><a href="#gui">GUI installation</a></li>
    </ul>
    <hr class="my-4">

<h3 id="overview"> Toolkit structure </h3>

<p>
The Seldonian Toolkit consists of two Python libraries and a GUI. The two libraries are the Seldonian Engine library and the Seldonian Experiments library. We recommend always installing the libraries in the same virtual environment. If using the GUI, install that in the same environment as well. 
</p>

<p>
    The toolkit is compatible and tested with Python versions 3.8, 3.9, and 3.10. <b>It is incompatible with Python versions lower than version 3.8</b>. We do not plan to support Python versions lower than 3.8, but we do plan to add support for Python version 3.11 soon. 
</p>

<h3 id="engine"> Engine installation </h3>  

<p> 
    The Seldonian Engine library can be installed with a single line:
</p>

{% highlight javascript %}
$ python -m pip install --upgrade seldonian-engine 
{% endhighlight javascript %}

<p>
    This will make the <code class="highlight">seldonian</code> package available for you to import in Python. Warnings about scripts not being in PATH can be ignored. To turn off these warnings in the future, one can add <code class="highlight">--no-warn-script-location</code> to the pip install command. To test that the install worked, run this command from the command line:
</p>

{% highlight javascript %}
$  python -c "from seldonian.seldonian_algorithm import SeldonianAlgorithm"
{% endhighlight javascript %}
    
<p>
    If it worked, there will be nothing printed to the command line. If you get this error:
</p>

{% highlight python %}
ModuleNotFoundError: No module named 'seldonian'
{% endhighlight python %}

<p>
    then make sure that the Python executable you are using to run this command is the same one you used to do the "pip install" command above. You can see the full path of your Python command by running:
</p>

{% highlight javascript %}
$ which python 
{% endhighlight javascript %}

<p>
    on Unix systems or:
</p>

{% highlight javascript %}
$ where python 
{% endhighlight javascript %}

<p>
    on Windows systems. The "gcm" command may work instead of "where" in Powershell.
</p>

<h3 id="experiments"> Experiments installation </h3>  

<p> 
    The Seldonian Engine library is a dependency of the Seldonian Experiments library. If you know you are planning to use the Experiments library, then you can skip the above steps of installing the Engine because it will be automatically installed when the Experiments library is installed. The Experiments library can be installed with a single line:
</p>

{% highlight javascript %}
$ python -m pip install --upgrade seldonian-experiments 
{% endhighlight javascript %}

<p>
    This will make the "experiments" package available for you to import in Python. To test that it worked, run this command from the command line:
</p>

{% highlight javascript %}
$ python -c "from experiments.generate_plots import SupervisedPlotGenerator"
{% endhighlight javascript %}

<p>
If it worked, there will be nothing printed to the command line. If you get this error:
</p>

{% highlight python %}
ModuleNotFoundError: No module named 'experiments'
{% endhighlight python %}

<p>
    then make sure that the Python executable you are using to run this command is the same one you used to do the "pip install" command above. You can see the full path of your Python command by running:
</p>


{% highlight javascript %}
$ which python 
{% endhighlight javascript %}

<p>
    on Unix systems or:
</p>

{% highlight javascript %}
$ where python 
{% endhighlight javascript %}

<p>
    on Windows systems. The "gcm" command may work instead of "where" in Powershell.
</p> 

<h3 id="gui"> GUI installation </h3>  

<p> 
    First, clone the GUI repo:
</p>

{% highlight bash %}
$ git clone https://github.com/seldonian-toolkit/GUI.git
{% endhighlight bash %}

<p>
    This will create a folder called "GUI" wherever you ran the above command. Enter that folder, and run the following from the command line (after entering your virtual environment, if relevant): 
</p>

{% highlight javascript %}
$ pip install -r requirements.txt
{% endhighlight javascript %}  

<p>
    The Engine library is one of the requirements of the GUI, so you may seem some output about it already being reinstalled. You can test that the installation worked by running the GUI:
</p>

{% highlight javascript %}
$ python run.py
{% endhighlight javascript %}

<p>
    This will start a webserver running at <a href="localhost:5001">localhost:5001</a> on your local machine. Go to that address in your browser, and you should see a page like the following: 
</p>

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gui_screenshot.png" | relative_url}}" class="img-fluid my-2" style="width: 95%"  alt="GUI screenshot"> 
        <figcaption align="left"> <b>Figure 1</b> - A screenshot of the top of the Seldonian Interface GUI webpage.   </figcaption>
    </figure>
</div>

</div>