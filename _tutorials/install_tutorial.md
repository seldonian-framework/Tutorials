---
layout: tutorial
permalink: /tutorials/install_toolkit_tutorial/
prev_url: /tutorials/alg_details_tutorial/
prev_page_name: (A) Seldonian Algorithm details 
next_url: /tutorials/simple_engine_tutorial/
next_page_name: (C) Simple Engine tutorial
---

<!-- Main Container -->
<div class="container p-3 my-2 border" style="background-color: #f3f4fc;">
<h2 align="center" class="mb-3">Tutorial B: Installing the Seldonian Toolkit libraries</h2>
<hr class="my-4">

<h3> Installation </h3>

<p>
The Seldonian Toolkit consists of two main libraries and a GUI. The two libraries are the Seldonian Engine library and the Seldonian Experiments library. While it is not necessary to do so, we recommend installing all three pieces of software in the same virtual environment. 
</p>

<h5> Engine installation </h5>  

<p> 
    The Seldonian Engine library can be installed with a single line:
</p>

<p>
    <code>
    $ python -m pip install --upgrade seldonian-engine --no-warn-script-location
    </code>
</p>

<p>
    This will make the "seldonian" package available for you to import in Python. To test that it worked, run this command from the command line:
</p>

<p> <code>
    $  python -c "from seldonian.seldonian_algorithm import SeldonianAlgorithm"
    </code>  
    If it worked, there will be nothing printed to the command line. 
    </p>

<p>
    If you get this error:
</p>

<p>
    <code> 
    ModuleNotFoundError: No module named 'seldonian'
    </code>
</p> 

<p>
    then make sure that the Python executable you are using to run this command is the same one you used to do the "pip install" command above. You can see the full path of your Python command by running:
</p>

<p>
    <code>
    $ which python 
    </code>
</p>

<p>
    on Unix systems or:
</p>

<p>
    <code>
    $ where python 
    </code>
</p>

<p>
    on Windows systems. The "gcm" command may work instead of "where" in Powershell.
</p>

<h5> Experiments installation </h5>  

<p> 
    The Seldonian Experiments library can be installed with a single line:
</p>

<p>
    <code>
    $ python -m pip install --upgrade seldonian-experiments --no-warn-script-location
    </code>
</p>

<p>
    This will make the "experiments" package available for you to import in Python. To test that it worked, run this command from the command line:
</p>

<p> 
    <code>
    $ python -c "from experiments.generate_plots import SupervisedPlotGenerator"
    </code>  
    If it worked, there will be nothing printed to the command line. 
</p>

<p>
    If you get this error:
</p>

<p>
    <code> 
    ModuleNotFoundError: No module named 'experiments'
    </code>
</p> 

<p>
    then make sure that the Python executable you are using to run this command is the same one you used to do the "pip install" command above. You can see the full path of your Python command by running:
</p>

<p>
    <code>
    $ which python 
    </code>
</p>

<p>
    on Unix systems or:
</p>

<p>
    <code>
    $ where python 
    </code>
</p>

<p>
    on Windows systems. The "gcm" command may work instead of "where" in Powershell.
</p> 

<h5> GUI installation </h5>  

<p> 
    First, clone the GUI repo:
</p>

<p>
    <code>
    $ git clone https://github.com/seldonian-toolkit/GUI.git
    </code>
</p>

<p>
    This will create a folder called "GUI" wherever you ran the above command. Enter that folder, and run the following from the command line (after entering your virtual environment, if relevant): 
</p>

<p> 
    <code>
    $ pip install -r requirements.txt
    </code>  
</p>

<p>
    You can test that the installation worked by running the GUI:
</p>
<p>
    <code> 
    $ python run.py
    </code>
</p> 

<p>
    This will start a webserver running at localhost:5001 on your local machine. Go to that address in your browser, and you should see a page displaying something like the following: 
</p>

<div align="center">
    <figure>
        <img src="{{ "/assets/img/gui_screenshot.png" | relative_url}}" class="img-fluid my-2" style="width: 95%"  alt="GUI screenshot"> 
        <figcaption align="left"> <b>Figure 1</b> - A screenshot of the top of the Seldonian Interface GUI webpage.   </figcaption>
    </figure>
</div>

</div>