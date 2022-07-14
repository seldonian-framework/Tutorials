---
layout: tutorial
permalink: /tutorials/gui_tutorial/
prev_url: /tutorials/simple_engine_tutorial/
prev_page_name: Simple Engine tutorial
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
<h1 class="mb-3">Tutorial: How to use the Seldonian Interface GUI</h1>
<hr class="my-4">
<h3>Introduction</h3>
<p>Generally, a Seldonian interface is where the user provides the data, metadata, and behavioral constraints that they want to enforce in their Seldonian machine learning algorithm. Different interfaces may be required for different use cases, and the Seldonian Interface GUI described here is one example of such an interface. This GUI is designed to be used when the behavioral constraint functions can be written as mathematical expressions, such as <code>Mean_Squared_Error - 2.0</code>, i.e. "ensure the mean squared error is less than 2.0. These expressions can be quite complex, as we will see. The GUI will help guide the user on how to create these expressions in the format that the Seldonian Engine can parse. </p>

<h3>Outline</h3>
<p>In this tutorial, you will learn how to:</p>
<ul>
    <li>Provide data and metadata using the GUI </li>
    <li>Build behavioral constraints using the GUI </li>
    <li>Understand the output of the GUI, which can be used to run Seldonian algorithms. </li>
</ul>
<h3
> Installation </h3>
<p> 
    The first step is to download the Seldonian Interface GUI if you have not already:
</p>
<p>
    <code>
    $ git clone https://github.com/seldonian-toolkit/GUI.git
    </code>
</p>
<p>
    Then, in a virtual environment, install the dependencies:
</p>
<p>
    <code>
    $ pip install -r requirements.txt
    </code>
</p> 
<h3> Running the GUI </h3>
<p>
    The GUI is designed to be used in the browser. It is currently tested in Google Chrome and Firefox. To start the GUI, enter the virtual environment in which you installed the requirements.txt file and run from the command line:
</p>
<p>
    <code>
    $ python run.py
    </code>
</p>
<p>This will start a webserver running on port 5001 on your machine. Once you have it running, click this link <a href="http://localhost:5001">http://localhost:5001</a> or manually enter that URL in your browser.</p>

<p>If the server was started successfully, you should see this page when navigating to the above URL:</p>
<img src="static/GUI_landing.png" class="img-fluid mx-auto d-block rounded shadow p-3 mb-5 bg-white" alt="GUI landing page"> 
</div>