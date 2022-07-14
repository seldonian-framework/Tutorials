---
layout: tutorial
permalink: /tutorials/install_engine_tutorial/
next_url: /tutorials/simple_engine_tutorial/
next_page_name: Simple Engine tutorial
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h1 class="mb-3">Tutorial: Installing the Seldonian Engine</h1>
    <hr class="my-4">
    <h3> Installation </h3>
    <p> 
        The Seldonian Engine can be installed in a single line:
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
        make sure that the python executable you are using to run this command is the same one you used to do the "pip install" command above. You can see the full path of your python command by running:
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
</div>