---
layout: home
permalink: /tutorials/install_engine_tutorial/
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
        $ python -m pip install --upgrade seldonian-engine
        </code>
    </p>
    <p>
        This will make the "seldonian" package available for you to import in Python. To test that it worked, run this command from the command line:
    </p>
    <p> <code>
        $  python -c "from seldonian import parse_tree,candidate_selection,safety_test"
        </code>  </p>
    <p>
        If you get this error:
        <code> 
        ModuleNotFoundError: No module named 'seldonian'
        </code> make sure that the python executable you are using to run this command is the same one you used to do the "pip install" command above. You can see the full path of your python command by running:
    </p>
    <p>
        <code>
        $ which python 
        </code>
    </p> 
</div>