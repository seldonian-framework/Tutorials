---
layout: home
permalink: /contest/
title: Seldonian \| Competition
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
<h1 class="mb-2 text-center">Seldonian Toolkit Competition</h1>
    <div class="container p-3 text-center text-secondary" style="background-color: #f3f4fc;">
            <img src="{{ "/assets/img/contest/STC.png" | relative_url }}" class="img-fluid rounded" alt="Seldonian Toolkit Competition Logo" width="25%">
            <br>
            DALL-E's interpretation of "An abstract art painting with hints of technology and safety."
    </div>
<hr class="my-2">
<p>
        <b>Description.</b> The first Seldonian Toolkit Competition took place in March and April 2023. The contest was designed as a way for undergraduate and graduate students in the US and Canada to learn about and develop safe and fair machine learning algorithms using the recently launched Seldonian Toolkit. This toolkit was created to make it easier for data scientists to apply machine learning responsibly: with high-confidence safety and fairness constraints on the machine learning system's behavior.
    </p>
    
    <p> This competition was intentionally open ended -- any application of the toolkit that includes a high-confidence constraint was acceptable. Students picked topics that they found interesting. 
    </p>
    <p>
        <b>Talks.</b> To help give participants a more complete view of AI safety and fairness topics, we invited speakers who are not working with Seldonian algorithms, but who are still studying issues related to safety and fairness.
    </p>
    <p>
        <b>Important Dates.</b>
        <ul>
            <li>
                <p>
                    <b>[Early Registration] March 2, 2023</b>: Participating teams should register by midnight (anywhere on Earth) on March 2, 2023. Registration is free and non-binding. Teams can register right now by clicking the button below and filling out the provided form.
                </p>
            </li>
            <li style="margin: 10px 0;"><b>[Kick-Off Event] March 6, 2023</b>: This event will consist of:
            <ul>
                <li>Prof. Philip Thomas providing a high-level introduction to AI safety and fairness topics using Seldonian algorithms. </li>
                <li>Dr. Austin Hoag providing coding examples of how the Seldonian Toolkit can be used. </li>
            </ul>  
            This event will be held from 5:30pm - 7:00pm Eastern Time on Zoom (<a href="https://umass-amherst.zoom.us/j/99288787234">link</a>), though participants are welcome to attend in-person at UMass Amherst (CS Building Room 151).</li>
            <li style="margin: 10px 0;">
                <b>[Late Registration] March 10, 2023</b>: Participating teams <b>must</b> register by midnight (anywhere on Earth) on March 10, 2023 in order to be eligible for any awards. Registration is free and non-binding. We strongly encourage teams to register before the March 2 deadline so that they can receive information about the timing and location of the Kick-Off Event. However, registration by March 10 is sufficient to participate in the contest.
            </li>
            <li style="margin: 10px 0;"><b>[Early-Contest Q&A] March 17, 2023, 3pm Eastern</b>: Prof. Philip Thomas and Dr. Austin Hoag will host an open Q&A session for participating teams. This event will take place at 3pm Eastern at <a href="https://umass-amherst.zoom.us/j/91864345408">this Zoom link</a>.</li>
            <li style="margin: 10px 0;"><b>[Invited Talk: <a href="https://people.umass.edu/yzick/">Prof. Yair Zick</a>] March 22, 2023, 1pm Eastern</b>: Prof. Zick will present a talk titled "A Simple, General Framework for Fair Allocation Under Matroid Rank Valuations." The talk will be hosted on <a href="https://umass-amherst.zoom.us/j/98544373996">Zoom</a>, as well as in-person in room 203 of the CS building at UMass. There is limited seating, so people interested in attending in-person, please confirm with Prof. Thomas that a seat is available.</li>
            <li style="margin: 10px 0;"><b>[Mid-Contest Q&A] March 27, 2023, 4pm Eastern</b>: Prof. Philip Thomas and Dr. Austin Hoag will host an open Q&A session for participating teams. This event will take place at 4pm Eastern at <a href="https://umass-amherst.zoom.us/j/97492091365">this Zoom link</a>.</li>
            <li style="margin: 10px 0;"><b>[Invited Talk: <a href="https://vaelgates.com/">Dr. Vael Gates</a>] March 28, 2023, 5pm Eastern</b>: Dr. Gates will present at <a href="https://stanford.zoom.us/j/8124240568?pwd=R21wcHkwZ0NIOXJVUCtyaUJrVVViUT09">this Zoom link</a>.</li>
            <li style="margin: 10px 0;"><b>[Late-Contest Q&A] April 13, 2023, 4pm Eastern</b>: Prof. Philip Thomas and Dr. Austin Hoag will host an open Q&A session for participating teams. This event will take place at 4pm Eastern at <a href="https://umass-amherst.zoom.us/j/98566248497">this Zoom link</a>.</li>
            <li style="margin: 10px 0;"><b>[Invited Talk: <a href="https://www.thomaskrendlgilbert.com/">Dr. Thomas Gilbert</a>] April 19, 2023, 4pm Eastern</b>: Dr. Thomas Gilbert will present a talk titled "Reward Reports for Reinforcement Learning" at <a href="https://umass-amherst.zoom.us/j/93438890163">this Zoom link</a></li>
            <li style="margin: 10px 0;">
                <b>[Final Submissions] April 21, 2023</b>: Final submissions are due at midnight (anywhere on Earth) on April 21, 2023. See the "Submissions" section below for more information about what should be submitted.
            </li>
            <li style="margin: 10px 0;">
                <b>[Award Announcement] April 28, 2023</b>: We aim to announce the winners of the competition by April 28, 2023. The exact date will depend on the volume of submissions.
            </li>
        </ul>
    </p>
</div>

<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h3 class="mb-3" id="framework">Participation and Submission</h3>
    <hr class="my-4" />
    <p>
        Participating teams should select an application of the Seldonian Toolkit. While there are no restrictions on the allowed applications, we recommend that you select an application for which you have access to training data and which your team members are familiar with. This could range from predicting how far landslides will travel based on features of the slope, with safety guarantees related to the chance of under-predictions, to predicting whether a tumor is benign or malignant with safety guarantees with respect to the false negative rate, to predicting whether someone will commit a crime in the future while enforcing fairness constraints with respect to race, gender, or age. Some teams might already have applications in mind, while others might begin by brainstorming possible applications. For teams still trying to select an application, we recommend searching for datasets that relate to machine learning problems where safety or fairness guarantees would be beneficial.
    </p>
    <p>
        After selecting an application, teams should apply the Seldonian Toolkit. In almost all cases, teams should use the Experiments component of the toolkit to show how effective the Seldonian Toolkit is for their application. The Experiments component is described in <a href="https://seldonian.cs.umass.edu/Tutorials/tutorials/fair_loans_tutorial/">this</a> tutorial. It provides plots that show how accurate the learned models are, how much data was required before the system could reliably return solutions, and how often the system violated the desired safety or fairness constraints. 
    </p>
    <p>
        Next, teams should put together a report describing their application, its importance, how the Seldonian Toolkit was applied, and the resulting performance of their system. This report should be provided as a markdown (.md) file in a GitHub repository that contains the source code for the project. The markdown file should clearly indicate the team name somewhere near the top. Each team should then fill out the submission form linked below, which asks for a link to the GitHub repository. <a href="https://github.com/mhyeh/Fairness-for-Lie-Detection">This</a> is an example of what a team might submit. 
    </p>
</div>

<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h3 class="mb-3" id="framework">Awards and Evaluation Criteria</h3>
    <hr class="my-4" />
    <p> The panel of judges, consisting of AI faculty from UMass Amherst (Professors Philip S. Thomas, Bruno Castro da Silva, and Scott Niekum), Stanford University (Professor Emma Brunskill), and Brown University (Professor George Konidaris), selected the following two submissions as a tie for the "Best Overall Student Project" award (each will receive $600).</p>
    <ul>
        <li><b>Fairness for Breast Cancer Recurrence Prediction</b> by Derek Lacy: <a href="https://github.com/d1lacy/Fairness-for-Breast-Cancer-Recurrence-Prediction">Link</a>.</li>
        <li><b>Fairness in Student Course Completion Based on Student Data</b> by Sahil Yerawar, Pranay Reddy, and Varad Pimpalkhute: <a href="https://github.com/pranay-ar/UnfairSlayers_Seldonian_oulad">Link</a>.</li>
    </ul>
    These projects can now be found on the <a href="{{ "/examples/" | relative_url }}">examples</a> page.
</div>

<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h3 class="mb-3" id="framework">Support</h3>
    <hr class="my-4" />
    <p> Participants are encouraged to post questions on the GitHub issues pages [links: <a href="https://github.com/seldonian-toolkit/Engine/issues">Engine</a>, <a href="https://github.com/seldonian-toolkit/Experiments">Experiments</a>, and <a href="https://github.com/seldonian-toolkit/GUI">GUI</a>]. We will do our best to answer these questions in a timely manner. For questions related to this competition but not directly related to the use of the Seldonian Toolkit, we encourage teams to ask during the kick-off event or the various Q&A sessions held throughout the contest. Teams can also email <a href="mailto:Seldonian@cs.umass.edu">Seldonian@cs.umass.edu</a>. However, responses to these emails may be slow depending on the volume of participants and questions.</p>
    <p>
        The UMass Data Science club has also created a Discord server where participants can interact with each other.
    </p>
    <div class="col-md-12 text-center"> 
        <a href="https://discord.gg/3hsH7mrDCM" class="btn btn-primary center-block">Discord Server</a>
    </div>
</div>

<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
    <h3 class="mb-3" id="framework">Sponsors</h3>
    <hr class="my-4" />
    <p> This contest is a collaboration between the Autonomous Learning Laboratory (ALL) at the University of Massachusetts and the Berkeley Existential Risk Initiative (BERI). The awards will be provided by BERI.</p>
</div>
