---
layout: home
permalink: /faq/
title: Seldonian \| FAQ
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
<h1 class="mb-3">Frequently Asked Questions</h1>

<hr class="my-4">
<h5 class="mb-3"><b>Does this toolkit support deep learning?</b></h5>
Yes. We support PyTorch models (see the <a href="{{ "/tutorials/pytorch_mnist" | relative_url }}">Creating your first Seldonian PyTorch model tutorial</a>), and we are actively working on supporting other deep learning libraries, starting with Tensorflow. If you have a potential deep learning application and are not sure where to start, please see our <a href="{{ "/support" | relative_url }}">Support</a> page, where you can find information for contacting us or opening an issue on Github. 

<hr class="my-4">
<h5 class="mb-3"><b>Why did you choose the write the toolkit in Python 3?</b></h5>
Machine learning researchers usually use and are most familiar with Python 3. Writing the entire toolkit in Python therefore makes it easier for researchers to extend and improve all of the methods in the toolkit, facilitating further research into safe and fair machine learning. 

<hr class="my-4">
<h5 class="mb-3"><b>What software does the toolkit contain?</b></h5>
The Seldonian Toolkit has four components (four repositories): <code>Engine</code>, <code>Experiments</code>, <code>Tutorials</code>, and <code>GUI</code>. The <code>Engine</code> library contains the code necessary to run the provided Seldonian algorithm and does not rely on the other libraries. The <code>Experiments</code> library uses the <code>Engine</code> repository and provides a framework for experimenting with and evaluating Seldonian and other machine learning algorithms. The <code>Tutorials</code> repository contains tutorials and examples that use the other libraries. The <code>GUI</code> repository contains the code to run the Seldonian Interface GUI, an example Seldonian interface written in Flask, a Python web framework.

<hr class="my-4">
<h5 class="mb-3"><b>Can I contribute?</b></h5>
Yes! This project is open source and hosted as several public repositories on GitHub <a href="https://github.com/seldonian-toolkit">here</a>. You are welcome to fork the project repositories and submit pull requests. We welcome both minor fixes and new functionality. If you have questions about the toolkit, please see the <a href="{{ "/support" | relative_url }}">Support</a> page.

<hr class="my-4">
<h5 class="mb-3"><b>Who funded this work?</b></h5>
This toolkit was primarily funded by the <a href="https://existence.org/">Berkeley Existential Risk Initiative (BERI)</a> via a donation from <a href="https://funds.effectivealtruism.org/">Effective Altruism Funds (EA Funds)</a> through their <a href="https://funds.effectivealtruism.org/funds/far-future">Long-Term Future Fund</a>. In the interest of transparency, we also note that several contributions also came from <a href="https://people.cs.umass.edu/~pthomas/">Prof. Thomas</a> and students from the <a href="https://www.cics.umass.edu/">Manning College of Information and Computer Sciences</a> at the University of Massachusetts. During the time of this project, Prof. Thomas and the students were funded by a variety of grants and gifts from the <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2018372">National Science Foundation (NSF)</a>, <a href="https://research.adobe.com/data-science-research-awards/"  data-bs-toggle="tooltip" data-bs-placement="bottom" title="Prof. Thomas received additional gift funding from Adobe Research beyond the linked Data Science Research Awards">Adobe Research</a>, <a href="https://research.facebook.com/blog/2021/12/announcing-the-winners-of-the-building-tools-to-enhance-transparency-in-fairness-and-privacy-rfp/" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Project title: High-confidence long-term safety and fairness guarantees">Meta Research</a>, <a href="https://research.google/outreach/air-program/recipients/" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Project title: Supervised Learning with Long-Term Fairness Guarantees">Google Research</a>, <a href="https://www.rtx.com/">Raytheon</a>, and the <a href="https://www.arl.army.mil/business/collaborative-alliances/current-cras/iobt-cra/">Army Research Laboratory (ARL)</a>.
</div>

