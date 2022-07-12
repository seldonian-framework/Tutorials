---
layout: home
permalink: /faq/
---

<!-- Main Container -->
<div class="container p-3 my-5 border" style="background-color: #f3f4fc;">
<h1 class="mb-3">Frequently Asked Questions</h1>

<hr class="my-4">
<h5 class="mb-3"><b>Does this library support deep learning?</b></h5>
Yes, this library supports deep learning using <a href="https://pytorch.org/">PyTorch</a>. However, due to the added complexity of enforcing safety and fairness constraints while training, large neural networks like those used for language models and some computer vision tasks are unlikely to be computationally infeasiable. This will be one priority for future versions.

<hr class="my-4">
<h5 class="mb-3"><b>Why did you choose the write the library in Python 3?</b></h5>
Machine learning researchers usually use and are most familiar with Python 3. Writing the entire library in Python therefore makes it easier for researhers to extend and improve all of the methods in the library, facilitating further research into safe and fair machine learning. 

<hr class="my-4">
<h5 class="mb-3"><b>Can the engine be used on its own?</b></h5>
Yes! As discussed in the <a href="overview.html">overview</a> section of this page, this library has three components (three repositories): <code>Engine</code>, <code>Tutorials</code>, and <code>experiment-framework</code>. The <code>Engine</code> repository contains the code necessary to run the provided Seldonian algorithm, and does not rely on the other two reposiories. The <code>Tutorials</code> repository uses the <code>Engine</code> repository, and provides tutorials and examples. The <code>experiment-framework</code> also uses the <code>Engine</code> repository, and provides a framework for experimenting with and evaluating Seldonian and other machine learning algorithms.

<hr class="my-4">
<h5 class="mb-3"><b>How can I contribute?</b></h5>
Yes! This project is open source and hosted as three repositories on GitHub <a href="https://github.com/seldonian-toolkit">here</a>. You are welcome to fork the project repositories and submit pull requests. We welcome both minor fixes and new functionality. If you have questions about the library, please see the <a href="support.html">support</a> page.

<hr class="my-4">
<h5 class="mb-3"><b>Who funded this library?</b></h5>
This library was primarily funded by the <a href="https://existence.org/">Berkeley Existential Risk Initiative (BERI)</a> via a donation from <a href="https://funds.effectivealtruism.org/">Effective Altruism Funds (EA Funds)</a> through their <a href="https://funds.effectivealtruism.org/funds/far-future">Long-Term Future Fund</a>. In the interest of transparency, we also note that several contributions also came from <a href="https://people.cs.umass.edu/~pthomas/">Prof. Thomas</a> and students from the <a href="https://www.cics.umass.edu/">Manning College of Information and Computer Sciences</a> at the University of Massachusetts. During the time of this project, Prof. Thomas and the students were funded by a variety of grants and gifts from the <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2018372">National Science Foundation (NSF)</a>, <a href="https://research.adobe.com/data-science-research-awards/"  data-bs-toggle="tooltip" data-bs-placement="bottom" title="Prof. Thomas received additional gift funding from Adobe Research beyond the linked Data Science Research Awards">Adobe Research</a>, <a href="https://research.facebook.com/blog/2021/12/announcing-the-winners-of-the-building-tools-to-enhance-transparency-in-fairness-and-privacy-rfp/" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Project title: High-confidence long-term safety and fairness guarantees">Meta Research</a>, <a href="https://research.google/outreach/air-program/recipients/" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Project title: Supervised Learning with Long-Term Fairness Guarantees">Google Research</a>, <a href="https://www.rtx.com/">Raytheon</a>, and the <a href="https://www.arl.army.mil/business/collaborative-alliances/current-cras/iobt-cra/">Army Research Laboratory (ARL)</a>.
</div>

