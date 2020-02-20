

Guidelines
''''''''''

Your contributions to the source code are welcome, and valuable. Please, make sure to **read** the `Code of conduct <https://github.com/thierrymoudiki/nnetsauce/blob/try-sphinx/CONTRIBUTING.md>`_ first. If you're not comfortable with Git/Version Control yet, please use `this form <https://forms.gle/HQVbrUsvZE7o8xco8>`_ to provide a feedback.

You can also contribute with some examples: `notebooks <https://github.com/thierrymoudiki/nnetsauce/tree/master/nnetsauce/demo>`_ or `flat files <https://github.com/thierrymoudiki/nnetsauce/tree/master/examples>`_. With the following naming convention:  ``yourgithubname_ddmmyy_shortdescriptionofdemo.[py|ipynb|R|Rmd]``.

In `Pull Requests <https://thierrymoudiki.github.io/blog/2020/02/14/misc/git-github>`_, let's strive to use `black <https://black.readthedocs.io/en/stable/>`_ for formatting files: 

.. code-block:: console

	pip install black
	black --line-length=80 file_submitted_for_pr.py
