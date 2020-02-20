

Testing
''''''''''

- Install packages required for testing: 

.. code-block:: console

	pip install nose2
	pip install coverage

- Run tests and print coverage:

.. code-block:: console

	git clone https://github.com/thierrymoudiki/nnetsauce.git
	cd nnetsauce
	nose2 --with-coverage


- Obtain coverage reports:

At the command line:

.. code-block:: console

	coverage report -m

or an html report:

.. code-block:: console

	coverage html
