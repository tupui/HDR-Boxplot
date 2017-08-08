Introduction
============

If you are reading this, first of all, thank you and welcome to this community.
For everyone to have fun, every good python projects requires some guidelines
to be observed.

Isn't it frustrating when you cannot understand some code just because there is
no documentation nor any test to assess that the function is working nor any
comments in the code itself? How are you supposed to code in these conditions?

Following these guidelines helps to communicate that you respect the time of 
the developers managing and developing this open source project. In return, 
they should reciprocate that respect in addressing your issue, assessing 
changes, and helping you finalize your pull requests.

What can I do?
==============

We love to receive contributions from our community — you! There are many ways
to contribute, from writing tutorials, improving the documentation, submitting
bug reports and feature requests or writing code which can be incorporated into
the project itself.

Please, don't use the issue tracker for *support questions* about python.
Stack Overflow is worth considering.

Getting Started
===============

Additionnal dependencies are requiered for testing: 

- `pytest <https://docs.pytest.org/en/latest/>`_
- mock
- `coverage <http://coverage.readthedocs.io>`_

1. Get the latest version of the code,
2. Launch tests to ensure all are passing: `pytest .`,
3. Do your modification :)

How to Report a Bug
===================

When filling an issue, make sure to answer these five questions:

1. What version of python are you using?
2. Have you updated all dependencies? If not, why and what are the versions?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

Your First Contribution
=======================

Thank you again if you are reading this! Help is essential. If you want to add
a modification, create a new branch. From here, the fun beggins. You can commit
any change you feel, start discussions about it in the PR, etc.

Check List
----------

Your request will only be considered for integration if in a **finished** state: 

0. Respect python coding rules,
1. The branch passes all tests,
2. Have tests regarding the changes,
3. Maintain test coverage,
4. Have the respective documentation.

For the testing part, prove your claims with a *pytest* and coverage reports.

In case, here are some quick references about python and testing.

Python
------

This is a python project, not some *C* or *Fortran* code. You have to adapt your
thinking to the python style. Otherwise, this can lead to performance issues.
For example, an ``if`` is expensive, you would be better off using a ``try except``
construction. *It is better to ask forgiveness than permission* ;). Also, when
performing computations, care to be taken with ``for`` loops. If you can, use
*numpy* operations for huge performance impact (sometimes x1000!).

Thus developers **must** follow guidelines from the Python Software Foundation.
As a quick reference:

* For text: `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
* For documentation: `PEP 257 <https://www.python.org/dev/peps/pep-0257/>`_
* Use reStructuredText formatting: `PEP 287 <https://www.python.org/dev/peps/pep-0287/>`_

And for a more Pythonic code: `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_
Last but not least, avoid common pitfalls: `Anti-patterns <http://docs.quantifiedcode.com/python-code-patterns/>`_

Testing
-------

Testing your code is paramount. Without continuous integration, you **cannot**
guaranty the quality of the code. Some minor modification on a module can have
unexpected implications. With a single commit, everything can go south!
The ``master`` branch is always on a passing state. This means that you should
be able to checkout from them an use the package without any errors.

The library `pytest <https://docs.pytest.org/en/latest/>`_ is used. It is simple
and powerfull. Checkout their doc and replicate constructs from existing tests.
If you are note already in love with it, you will soon be. All tests can be
launched using::

    coverage run -m pytest .
    coverage report -m

These commands fire `coverage <http://coverage.readthedocs.io>`_ in the same time.
The output consists in tests results and coverage report.
