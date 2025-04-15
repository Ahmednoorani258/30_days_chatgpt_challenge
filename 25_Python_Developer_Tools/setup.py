# 5. Packaging: Sharing Your Code Toy 🎁
# What’s Packaging?
# Packaging is like putting your toy in a box with instructions so your friend can play with it. In coding, it means making your project a package others can install and use.

# Definition: Packaging is turning your code into a shareable bundle.
# What’s setup.py?
# setup.py is a file that tells Python how to package your code, like a label on your toy box.

from setuptools import setup, find_packages

setup(
    name='my_cool_project',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)