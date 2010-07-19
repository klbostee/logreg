from setuptools import setup, find_packages

setup(
    name = 'logreg',
    version = '0.1',
    author = 'Klaas Bosteels',
    author_email = 'klaas@last.fm',
    license = 'Apache Software License (ASF)',
    packages = find_packages(),
    zip_safe = True,
    install_requires = ['scipy','pylab']
)
