from distutils.core import setup

setup(
    name='resonances-ml',
    version='',
    packages=['resonancesml', 'resonancesml.commands'],
    url='',
    license='',
    author='ANtlord',
    author_email='',
    description='',
    install_requires = [
        'click==6.6',
        'scikit-learn==0.17.1',
        'numpy==1.11.1',
        'scipy==0.18.0',
        'texttable==0.8.4',
        'pandas==0.18.1',
        'matplotlib==1.5.1',
        'imbalanced-learn==0.1.8'
    ],
)
