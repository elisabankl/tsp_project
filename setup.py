from setuptools import setup, find_packages

setup(
    name='tsp-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for solving the asymmetric Traveling Salesman Problem with precedence constraints using reinforcement learning.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gymnasium',
        'numpy',
        'stable-baselines3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)