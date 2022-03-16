from setuptools import setup

setup(
    name="navdreams",
    description='NavDreams python package, including simulator and tools',
    author='Daniel Dugas',
    version='0.0.1',
    packages=["navdreams",
              ],
    python_requires='>=3.6, <3.7',
    install_requires=[
        'numpy', 'matplotlib',
    ],
)
