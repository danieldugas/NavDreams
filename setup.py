from setuptools import setup

setup(
    name="navdreams",
    description='NavDreams python package, including simulator and tools',
    author='Daniel Dugas',
    version='0.0.8',
    packages=["navdreams",
              ],
    python_requires='>=3.6',
    install_requires=[
        'numpy', 'matplotlib',
        'pyrvo2-danieldugas',
        'matplotlib',
        'ipython',
        'pyyaml',
        'stable-baselines3',
        'pyglet',
        'navrep',
        'typer',
        'pandas',
        'strictfire',
        'pyniel',
        'mlagents',
        'jedi', # recommend 0.17 jedi because newer versions break ipython (py3.6)
        'gym<=0.18.0', # recommend 0.18.0 due to  gym error in (py3.6) https://stackoverflow.com/questions/69520829/openai-gym-attributeerror-module-contextlib-has-no-attribute-nullcontext # noqa
    ],
)
