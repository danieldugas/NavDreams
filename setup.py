from setuptools import setup

setup(
    name="navrep3d",
    description='navigation representations in 3D environments',
    author='Daniel Dugas',
    version='0.0.1',
    packages=["navrep3d",
              ],
    python_requires='>=3.6, <3.7',
    install_requires=[
        'numpy', 'matplotlib',
    ],
)
