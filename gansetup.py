from setuptools import setup
from setuptools import find_packages

setup(name='gan',
      version='0.1',
      description='gan',
      author='LnB',
      author_email='thomas.kipf@gmail.com',
      license='MIT',
      install_requires=['numpy',
                        'imageio',
                        'scipy'
                        ],
      packages=find_packages())
