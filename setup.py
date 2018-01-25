from setuptools import setup, find_packages

if __name__ == '__main__':
  setup(name='trainer',
  packages = find_packages(),
  install_requires=[
      'keras',
      'h5py',
    'matplotlib.pyplot'
  ],
  zip_safe=False
        )
