from setuptools import setup

setup(name='forse',
      version='0.1',
      description='',
      url='',
      author='Nicoletta Krachmalnicoff, Giuseppe Puglisi',
      author_email='nkrach@sissa.it, gpuglisi@stanford.edu',
      license='MIT',
      packages=['forse', 'forse.networks'],
      package_dir={'forse': 'forse', 'forse.networks': 'forse/networks'},
      zip_safe=False)
