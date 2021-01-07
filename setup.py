from distutils.core import setup
setup(
  name = 'RhythmCount',
  packages = ['RhythmCount'],
  version = '0.0',
  license='MIT',
  description = 'Python package for cosinor based rhytmometry in count data',
  author = 'Nina Velikajne, Miha Moškon',
  author_email = 'nv6920@studnet.uni-lj.si, miha.moskon@fri.uni-lj.si',
  url = 'https://github.com/ninavelikajne/RhythmCount',
  download_url = 'https://github.com/ninavelikajne/RhythmCount/archive/v0.0.tar.gz',
  keywords = ['cosinor', 'rhytmometry', 'regression', 'count data'],
  install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
          'statsmodels',
          'scipy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)