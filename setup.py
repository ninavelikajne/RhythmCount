import setuptools

with open("PyPi.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RhythmCount",
    version="0.1",
    author="Nina Velikajne, Miha Moškon",
    author_email="nv6920@student.uni-lj.si, miha.moskon@fri.uni-lj.si",
    description="Python package to analyse the rhythmicity in count data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ninavelikajne/RhythmCount",
    packages=setuptools.find_packages(),
    keywords = ['cosinor', 'rhytmometry', 'regression', 'count data'],
    install_requires=[
          'pandas==1.3.3',
          'numpy==1.21.2',
          'matplotlib==3.4.3',
          'statsmodels==0.12.2',
          'scipy==1.7.1'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)