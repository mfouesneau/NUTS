from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name = "nuts",
    version = 1.0,
    description = "No-U-Turn Sampler (NUTS) implementation for python",
    long_description = readme(),
    author = "Morgan Fouesneau",
    author_email = "",
    url = "https://github.com/mfouesneau/NUTS",
    packages = find_packages(),
    package_data = {},
    include_package_data = True,
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    zip_safe=False
)
