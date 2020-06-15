import os

from setuptools import find_packages, setup # type: ignore


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as opened:
        return opened.read()


def main():
    setup(
        name='tffm2',
        version='0.0.22',
        url='https://github.com/jamborta/tffm2',
        description=('TensforFlow implementation of arbitrary order '
                     'Factorization Machine'),
        classifiers=[
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Programming Language :: Python :: 3',
        ],
        license='MIT',
        install_requires=[
            'scikit-learn',
            'numpy',
            'tqdm'
        ],
        packages=find_packages()
    )


if __name__ == "__main__":
    main()
