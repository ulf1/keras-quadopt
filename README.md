[![PyPI version](https://badge.fury.io/py/keras-quadopt.svg)](https://badge.fury.io/py/keras-quadopt)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/satzbeleg/keras-quadopt.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/satzbeleg/keras-quadopt/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/satzbeleg/keras-quadopt.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/satzbeleg/keras-quadopt/context:python)

# keras-quadopt
Solving quadratic optimization problems with resource constraints and upper boundaries using TF2/Keras.

## Usage

```py
import tensorflow as tf
import keras_quadopt as kqp
import time

# goodness scores
good = tf.constant([.51, .53, .55, .57])

# similarity matrices
simi_1 = tf.constant([
    [1, .9, .8, .7],
    [.9, 1, .6, .5],
    [.8, .6, 1, .4],
    [.7, .5, .4, 1],
])

simi_2 = tf.constant([
    [1, .7, .8, .3],
    [.7, 1, .4, .2],
    [.8, .4, 1, .6],
    [.3, .2, .6, 1],
])

# preference parameters
lam = 0.4
beta_1 = 0.25
beta_2 = 0.75

# compute weights
simi = kqp.aggregate_matrices(simi_1, beta_1, simi_2, beta_2)

start = time.time()
wbest, fbest = kqp.get_weights(good, simi, lam)
print(f"elapsed: {time.time() - start}")
```




## Appendix

### Installation
The `keras-quadopt` [git repo](http://github.com/satzbeleg/keras-quadopt) is available as [PyPi package](https://pypi.org/project/keras-quadopt)

```sh
pip install keras-quadopt
pip install git+ssh://git@github.com/satzbeleg/keras-quadopt.git
```

### Install a virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
pip install -r requirements-dev.txt --no-cache-dir
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

### Python commands

* Check syntax: `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
* Run Unit Tests: `PYTHONPATH=. pytest`

Publish

```sh
pandoc README.md --from markdown --to rst -s -o README.rst
python setup.py sdist 
twine upload -r pypi dist/*
```

### Clean up 

```sh
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .pytest_cache
rm -r .venv
```


### Support
Please [open an issue](https://github.com/satzbeleg/keras-quadopt/issues/new) for support.


### Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/satzbeleg/keras-quadopt/compare/).
