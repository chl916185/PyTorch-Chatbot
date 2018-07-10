# PyTorch Chatbot

## Setting up an environment

Create a virtual environment.

``` shell
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $
```

Install the package.

``` shell
(.venv) $ pip install -e .
```


Next, install continous integration dependencies in order to run tests.

``` shell
(.venv) $ pip install -r requirements/ci.txt
```

You should be able to run `make run` now. There is also `make clean`, `make lint`, and `make test`.

If you want to leave the virtual environment, do the following.

``` shell
(.venv) $ deactivate
$
```
