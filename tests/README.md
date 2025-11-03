# splitutils unit tests

* set up a virtual environment `venv_test_splitutils`, activate it, and install requirements. For Linux this works e.g. as follows:

```bash
$ virtualenv --python="/usr/bin/python3" venv_test_splitutils
$ source venv_test_splitutils/bin/activate
(venv_test_splitutils) $ pip install -r requirements.txt
```

# run tests

* all tests

```bash
(venv_test_splitutils) $ pytest
```

* single test, e.g.

```bash
(venv_test_splitutils) $ pyptest test_traindevtest_split.py
```






