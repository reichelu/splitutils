# test pypi package splitutils

```bash
$ virtualenv --python="/usr/bin/python3" pypi_splitutils
$ source pypi_splitutils/bin/activate

# install from test.pypi
$ python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple splitutils

# install from pypi
(pypi_splitutils) $ pip install splitutils

# run test script
(pypi_splitutils) $ python test_traindevtest_split_swap_prt.py
```