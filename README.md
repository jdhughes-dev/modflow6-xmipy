# xmipy

![Continuous integration](https://github.com/Deltares/xmipy/workflows/Continuous%20integration/badge.svg)
[![codecov](https://codecov.io/gh/Deltares/xmipy/branch/develop/graph/badge.svg)](https://codecov.io/gh/Deltares/xmipy)


`xmipy` is an extension to [bmipy](https://pypi.org/project/bmipy/) including an implementation of the abstract methods.
The extended interface is required to couple certain hydrological kernels, particularly MODFLOW 6. Currently it is a joint development of the USGS and Deltares. The [imod_coupler](https://github.com/Deltares/imod_coupler) uses it, for example, to couple MODFLOW 6 and MetaSWAP.

`xmipy` can be downloaded from [conda-forge](https://anaconda.org/conda-forge/xmipy) or [PyPI](https://pypi.org/project/xmipy/).

This branch is a BMI 3.0-alpha1 prototype. It tracks the `bmi-3-development`
branch of [csdms/bmi](https://github.com/csdms/bmi), requires bmipy from the
`bmi-3` branch of [jdhughes-dev/bmi-python](https://github.com/jdhughes-dev/bmi-python),
and needs a libmf6 built from the `bmi-3` branch of
[jdhughes-dev/modflow6](https://github.com/jdhughes-dev/modflow6).

# Contributing

In order to develop on `xmipy`, you have to download pixi.
Pixi can be downloaded at [pixi.sh](https://pixi.sh/latest/).

In order to run the test suite, execute:

```bash
pixi run tests
```
