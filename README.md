[![Spark-DSG Build and Test](https://github.com/ntnu-arl/Spark-DSG/actions/workflows/ci.yaml/badge.svg?branch=trackgraph)](https://github.com/ntnu-arl/Spark-DSG/actions/workflows/ci.yaml)

This fork is a TrackGraph dependency. Use the `trackgraph` branch and the
[TrackGraph setup guide](https://github.com/ntnu-arl/trackgraph_ros/tree/main#setup)
for the complete workspace.

## Spark-DSG

This is the core c++ library that contains the dynamic scene graph data-structure used by TrackGraph. It also has python bindings.

### Change Notes

- 4/22/24: Python bindings and unit tests are built by default. You will need to install relevant dependencies (via `sudo apt install python3-dev` or `rosdep`) or disable the bindings and tests via cmake options.
- 3/16/24: Updated dependencies to use system libraries for `nlohmann_json`. You will need to install it either via `sudo apt install nlohmann-json3-dev` or use `rosdep` to update dependencies.

### Building for Python

  1. Install requirements and make a virtual environment:

```bash
sudo apt install python3-venv libzmqpp-dev nlohmann-json3-dev
mkdir /path/to/environment
cd /path/to/environment
python3 -m venv dsg  # or some other environment name

# you may also want to upgrade pip, though it shouldn't be necessary
# source dsg/bin/activate
# pip install --upgrade pip
```

  2. Install the python package
```bash
source /path/to/dsg/environment/bin/activate
git clone --branch trackgraph https://github.com/ntnu-arl/Spark-DSG.git
pip install ./Spark-DSG
```

### Python Bindings Usage

See [this notebook](examples/python_api.py) for some examples for the bindings (you'll want to clone the repo, even if you installed from github).
You'll want to install `jupyter` and `jupytext` if you want to run it as a notebook, though you can also just run it directly as a python script.
You can find an example scene graph [here](https://drive.google.com/file/d/1jwcjrE4-6PvOgEgJipETkQaLgC43biFT/view?usp=sharing).

### Python API documentation

Generating the python documentation should be as simple as:

```
source /path/to/dsg/environment/bin/activate
cd doc
pip install sphinx  # if you haven't already
make html
python -m http.server  # to serve them locally
```

### Building For ROS

This repository is a valid ROS package and should build if placed in a workspace.

## License

Released under BSD-3-Clause.
