# Step 0

If you are on a Windows machine or plan to work on a windows machine, note: most/all instruction will be linux/ubuntu-based. You should download [WSL (Windows Subsystem for Linux)](https://ubuntu.com/wsl), which will provide you an ubuntu terminal on your windows machine.

# Step 1 - Setup

## Installing VSCode and Python Setup

* Install [VScode](https://code.visualstudio.com/download). 
* Install a package manager:
  * (Nice, but not necessary) Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
    * once installed, create a virtual environment, and then activate that virtual environment
```
conda create --name python_sandbox
conda activate python_sandbox
```

  * Make sure that [pip](https://pip.pypa.io/en/stable/installation/) is installed. If not, install it
  * After pip is installed, installed all of the necessary python packages used by this script

```
pip install matplotlib numpy
```

## Installing Extensions

  * Open up VScode. 
    * If you are on a Windows machine, follow [these instructions](https://code.visualstudio.com/docs/remote/wsl-tutorial) to make WSL your terminal in VSCoode 
    * Select your conda environment as your 'python kernel' - follow [these instructions](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management#:~:text=You%20can%20open%20the%20kernel,Notebook%3A%20Select%20Notebook%20Kernel%20command.&text=Note%3A%20In%20the%20previous%20versions,all%20available%20kernels%20by%20default.)
  * Go to the [Extensions Marketplace](https://code.visualstudio.com/docs/editor/extension-marketplace) on the left-hand side
  * Search for and install the following extensions:
    * `Jupyter Keymap` -- lets you develop jupyter notebooks locally, without any network connection.
    * `Markdown All in One` -- useful for making markdown files readable
    * `Remote SSH` -- this will be useful once we have GPU resources that we can log into

# Step 2 - Inspecting the code
  * Open up the folder containing this README and go to the .ipynb file. This is a jupyter notebook that references classes and functions in neighboring files.
  * Read `metadata.py` to see how the Metadata class stores certain variables.
  * Read the functions in `plotting.py` to see how plots are generated from input variables that are passed into each of the functions

# Step 3 - Running the code
  * You can run the code either by running each of the cells in the `Assignment0_DataExploration.ipynb` notebook or by running the `Assignment0_DataExploration.py` file. It would be good to run each of them to see how a python script behaves differently from a jupyter notebook.
    * Jupyter notebooks are typically good for data inspection and debugging, while python scripts are typically useful when running one process after another.