# NapShift (version 1.0.0)

![Logo](./logos/Logo_main.png)

This repository provides a "Python implementation" of the NapShift program
to estimate the backbone atoms' chemical shift values from NMR protein PBD
files.

M. Vrettas, PhD.

## Installation

There are two options to install the software.

1. The easiest way is to visit the GitHub web-page of the project and
[download the code](https://github.com/vrettasm/NapShift/archive/master.zip)
in zip format. This option does not require a prior installation of git on the
computer.

2. Alternatively one can clone the project directly using git as follows:

    `$ git clone https://github.com/vrettasm/NapShift.git`

## Required packages

The recommended version is **Python 3.7** (and above). Some required packages
are:

> tensorflow, numpy, pathlib, pandas, etc.

To simplify the required packages just use:

    $ pip install -r requirements.txt

## Virtual environment (recommended)

It is highly advised to create a separate virtual environment to avoid
messing with the main Python installation. On Linux and macOS systems
type:

    $ python3 -m venv napshift_venv

Note: "napshift_venv" is an _optional_ name.

Once the virtual environment is created activate it with:

    $ source napshift_venv/bin/activate

Then we can install all the requirements as above:

    $ pip3 install -r requirements.txt

or

    $ python3 -m pip install -r requirements.txt

N.B. For Windows systems follow the **equivalent** instructions.

## How to run

To execute the program (within the activated virtual environment), you can either
navigate  to the main directory of the project (i.e. where the napshift.py is located),
or locate it through the command line and then run the following command:

    $ ./napshift.py -f path/to/filename.pdb

This is the simplest way to run NapShift. It will create a file named:
"prediction_filename_model_0_chain_A.tab" in the _current working directory_,
with the predicted chemical shift values for all backbone atoms (N, C, Ca, Cb, H, Ha).

   > **Hint**: If you want to run the program on multiple files (in the same directory) you
   > can use the '*' wildcard as follows:
   >  
   > $ ./napshift.py -f path/to/*.pdb

This will run NapShift on all the files (in the directory) with the '.pdb' extension.

---

To explore all the options of NapShift, use:

    $ ./napshift.py -h

You will see the following menu:

![Help](./logos/Help_menu.png)

## References (and documentation)

The original work is described in detail at:

1. Guowei Qi, Michail D. Vrettas, Carmen Biancaniello, Maximo Sanz-Hernandez,
Conor T. Cafolla, John W. R. Morgan, Yifei Wang, Alfonso De Simone, and
David J. Wales (2022).
_"Enhancing Biomolecular Simulations With Hybrid Potentials Incorporating NMR Data."_
Accepted for publication at Journal of Chemical Theory and Computation.

2. The documentation of the code can be found in: [NapShift_v01-doc](./docs/NapShift_v01.pdf)

### Contact

For any questions/comments (**_regarding this code_**) please contact me at:
vrettasm@gmail.com