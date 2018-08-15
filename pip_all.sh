#!/bin/bash
#
##################
## Instructions ##
##################
# first do this:
#  $ python3 -m venv venv
#  $ source venv/bin/activate
#
# then run this:
#  (venv) $ ./pip_all.sh
#

##########################################
## Verify we are running the proper pip ##
##########################################
which pip
pip install --upgrade pip

###########################################
## Install Jupyter notebook for e2e test ##
###########################################
pip install jupyter
pip install ipykernel
python3 -m ipykernel install --user

########################################
## Install additional python packages ##
########################################
python3 -m pip install numpy 
python3 -m pip install scipy 
python3 -m pip install matplotlib 
#python3 -m pip install ipython 
python3 -m pip install pandas 
python3 -m pip install sympy 
#python3 -m pip install nose
python3 -m pip install opencv-python
python3 -m pip install scikit-learn
python3 -m pip install tensorflow
python3 -m pip install tqdm
python3 -m pip install scikit-image

###########################
## Install PEP8 checkers ##
## - Pylint              ##
## - pycodestyle         ##
## - pep8                ##
###########################
pip install pylint
pip install pep8
pip install pycodestyle


