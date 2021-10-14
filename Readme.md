###########################################################Prerequisite#########################################################################
------------------------------------------------------------------------------------------------------------------------------------------------
Pycharm or python ide
Anaconda
python 3.8
pip installed
------------------------------------------------------------------------------------------------------------------------------------------------
###########################################################Dataset#############################################################################
------------------------------------------------------------------------------------------------------------------------------------------------
training data set is in folder COMS6200_TEAM_PROJECT/data/UNSW_NB15_training-set.csv
testing data set is in folder COMS6200_TEAM_PROJECT/data/UNSW_NB15_testing-set.csv
------------------------------------------------------------------------------------------------------------------------------------------------
###########################################################Anaconda jupyter notebook############################################################
------------------------------------------------------------------------------------------------------------------------------------------------
jupyter notebook file: COMS6200_TEAM_PROJECT/src/coms6200 project.ipynb
ensure you set working directory in %/COMS6200_TEAM_PROJECT/src/
open coms6200 project.ipynb
ensure the dataset exist in COMS6200_TEAM_PROJECT/data/
for any packet that is not import in anaconda, follow the below example command and run it at cell
# import sys
# !{sys.executable} -m pip install xxxx_packet
--------------------------------------------------
#if xgboost is not imported
import sys
{sys.executable} -m pip install xgboost
--------------------------------------------------
coms6200 project.ipynb's cell are not independent, they are in sequence order, so maybe try run all instead random pick and run
coms6200 project.ipynb is only for data preprocess and hyperparameter tuning, the good parameter and dataset will passed to data generator
manually copy the hyperparameter and data preprocess code to the data generator file
------------------------------------------------------------------------------------------------------------------------------------------------
###########################################################Data generator#######################################################################
------------------------------------------------------------------------------------------------------------------------------------------------
data generator file: COMS6200_TEAM_PROJECT/src/Model_evaluation_data_generator.py
ensure you install all the imported packege
click run to generate the file
output file: COMS6200_TEAM_PROJECT/data/data.json
------------------------------------------------------------------------------------------------------------------------------------------------
###########################################################Website visualize####################################################################
------------------------------------------------------------------------------------------------------------------------------------------------
   
------------------------------------------------------------------------------------------------------------------------------------------------
