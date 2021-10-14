Prerequisite
------------------------------------------------------------------------------------------------------------------------------------------------
Ensure you have following item:<br />
1.Pycharm or python ide<br />
2.Anaconda<br />
3.Python 3.8<br />
4.Pip installed<br />
------------------------------------------------------------------------------------------------------------------------------------------------
Dataset
------------------------------------------------------------------------------------------------------------------------------------------------
Training data set is in folder ->../data/UNSW_NB15_training-set.csv<br />
Testing data set is in folder -> ../data/UNSW_NB15_testing-set.csv<br />
------------------------------------------------------------------------------------------------------------------------------------------------
Anaconda jupyter notebook
------------------------------------------------------------------------------------------------------------------------------------------------
Jupyter notebook file directory-> ../src/coms6200 project.ipynb<br />
!!!!!!!!------Ensure you set working directory in %/../src/ -------!!!!!<br />
1. Open coms6200 project.ipynb<br />
2. ensure the dataset exist in ../data/<br />
3. for any packet that is not import in anaconda, follow the below example command and run it at cell<br />
import sys<br />
!{sys.executable} -m pip install xxxx_packet<br />
example: if xgboost is not imported<br />
--------------------------------------------------------------------------<br />
import sys<br />
{sys.executable} -m pip install xgboost<br />
--------------------------------------------------------------------------<br />
1. coms6200 project.ipynb's cell are not independent, they are in sequence order, so maybe try run all instead random pick and run<br />
2. coms6200 project.ipynb is only for data preprocess and hyperparameter tuning, the good parameter and dataset will passed to data generator<br />
3. manually copy the hyperparameter and data preprocess code to the data generator file<br />
------------------------------------------------------------------------------------------------------------------------------------------------
Data generator
------------------------------------------------------------------------------------------------------------------------------------------------
1. Data generator file -> ../src/Model_evaluation_data_generator.py<br />
2. Ensure you install all the imported packege<br />
3. go to Model_evaluation_data_generator.py and Click run to generate the file<br />
4. Output file-> ../data/data.json<br />
5. manually copy data.json to result.json<br />
------------------------------------------------------------------------------------------------------------------------------------------------
Website visualize
------------------------------------------------------------------------------------------------------------------------------------------------
Main File: app.py -> ..\src\app.py<br />
Callback File: callbacks.py -> ..\callbacks\callbacks.py<br />
Data File: result.json -> ..\data\result.json<br />
How to execute the app:<br />
1. Before the initial execution of the app, making sure that you have imported the required packages (check package versions)<br />
2. Go to app.py and click run to execute the app<br />
3. If you see the console message that reads "Dash is running on http://127.0.0.1:8050/", it means the Dash app is successfully running<br />
4. Check the console message if the running message is not appearing<br />
5. Paste the Dash URL into your browser, and this should allow you to access the app content<br />
6. Follow the app's instruction and select the desired models/parameters<br />

------------------------------------------------------------------------------------------------------------------------------------------------
