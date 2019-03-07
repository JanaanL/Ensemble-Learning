This is a machine learning library developed by Janaan Lake for CS5350 at the University of Utah.

The ensemble_learning.py library contains AdaBoost, Bagging and Random Tree algorithms.  Currently, they are used on the bank dataset found at https://archive.ics.uci.edu/ml/datasets/Bank+Marketing.

To run these algorithms, run the following script on the command line: 
python3 ensemble_learning.py

Alternatively, run the run.sh shell script.  

The following methods in ensemble_learning.py will run the algorithms listed above:

test_adaboost()
test_bagging()
test_random_forest()

Also, the bias and variance will be calculated on the bagging ensemble created from the bank dataset and compared to the same data run on single decision trees.  This is done with the calculate_bv_bagged() method.  The same calculation will be done on the random forest ensemble with the calculate_bv_random_forest() method.

All of these methods run for a default of 1,000 iterations unless a different parameter is passed in the function call.  Currently, they are set for 10 iterations so you can quickly see the results. 
