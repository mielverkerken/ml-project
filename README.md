# ml-project
## Notebooks

## Code structure
This repository consits of three main folders: the data, util functions and the actual libraries which includes our code base. 
In util, three important files that are used often are 'constants.py', which includes all pre-defined constants, as well as a list of all features. Next there is 'helpers.py' where we added two functions to create scorer objects used to evaluate the ML algorithms using top3k and map3k metrics. The other file 'visanim.py' is an extension on 'vis.py' to allow visualisation of a sample instead of a single frame.
In libraries, we subdivided the functions in the different steps in the machine learning training process.

- 'analyse.py' includes functions to plot confusion matrices, learning curves and 2D parameter plots.
- 'augmentation.py' includes functions to augment the data
- 'base.py' includes general and multi-purpose functions (e.g distance between two points)
- 'data_split.py' includes our custom StratifiedGroupKFold implementation
- 'feature_extraction.py' includes all functions to extract the features out of the preprocessed data, and create a feature matrix.
- 'model_training.py' includes our pipeline implementations. 
- 'preprocessing.py' includes all the functions to preprocess the data