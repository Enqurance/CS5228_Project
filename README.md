## Quick Start

You can setup the environment for running this repo quickly with conda. 
1. Create the environment.

```
conda env create -f environment.yml
```

2. Activate the environment.

```
conda activate cs5228_project
```

## How to Run

1. Run `EDA/EDA.ipynb`
2. Run `DataPreprocess.ipynb`
3. (Optional) Run `FineTuning.ipynb`: run the notebook if you want to check out the fine tuning steps.
3. Run `DataMining.ipynb`
   
## File Structure

`data/*`

Original training and test data, preprocessed training and test data and prediction results from different models.

`EDA/EDA.ipynb`

Jupiter notebook containing the steps for EDA.

`images/*`

Images used by the jupiter notebooks in this repositry or the final report.

`util/DataMining.py`

Util functions used by the DataMining.ipynb jupiter notebook.

`util/DataPreprocess.py`

Util functions used by the DataPreprocess.ipynb jupiter notebook.

`DataMining.ipynb`

Jupiter notebook containing the data mining and prediction steps for XGB and XGB by make models.

`DataPreprocess.ipynb`

Jupiter notebook containing the preprocess steps for the training and the Kaggle test dataset.

`FineTuning.ipynb`

Jupiter notebook containing the hyperparameter finetuning steps, and some exploratory work for feature selections.
