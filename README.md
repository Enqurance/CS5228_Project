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

- `data/*`

    Original training and test data, preprocessed training and test data and prediction results from different models.
    
    The final train file after preprocessed is the `train_preprocessed_impute.csv`, optimized from the `train_preprocessed.csv` by applying IterativeImputer of sklearn.impute.
    
    The final text file after preprocessed is the `test_preprocessed.csv`.
    
    The different /*result.csv are prediction results from different models. The final best kaggle scores are from the `xgb_result_best.csv`.


- `EDA/EDA.ipynb`

    Jupiter notebook containing the steps for EDA.


- `images/*`

    Images used by the jupiter notebooks in this repository or the final report.


- `util/DataMining.py`

    Util functions used by the DataMining.ipynb jupiter notebook.


- `util/DataPreprocess.py`

    Util functions used by the DataPreprocess.ipynb jupiter notebook.


- `DataMining.ipynb`

    Jupiter notebook containing the data mining and prediction steps for XGB and XGB by make models.


- `DataPreprocess.ipynb`

    Jupiter notebook containing the preprocess steps for the training and the Kaggle test dataset.


- `FineTuning.ipynb`

    Jupiter notebook containing the hyperparameter fine_tuning steps of partial models, and some exploratory work for feature selections.
