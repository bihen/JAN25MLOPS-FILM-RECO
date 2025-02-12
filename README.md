Movie Recommender System based on 20m Movielens Dataset
==============================

This project is for the DataScientest course.
We are building a Movie Recommendation System based on the 20m movielens dataset.(https://grouplens.org/datasets/movielens/20m/)

For this we try different approaches to a recommendation system. These are yet to be clearly determined.

Data Preperation
==============================
As part of data preperation, a new .csv file is created, movie_genome_df.csv. This file is required for models using genome data for prediction. It can be manually created using the Keras-HybridFilteringWithGenomes.ipynb notebook.
Other than that all data is prepared locally in each notebook.

Models
==============================
We have trained different models on the 20M Dataset. 
These include classic machine learning models, specificially using the Surprise python library (https://surpriselib.com/), but also Deep Learning Models using Keras.

We will try to include all trained models in the github release, however due to filesize (Models range from 500MB to 8GB of filesize), there may be limitations.

Folder Structure:
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
