# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:5000/

### About:
1. This project is part of [Udacity's Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

2. This project divided into two parts:

    - Learning the process of how to Extract, transform and load data (ETL).

    - Learning how to use Machine Learning Pipeline.

3. Data was provided by [Figure Eight](https://www.figure-eight.com/).