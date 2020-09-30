# Disaster Response Pipeline Project

### Overview
This projects uses message data to improve the organization of help for people affected by real natural disasters.
A classifier is implemented to predict categories of each message to forward them to the responsible organizations.


### Files
* app: contains files for the web app
* data: contains data files as well as a python script for data preparation
* models: contains a python script for model training as well as the pickled classifier model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
