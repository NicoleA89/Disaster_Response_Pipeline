# Disaster Response Pipeline

## Libraries
- nltk
- numpy
- pandas
- scikit-learn
- sqlalchemy

## Content
In this project various disaster messages were investigated. A model for an API which classifies disaster messages. A model was built with which messages can be classified that were sent following a catastrophic event. The project includes a web app that can be filled with new messages that are then being classified to certain categories (e.g. weather related, first aid, etc.). The web app also displays visualizations of the data:

![Messages_by genre](/images/Messages_Genre.png)

![Messages_by category](/images/Messages_category.png)

## Files
These are thee main folders:

### 1. Data
- disaster_categories.csv (dataset with all categories)
- disaster_messages.csv (dataset with all messages)
- process_data.py (ETL pipeline to extract, clean, merge and save data into a database)
- DisasterResponse.db (result from ETL pipeline)

### 2. Models
- train_classifier.py (machine learning pipeline to train and export classifier)
- classifier.pkl (result from machine learning pipeline)

### 3. App
- run.py (flask file to run web application)

## Acknowledgements
I thank Figure Eight for providing the data and Udacity for the advice and review.

## Instructions
Run the following commands in the project's root directory to set up the database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/
