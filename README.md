# udacity-data-engineering
Udacity BNT Data Scientist Path Data Engineering Course

This project requires python 3.11.

# Disaster Response Pipeline Project

This project is about classifying the text messages during a disaster scenario. The messages are classified
into multi-classes depending on the content of the message. Please follow the steps below to create and store
the cleaned dataset, train the model and use it in a local web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
