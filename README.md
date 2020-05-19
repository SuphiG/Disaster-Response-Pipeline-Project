# Disaster Response Pipeline Project

 In this project, I analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. I find a data set containing real messages that were sent during disaster events

![Screenshot 1](https://user-images.githubusercontent.com/58215303/82330889-114fde80-99ec-11ea-80a4-7f5c00b39cb6.PNG)

![Screenshot 2](https://user-images.githubusercontent.com/58215303/82331021-3e9c8c80-99ec-11ea-9810-c8a98a8243b1.PNG)



### File Descriptions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
