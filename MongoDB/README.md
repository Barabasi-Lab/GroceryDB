# GroceryDB on MongoDB
MongoDB is a NoSQL database that uses a JSON-like format for data storage. 
All the data scraped from Target, Walmart, and Whole Foods are available on our MongoDB server.
We provide both the raw data and the cleaned data that is used for GroceryDB.

## Connecting to MongoDB
You will require two files to connect: query_builder.py and config.json, where the py file contains 
the functions necessary to establish a connection to MongoDB and the json file contains the necessary 
keys to successfully connect.

query_builder.py requires python packages: json, pymongo, and certifi.

After you have the necessary packages and files on your computer, run the jupyter notebook example 
to load the MongoDB data for downstream use.
