# GroceryDB on MongoDB
MongoDB is a NoSQL database that uses a JSON-like format for data storage. 
All the data scraped from Target, Walmart, and Whole Foods are available on our MongoDB server.
We provide both the raw data and the cleaned data that is used for GroceryDB.

## Connecting to MongoDB
You will require two files to connect: query_builder.py and config.json, where the py file contains 
the functions necessary to establish a connection to MongoDB and the json file contains the necessary 
keys to successfully connect.

Install required python packages: pymongo, certifi, json, tqdm, and pandas.

After you have the necessary packages and files on your computer, run the jupyter notebook example 
to load the MongoDB data for downstream use.

## Datasets Available

- CleanedData: contains the FPro scores of all products from Target, Walmart, and Whole Foods. The FPro scores
  are calculated using a panel of nutrients from each product's provided nutrient table. For each nutrient
  the reported value and the convert value to g/100g are given. There are two FPro scores given, one with a 12
  nutrient panel and another with a 10 nutrient panel. Calories, price per calorie, and price per gram are found
  in this dataset.

- ProductIngredients: contains the ingredient list of each product from Target, Walmart, and Whole Foods in the
  format of an ingredient tree. Each ingredient reports its order in the ingredient list, parent order, depth, and
  distance to root node. Our disambiguation of ingredient names is given as well as the original name. Ingredients
  are identified as additive if they are considered additives by the USDA.
