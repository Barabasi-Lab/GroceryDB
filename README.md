# What is GroceryDB?

The offering of grocery stores is a strong driver of consumer decisions. While highly processed foods such as packaged products, processed meat and sweetened soft drinks have been increasingly associated with unhealthy diets, information on the degree of processing characterizing an item in a store is not straightforward to obtain, limiting the ability of individuals to make informed choices. GroceryDB, a database with over 50,000 food items sold by Walmart, Target and Whole Foods, shows the degree of processing of food items and potential alternatives in the surrounding food environment. The extensive data gathered on ingredient lists and nutrition facts enables a large-scale analysis of ingredient patterns and degrees of processing, categorized by store, food category and price range. Furthermore, it allows the quantification of the individual contribution of over 1,000 ingredients to ultra-processing. GroceryDB and the associated http://TrueFood.Tech/ website make this information accessible, guiding consumers toward less processed food choices.

# Related Publications

- Nutrient concentrations in food display universal behaviour ([Nature Food, 2022](https://www.nature.com/articles/s43016-022-00511-0))

- Machine learning prediction of the degree of food processing ([Nature Communications, 2023](https://www.nature.com/articles/s41467-023-37457-1))

- Prevalence of processed foods in major US grocery stores ([Nature Food, 2025](https://rdcu.be/d55mU))

# Data Files

- data/GroceryDB Source Data.xlsx &rarr; This file includes all the processed data used to generate the figures within the publication at Nature
  Food.

- data/GroceryDB_foods.csv &rarr; This file includes all the foods in GroceryDB as well as their store, brand, FPro ([food processing score](https://www.nature.com/articles/s41467-023-37457-1)), and nutrition facts normalized per 100 grams.

- data/UpdatedProductsIngredients_11_15.zip &rarr; This is a zipped json file of the disambiguated product ingredient trees used calculate tree features within the Nature
  Food publication. Schematic trees in the paper are generated from this file.

- data/GroceryDB_IgFPro.csv &rarr; This file includes the IgFPro (ingredient food processing score) that estimates the contribution of over 1,000 ingredients to food ultra-processing.

- data/GroceryDB_training_dataset_SRFNDSS_2001_2018_NOVA123_multi_compositions_12Nutrients.csv &rarr; This file includes the foods and their manual NOVA labels that we used to train [FoodProX](https://www.nature.com/articles/s41467-023-37457-1) and obtain the FPro of products in grocery stores.

- [USDA_FDC_BFPD_April2021_branded_food_classified_FPro_12NutPanel_min_10_nuts.csv](https://drive.google.com/file/d/1MD5LeSHtCe-Km6DCw39z1UTfGhlOTA6N/view?usp=drive_link) &rarr; This file provides FPro (column 'AZ') for foods in USDA BFPD (Global Branded Food Products Database, version April 2021) database that have the minimum of 10 out of 12 mandated nutrients on nutrition fact labels.
  
- [NHANES_2003_2018_FoodSource_Consumed.csv](https://drive.google.com/file/d/1MDJtbm5nnY2DPz_Q5ijhrusZ5lSlu9Dl/view?usp=drive_link) &rarr; This file provides the source of food consumed by NHANES participants, capturing the variables DR1FS and DR2FS that corresponds to \Where did
you get (this/most of the ingredients for this)?", found at [NHANES](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2017).

# GroceryDB on MongoDB
MongoDB is a NoSQL database that uses a JSON-like format for data storage. 
All the data scraped from Target, Walmart, and Whole Foods are available on our MongoDB server.
We provide both the cleaned data that is used for GroceryDB.

## Connecting to MongoDB
You will require two files to connect: query_builder.py and config.json, where the py file contains 
the functions necessary to establish a connection to MongoDB and the json file contains the necessary 
keys to successfully connect.

Install required python packages: pymongo, certifi, json, tqdm, and pandas.

After you have the necessary packages and files on your computer, run the jupyter notebook example 
to load the MongoDB data for downstream use.

## Datasets Available

- **CleanedData**: contains the FPro scores of all products from Target, Walmart, and Whole Foods. The FPro scores
  are calculated using a panel of nutrients from each product's provided nutrient table. For each nutrient
  the reported value and the convert value to g/100g are given. There are two FPro scores given, one with a 12
  nutrient panel and another with a 10 nutrient panel. Calories, price per calorie, and price per gram are found
  in this dataset.

- **ProductIngredients**: contains the ingredient list of each product from Target, Walmart, and Whole Foods in the
  format of an ingredient tree. Each ingredient reports its order in the ingredient list, parent order, depth, and
  distance to root node. Our disambiguation of ingredient names is given as well as the original name. Ingredients
  are identified as additive if they are considered additives by the USDA.

# Cite GroceryDB

If you find GroceryDB useful in your research, please add the following citation:

```
@misc{GroceryDB,
      title={Prevalence of processed foods in major US grocery stores}, 
      author={Babak Ravandi and Gordana Ispirova and Michael Sebek and Peter Mehler and Albert-László Barabási and Giulia Menichetti},
      journal={Nature Food}
      year={2025},
      dio={10.1038/s43016-024-01095-7},
      url = {https://www.nature.com/articles/s43016-024-01095-7}
}
```
