# What is GroceryDB?
The offering of grocery stores is a strong driver of consumer decisions, shaping their diet and long-term health. 
While processed food has been increasingly associated with unhealthy diet, information on the degree of processing characterising an item in a store is virtually impossible to obtain, limiting the ability of individuals to make informed choices. 
Here we introduce GroceryDB, a database with over 50,000 food items sold by Walmart, Target, and Wholefoods, unveiling the degree of processing characterizing each food. 
GroceryDB indicates that 73% of the US food supply is ultra-processed, and on average ultra-processed foods are 52% cheaper than minimally-processed alternatives. 
We find that the nutritional choices of the consumers, translated as the degree of food processing, strongly depend on the food categories and grocery stores. 
We show that there is no single nutrient or ingredient "bio-marker" for ultra-processed food, allowing us to quantify the individual contribution of over 1,000 ingredients to ultra-processing. 
GroceryDB and the associated http://TrueFood.Tech/ website make this information available, aiming to simultaneously empower the consumers to make healthy food choices, and aid policy makers to reform the food supply.

# Data Files

- data/GroceryDB_foods.csv &rarr; This file includes all the foods in GroceryDB as well as their store, brand, FPro ([food processing score](https://www.medrxiv.org/content/10.1101/2021.05.22.21257615)), and nutrition facts normalized in 100 grams.

- data/GroceryDB_IgFPro &rarr; This file includes the IgFPro (ingredient food processing score) that estimates the contribution of over 1000 ingredient to food ultra-processing.

- data/GroceryDB_training_dataset_SRFNDSS_2001_2018_NOVA123_multi_compositions_12Nutrients.csv &rarr; This file includes the foods and their manual NOVA labels that we used to train [FoodProX](https://www.medrxiv.org/content/10.1101/2021.05.22.21257615) to obtain FPro of products in grocery stores.