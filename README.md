# What is GroceryDB?
The offering of grocery stores is a strong driver of consumer decisions, shaping their diet and long-term health. 
While processed food has been increasingly associated with unhealthy diet, information on the degree of processing characterising an item in a store is virtually impossible to obtain, limiting the ability of individuals to make informed choices. 
Here we introduce GroceryDB, a database with over 50,000 food items sold by Walmart, Target, and Wholefoods, unveiling the degree of processing characterizing each food. 
GroceryDB indicates that 73% of the US food supply is ultra-processed, and on average ultra-processed foods are 52% cheaper than minimally-processed alternatives. 
We find that the nutritional choices of the consumers, translated as the degree of food processing, strongly depend on the food categories and grocery stores. 
We show that there is no single nutrient or ingredient "bio-marker" for ultra-processed food, allowing us to quantify the individual contribution of over 1,000 ingredients to ultra-processing. 
GroceryDB and the associated http://TrueFood.Tech/ website make this information available, aiming to simultaneously empower the consumers to make healthy food choices, and aid policy makers to reform the food supply.

# Related Publications

- Machine Learning Prediction of Food Processing ([medRxiv, 2021](https://www.medrxiv.org/content/10.1101/2021.05.22.21257615))

- GroceryDB: Prevalence of Processed Food in Grocery Stores
 ([medRxiv, 2022](https://www.medrxiv.org/content/10.1101/2022.04.23.22274217))

# Data Files

- data/GroceryDB_foods.csv &rarr; This file includes all the foods in GroceryDB as well as their store, brand, FPro ([food processing score](https://www.medrxiv.org/content/10.1101/2021.05.22.21257615)), and nutrition facts normalized in 100 grams.

- data/GroceryDB_IgFPro &rarr; This file includes the IgFPro (ingredient food processing score) that estimates the contribution of over 1000 ingredients to food ultra-processing.

- data/GroceryDB_training_dataset_SRFNDSS_2001_2018_NOVA123_multi_compositions_12Nutrients.csv &rarr; This file includes the foods and their manual NOVA labels that we used to train [FoodProX](https://www.medrxiv.org/content/10.1101/2021.05.22.21257615) and obtain the FPro of products in grocery stores.

- data/GroceryDB_cereal_brands_manually_annotated.xlsx &rarr; This file contains the brands of cereals we used in the analysis.

- [USDA_FDC_BFPD_April2021_branded_food_classified_FPro_12NutPanel_min_10_nuts.csv](https://www.dropbox.com/s/1o99s1jgf66evls/USDA_FDC_BFPD_April2021_branded_food_classified_FPro_12NutPanel_min_10_nuts.csv?dl=0) &rarr; This file provides FPro (column 'AZ') for foods in USDA BFPD (Global Branded Food Products Database, version April 2021) database that have the minimum of 10 out of 12 mandated nutrients on nutrition fact labels.
- 
- [NHANES_2003_2018_FoodSource_Consumed.csv](https://www.dropbox.com/s/lae45qdgu8ifdk4/NHANES_2003_2018_FoodSource_Consumed.csv?dl=0) &rarr; This file provides the source of food consumed by NHANES participants, capturing the variables DR1FS and DR2FS that corresponds to \Where did
you get (this/most of the ingredients for this)?", found at (https://wwwn:cdc:gov/Nchs/
Nhanes/2017-2018/DR1IFF J:htm#DR1FS).

# Cite GroceryDB

If you find GroceryDB useful in your research, please add the following citation:

```
@misc{GroceryDB,
      title={GroceryDB: Prevalence of Processed Food in Grocery Stores}, 
      author={Babak Ravandi and Peter Mehler and Albert-László Barabási and Giulia Menichetti},
      year={2022},
      eprint={medRxiv:2022.04.23.22274217},
      url = {https://www.medrxiv.org/content/10.1101/2022.04.23.22274217}
}
```