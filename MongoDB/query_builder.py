import json
from pymongo import MongoClient
from datetime import datetime
import certifi

mongoDB_CleanedData = 'CleanedData_temp'

#Removing categories that do not work in our GroceryDB analysis
bad_categories = ['non-food', 'multi-items', 'alcohol']

bad_categories_more_strict = bad_categories + [
    'exempt', 'no-category', 'maybe-bug', 'find-category']

with open('config.json', 'r') as f:
    config_local = json.loads(f.read())
    pass

stores = ['Target', 'WholeFoods', 'Walmart']
stores_dict = {'Target': 'TG', 'WholeFoods': 'WF', 'Walmart': 'WM'}
stores_short_to_long_name_dict = {'TG': 'Target', 'WF': 'WholeFoods', 'WM': 'Walmart'}


def get_mongo_databases(verbose=False):
    clients = {}
    databases = {}

    clients['products'] = MongoClient(config_local['connections_strings']['GDB-Products-C1'], tlsCAFile=certifi.where())

    # Send a ping to confirm a successful connection
    clients['products'].admin.command('ping')
    print("Ping Successful! You are connected to MongoDB!")

    if verbose:
        print('GroceryDB Permissions:')
        status=clients['products'].admin.command("connectionStatus")
        print(status['authInfo']['authenticatedUserRoles'][0]['role'])
        print(status['authInfo']['authenticatedUserRoles'][1]['role'])

    databases['GroceryDB'] = clients['products']['GroceryDB']

    '''--------------------------------'''
    # clients['images'] = MongoClient(config_local['connections_strings']['GDB-Images-C1'], tlsCAFile=certifi.where())

    # if verbose:
    #     print('---------------\nImage Cluster DB:')
    #     for db in clients['images'].list_databases():
    #         print(db)
    #     pass
    #
    # databases['Product_Images'] = clients['images']['Product_Images']

    return databases


def update_scraped_navigation_bar(store, original_ID, navigation_bar):
    if navigation_bar is None or len(navigation_bar) == 0:
        return

    DB = get_mongo_databases()

    nav_bar_dict = {}

    nav_bar_dict['nav_bar'] = navigation_bar
    nav_bar_dict['Download_Time_gdb'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    query = DB['GroceryDB'][store].update(
        {
            "Product_Original_ID": original_ID
        },
        {
            "$push": {"Navigation_bar": nav_bar_dict}
        }
    )

    if query['nModified'] != 1:
        print('[WARNING] Navigation bar did not get updated in DB for', original_ID)

    pass


def get_latest_scraped_product_detail(store):
    DB = get_mongo_databases()

    '''ToDo no need to have projection as a dictionary!'''

    projection = {
        'Target': {
            "last_product_detail.item": 1,
            "last_product_detail.price": 1,
            'last_product_detail.Download_Time_gdb': 1,
            'last_product_detail.Fetch_URL_gdb': 1,
            'last_product_detail.status_gdb': 1,
            'last_product_detail.Status': 1
        },
        'WholeFoods': {
            'last_product_detail': 1
        },
        'Walmart': {
            'last_product_detail': 1
        }
    }

    projection[store].update({
        'Navigation_bar': 1,
        'Product_Original_ID': 1,
        'Product_URL': 1
    })

    # projection = {
    #     "last_product_detail.item": 1,
    #     "last_product_detail.price": 1,
    #     'last_product_detail.Download_Time_gdb': 1,
    #     'last_product_detail.Fetch_URL_gdb': 1,
    #     'last_product_detail.status_gdb': 1,
    #     'last_product_detail.Status': 1,
    #     'Navigation_bar': 1,
    #     'Product_Original_ID': 1,
    #     'Product_URL': 1
    # }

    query = DB['GroceryDB'][store].aggregate([
        {
            "$addFields": {
                "last_product_detail": {
                    "$slice": ["$Product_Detail", -1],
                },
            },
        },
        {
            "$project": projection[store]
        },
    ])

    return query


if __name__ == "__main__":
    get_latest_scraped_product_detail('Target')
    pass