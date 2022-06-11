# coding=utf-8
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from collections import OrderedDict
from tqdm.notebook import tqdm, trange
import networkx as nx
import GraphHierarchy as gh
from scipy import stats

import gspread

import query_builder

nutrient_panels_grocery_store_data = {
    "12P": [
        'protein', 'totalFat', 'carbohydrates', 'sugar', 'fiber', 'calcium', 'iron',
        'sodium', 'vitaminC', 'cholesterol', 'saturatedFat', 'vitaminA'
    ],
    "11P": [
        'protein', 'totalFat', 'carbohydrates', 'sugar', 'fiber', 'calcium', 'iron',
        'sodium', 'vitaminC', 'cholesterol', 'saturatedFat'
    ],
    '10P': ['protein', 'totalFat', 'carbohydrates', 'sugar', 'fiber', 'calcium',
            'iron', 'sodium', 'cholesterol', 'saturatedFat'],
    #      ['Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', ' Fiber, total dietary', 'Calcium',
    #       'Iron', 'Sodium', 'Cholesterol', 'Fatty acids, total saturated']
}

# 'price_gdb'
cols_gdb_standard = ['original_ID', 'name', 'store', 'brand', '12P FPro', 'url',
                     'store category cleaned', 'UoM Grams Converted Value']

cols_gdb_nut_status = ['original_ID', 'isConverted', 'has10P', 'name', 'store', 'brand', '12P FPro',
                       'url', 'UoM Grams Converted Value']

stores_acronym_dict = {"WF": "WholeFoods", "WM": "Walmart", "TG": "Target"}
stores_acronym_reverse_dict = {"WholeFoods": "WF", "Walmart": "WM", "Target": "TG"}

bad_categories = [
    'non-food', 'multi-items', 'alcohol',
    # 'baby-food',
]

bad_categories_more_strict = bad_categories + [
    'exempt', 'no-category', 'maybe-bug', 'find-category'
]


def rgb_to_hex(rgb):
    rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    result = '#%02x%02x%02x' % tuple(rgb)
    # print(result)
    return result


colors_stores = {
    # OLD
    # 'Walmart': rgb_to_hex([56, 108, 176]),
    # 'Target': rgb_to_hex([240, 2, 127]),
    # 'WholeFoods': rgb_to_hex([191, 91, 23])
    #
    # 'Walmart_0': rgb_to_hex([252, 182, 79]),
    'Walmart': rgb_to_hex([253, 187, 48]),
    #
    'Target': rgb_to_hex([204, 0, 1]),
    'WholeFoods': rgb_to_hex([0, 111, 70])
}

NOVA_predictions_colors_dict = {
    'NOVA 1': np.array([0.4, 0.7607843137254902, 0.6470588235294118]) * 255,
    'NOVA 2': np.array([0.9882352941176471, 0.5529411764705883, 0.3843137254901961]) * 255,
    'NOVA 3': np.array([0.5529411764705883, 0.6274509803921569, 0.796078431372549]) * 255,
    'NOVA 4': np.array([0.9058823529411765, 0.5411764705882353, 0.7647058823529411]) * 255
}

stores = ['Walmart', 'Target', 'WholeFoods']


def normalizeBetweenTwoRanges(val, minVal, maxVal, newMin, newMax):
    return newMin + (
            ((val - minVal) * (newMax - newMin)) / (maxVal - minVal)
    )


from pandas import ExcelWriter


def save_xls(dfs_dict, xls_path, open=False, index=False):
    if type(dfs_dict) is not dict:
        dfs_dict = {'sheet1': dfs_dict}

    with ExcelWriter(xls_path) as writer:
        for df_name, df in dfs_dict.items():
            df.to_excel(writer, df_name, index=index)
        writer.save()

    try:
        if open is True:
            os.system('start EXCEL.EXE "{}"'.format(os.path.abspath(xls_path)))
    except:
        pass
    pass


def save_csv(dfs_dict, csv_path, open=False, index=False):
    dfs_dict.to_csv(csv_path, index=index)

    if open is True:
        os.system('start EXCEL.EXE "{}"'.format(os.path.abspath(csv_path)))
    pass


PROJECT_PATH = '../'


def get_google_sheet():
    # followed https://www.youtube.com/watch?v=T1vqS1NL89E
    gc = gspread.service_account(filename=PROJECT_PATH + '/google_service_credentials.json')

    g_sheets = gc.open_by_key('11f03waswPa_XLkrodA3Cee3H01rEW-KP3PO2HhSV7oo')

    return g_sheets


'''
Ingredient Network
'''


class aggregate_ingredient_trees():

    def __init__(self, gdb_list):

        # Just initialize the class level parameters (its better notation)
        self.limit_depth = None
        self.inherit_from_parent_at_depth = None
        self.stop_placing_order_at_depth = None

        # Please cache all data you need in 'gdb_list'
        self.gdb_list = gdb_list

    def generate_network(
            self,
            limit_depth,
            inherit_from_parent_at_depth,
            stop_placing_order_at_depth,
            use_depth_as_order
    ):
        """
        PLEASE ADD DESCRIPTION OF THE INPUT OUTPUT
        """

        self.limit_depth = limit_depth
        self.inherit_from_parent_at_depth = inherit_from_parent_at_depth
        self.stop_placing_order_at_depth = stop_placing_order_at_depth
        self.use_depth_as_order = bool(use_depth_as_order)

        count = 0

        main_list = []

        for doc in self.gdb_list:

            if len(doc['ingredient_tree']) == 0:
                continue
            else:
                temp = self.harmonize_node(
                    tree_nodes=doc['ingredient_tree'],
                    init='',
                    from_node_dict={
                        'ID': None,
                        'descriptors': [],
                        'type': None
                    },
                    current_depth=1
                )

            # if count == 4:
            #    break
            count += 1

            main_list.extend(temp)

        return main_list

    def harmonize_node(self, tree_nodes, init, from_node_dict, current_depth):
        """
        PLEASE ADD DESCRIPTION OF THE INPUT OUTPUT
        """
        main_list = []
        count = 1

        if tree_nodes == []:
            return None

        for to_node_dict in tree_nodes:
            # checking if we must stop placing order
            to_node_ID = None
            to_node_descriptors = to_node_dict['descriptors']
            to_node_type = to_node_dict['ingredient_type']

            if to_node_dict['ingredient_name'] == None:
                continue
            if self.stop_placing_order_at_depth == -1:
                if self.use_depth_as_order is True:
                    new_init = str(current_depth)
                    to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                    pass
                else:
                    # Checking if we must start inheriting exact order from parents
                    if self.inherit_from_parent_at_depth != -1:
                        if current_depth >= self.inherit_from_parent_at_depth:
                            new_init = init
                            to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                        else:
                            if init == '':
                                new_init = str(count)
                                to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                            else:
                                new_init = init + ',' + str(count)
                                to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                    else:
                        if init == '':
                            new_init = str(count)
                            to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                        else:
                            new_init = init + ',' + str(count)
                            to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                    pass
            else:
                # Check the depth at which we stop placing order
                if current_depth >= self.stop_placing_order_at_depth:
                    new_init = ''
                    to_node_ID = to_node_dict['ingredient_name']
                else:
                    # Checking if we must start inheriting exact order from parents
                    if self.inherit_from_parent_at_depth != -1:
                        if current_depth >= self.inherit_from_parent_at_depth:
                            new_init = init
                            to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                        else:
                            if init == '':
                                new_init = str(count)
                                to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                            else:
                                new_init = init + ',' + str(count)
                                to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                    else:
                        if init == '':
                            new_init = str(count)
                            to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']
                        else:
                            new_init = init + ',' + str(count)
                            to_node_ID = new_init + ': ' + to_node_dict['ingredient_name']

            # Designing dict
            temp_dict = {
                'from': from_node_dict['ID'],
                'from_type': from_node_dict['type'],
                'from_descriptors': from_node_dict['descriptors'],
                'to': to_node_ID,
                'to_type': to_node_type,
                'to_descriptors': to_node_descriptors
            }
            # appending to list
            main_list.append(temp_dict)

            next_from_node_dict = {
                'ID': to_node_ID,
                'descriptors': to_node_descriptors,
                'type': to_node_type
            }
            # checking if we must stop aggregating at certain depth
            if self.limit_depth != -1:
                if current_depth < self.limit_depth:
                    fetch_sub_connections = self.harmonize_node(
                        tree_nodes=to_node_dict['sub_ingredients'],
                        init=new_init,
                        from_node_dict=next_from_node_dict,
                        current_depth=current_depth + 1
                    )
                    if fetch_sub_connections != None:
                        main_list.extend(fetch_sub_connections)
            else:
                fetch_sub_connections = self.harmonize_node(
                    tree_nodes=to_node_dict['sub_ingredients'],
                    init=new_init,
                    from_node_dict=next_from_node_dict,
                    current_depth=current_depth + 1
                )
                if fetch_sub_connections != None:
                    main_list.extend(fetch_sub_connections)

            count += 1

        return main_list

    def fetch_products(self):

        DB = query_builder.get_mongo_databases()

        products = DB['GroceryDB']['ProductIngredients'].find({})

        p_list = [p for p in products]

        return p_list


'''
--------
'''


def extract_categories(store_df, column_category):
    store_df = store_df.set_index('original_ID')

    product_cats_dict = OrderedDict()
    products_with_cat_problem = []

    for original_id, category in store_df[column_category].items():
        try:
            product_cats = []
            for c in category:
                c = str(c).strip().lower()

                if c in ['365', 'target']:
                    continue
                    pass

                c = c.split('.')[-1]
                c = c.replace('&', '-').replace(',', '-').replace(' ', '-')

                while '--' in c:
                    c = c.replace('--', '-')

                product_cats.append(c)
                pass
            product_cats_dict[original_id] = product_cats
        except:
            products_with_cat_problem.append(original_id)
            pass
        pass

    print('Number of products with category issue:', len(products_with_cat_problem))
    product_cats_df = pd.DataFrame(product_cats_dict.values())
    product_cats_df = product_cats_df.fillna('')
    product_cats_df.columns = [str(c) for c in product_cats_df.columns]

    cats_df = product_cats_df.groupby(list(product_cats_df.columns)).agg(count=('0', np.size))
    print('Number of category branches:', len(cats_df))

    return cats_df, product_cats_dict


def ClassifyDB(db, model_per_fold, nut_sel, fill_missing_nuts_with_zero,
               convert_to_log):
    db = db.copy()
    # Xnut=db.loc[:, 'Protein': 'Total isoflavones']
    dbsel = db.loc[:, nut_sel]

    if fill_missing_nuts_with_zero:
        dbsel.fillna(0, inplace=True)

    if convert_to_log:
        dbsel = dbsel.apply(np.log).fillna(-20).replace([np.inf, -np.inf], -20)

    Xnut = dbsel.values

    indfold = 0
    for model in model_per_fold:
        indfold += 1
        y_pred = model.predict(Xnut)
        y_probs = model.predict_proba(Xnut)
        db['classf' + str(indfold)] = y_pred
        db['p1f' + str(indfold)] = y_probs[:, 0]
        db['p2f' + str(indfold)] = y_probs[:, 1]
        db['p3f' + str(indfold)] = y_probs[:, 2]
        db['p4f' + str(indfold)] = y_probs[:, 3]
        db['FProf' + str(indfold)] = (1 - db['p1f' + str(indfold)] + db['p4f' + str(indfold)]) / 2

    for p in range(1, 5):
        db['p' + str(p)] = db.loc[:,
                           ['p' + str(p) + 'f1', 'p' + str(p) + 'f2', 'p' + str(p) + 'f3', 'p' + str(p) + 'f4',
                            'p' + str(p) + 'f5']].mean(axis=1)
        db['std_p' + str(p)] = db.loc[:,
                               ['p' + str(p) + 'f1', 'p' + str(p) + 'f2', 'p' + str(p) + 'f3', 'p' + str(p) + 'f4',
                                'p' + str(p) + 'f5']].std(axis=1)

    db['FPro'] = db.loc[:, ['FProf1', 'FProf2', 'FProf3', 'FProf4', 'FProf5']].mean(axis=1)
    db['std_FPro'] = db.loc[:, ['FProf1', 'FProf2', 'FProf3', 'FProf4', 'FProf5']].std(axis=1)

    db['min_FPro'] = db.loc[:, ['FProf1', 'FProf2', 'FProf3', 'FProf4', 'FProf5']].min(axis=1)

    db['max_p'] = db.loc[:, ['p1', 'p2', 'p3', 'p4']].idxmax(axis=1)
    db['class'] = [int(s[1]) - 1 for s in db.loc[:, ['p1', 'p2', 'p3', 'p4']].idxmax(axis=1)]
    db['min_in_which_fold'] = db.loc[:, ['FProf1', 'FProf2', 'FProf3', 'FProf4', 'FProf5']].idxmin(axis=1)
    db['min_fold_id'] = [int(s[-1]) for s in
                         db.loc[:, ['FProf1', 'FProf2', 'FProf3', 'FProf4', 'FProf5']].idxmin(axis=1)]
    db['min_class'] = [db['classf' + str(db['min_fold_id'].iloc[n])].iloc[n] for n in range(db.shape[0])]

    for ind in range(1, 5):
        db['p' + str(ind) + '_minFPro'] = [db['p' + str(ind) + 'f' + str(db['min_fold_id'].iloc[n])].iloc[n] for n in
                                           range(db.shape[0])]
        pass
    return db


def get_grocery_data_nutrient_columns(store_df, count_calorie):
    nut_cols = store_df.loc[:, 'Protein':store_df.columns[-1]].columns
    nut_cols = [c for c in nut_cols if c.endswith(' Conv') is False]

    if count_calorie is False:
        for nut in nut_cols:
            if 'calori' in nut.lower():
                raise Exception(f'Calorie should not be in the list of nutrients {nut}')

    return nut_cols


def add_count_nutrients_for_grocery_store(store_df, nutrient_columns, filter_nutrients):
    number_nutrients = (
        (~store_df[nutrient_columns].isnull())
            .sum(axis=1)
    )

    if 'Number Nutrients' not in store_df.columns:
        store_df.insert(9
                        , 'Number Nutrients', number_nutrients)
    else:
        store_df['Number Nutrients'] = number_nutrients

    ##################
    number_nutrients_filtered = (
        (~store_df[set(nutrient_columns) - set(filter_nutrients)].isnull())
            .sum(axis=1)
    )

    if 'Number Nutrients Filtered' not in store_df.columns:
        store_df.insert(8, 'Number Nutrients Filtered', number_nutrients_filtered)
    else:
        store_df['Number Nutrients Filtered'] = number_nutrients_filtered

    return store_df


'''
------------------
'''


def parenthetic_contents(string):
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            yield (len(stack), string[start + 1: i], start + 1, i)
    pass


def order_ingredient_ignore_paranthesis(ingredient_list, search_for):
    T = ','.join([ing for ing in ingredient_list if ing is not None])

    t_tmp = T

    for level, txt_inside, start, end in parenthetic_contents(T):
        if level == 0:
            start = start - 1
            end = end + 1
            #         print(level, '->', txt_inside)
            t_tmp = t_tmp[0:start] + ''.join(['☼' for x in t_tmp[start:end]]) + t_tmp[end:]
        pass

    ingreds_main = [x.strip() for x in t_tmp.replace('☼', '').split(',')]

    for i, ingred in enumerate(ingreds_main):
        if search_for in ingred:
            return i + 1
        pass

    return 0


def order_ingredient(ingredient_list, search_for_list):
    T = ','.join([ing for ing in ingredient_list if ing is not None])

    t_tmp = T

    for level, txt_inside, start, end in parenthetic_contents(T):
        if level == 0:
            start = start
            end = end
            #         print(level, '->', txt_inside)
            t_tmp = t_tmp[0:start] + t_tmp[start:end].replace(',', '+') + t_tmp[end:]
        pass

    main_ingredients = [x.strip() for x in t_tmp.replace('☼', '').split(',')]

    for i, ingred in enumerate(main_ingredients):

        if sum([search_for in ingred for search_for in search_for_list]) == len(search_for_list):
            return i + 1
        pass

    return 0


# import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import TSNE
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score, \
    precision_recall_curve
# from scipy import interp
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import confusion_matrix
# import joblib
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


# from tqdm import tqdm
# import plotly.express as px
# from sklearn.decomposition import PCA
# import umap
# import matplotlib
# import operator


# AUC ROC Curve Scoring Function for Multi-class Classification
# "macro"
# "weighted"
# None


def multiclass_roc_auc_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_probs, average=average)


def multiclass_average_precision_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return average_precision_score(y_test, y_probs, average=average)


def multiclass_roc_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    fpr = dict()
    tpr = dict()
    for i in range(y_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])

    return (fpr, tpr)


def multiclass_average_precision_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    precision = dict()
    recall = dict()
    for i in range(y_probs.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_probs[:, i])

    return (precision, recall)


# returns performances and splits/models used in the cross-validation
def AUCAUPkfold(X, y, smoteflag, verbose=True):
    numfolds = 5
    numlabels = 4
    cv = StratifiedKFold(n_splits=numfolds, shuffle=True)
    Xs = np.copy(X)
    ys = np.copy(y)

    if smoteflag == True:
        smote = SMOTE()
        Xs, ys = smote.fit_resample(Xs, ys)

    performancesAUC = np.empty([numfolds, numlabels])
    performancesAUP = np.empty([numfolds, numlabels])
    splits = []
    model_per_fold = []
    index = 0
    for train, test in cv.split(Xs, ys):
        # print("%s %s" % (train, test))
        # clf = RandomForestClassifier(n_estimators = 200, max_features='sqrt', max_depth=420)
        clf = RandomForestClassifier(n_estimators=1800, max_features='sqrt', max_depth=260)
        # {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 420}

        splits.append([Xs[train, :], ys[train]])
        clf.fit(splits[index][0], splits[index][1])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test, :])
        y_probs = clf.predict_proba(Xs[test, :])
        performancesAUC[index, :] = np.array(multiclass_roc_auc_score(ys[test], y_probs, average=None))
        performancesAUP[index, :] = np.array(multiclass_average_precision_score(ys[test], y_probs, average=None))
        index += 1
        model_per_fold.append(clf)
        # if verbose==True:
        #    print(multiclass_roc_auc_score(ys[test], y_probs, average=None))

    if verbose == True:
        print("AUC: average over the folds")
        print(performancesAUC.mean(axis=0))
        print("AUC: std over the folds")
        print(performancesAUC.std(axis=0))

    if verbose == True:
        print("AUP: average over the folds")
        print(performancesAUP.mean(axis=0))
        print("AUP: std over the folds")
        print(performancesAUP.std(axis=0))

    return (performancesAUC, performancesAUP, splits, model_per_fold)


def plot_price_per_cal(
        col_FPro,
        col_yaxis,
        n_label_x_pad,
        figsize,
        dpi,
        store_compare_data_df,
        FPro_bin_size,
        print_n_on_plot,
        title,
        width_box,
        min_num_records_in_each_store_bin,
        verbose,
        min_FPro_filter=0,
        remove_legend=True
):
    #
    num_initial_products = len(store_compare_data_df)

    store_compare_data_df = store_compare_data_df.replace([np.inf, -np.inf], np.nan)
    store_compare_data_df = (
        store_compare_data_df[~(store_compare_data_df[col_yaxis].isnull())]
            .reset_index(drop=True)
    )

    store_compare_data_df = store_compare_data_df[store_compare_data_df[col_FPro] >= min_FPro_filter]

    col_bin = 'FPro bin'
    store_compare_data_df[col_bin] = -1

    bin_begin = 0
    for bin_end in np.arange(FPro_bin_size, 1.00 + FPro_bin_size, FPro_bin_size):
        if bin_end > 1:
            bin_end = 1.0

        mask_bin = (store_compare_data_df[col_FPro] >= bin_begin) & (store_compare_data_df[col_FPro] < bin_end)

        # print('bin_begin:', bin_begin, 'bin_end:', bin_end)
        store_compare_data_df.loc[mask_bin, col_bin] = round(bin_end, 4)

        bin_begin = bin_end
        pass

    # return store_compare_data_df
    filt_df = store_compare_data_df.groupby(['FPro bin', 'store']).agg(count=('original_ID', np.size)).reset_index()

    print('len BEFORE filtering num records in each bin/store:', len(store_compare_data_df))

    for filt_record in filt_df[filt_df['count'] < min_num_records_in_each_store_bin].to_dict('records'):
        store_compare_data_df = store_compare_data_df[
            ~(
                    (store_compare_data_df['FPro bin'] == filt_record['FPro bin']) &
                    (store_compare_data_df['store'] == filt_record['store'])
            )
        ]
        pass

    print('len AFTER filtering num records in each bin/store:', len(store_compare_data_df))

    corr, corr_pval = stats.spearmanr(store_compare_data_df[col_FPro], store_compare_data_df[col_yaxis])

    # SMALL_SIZE = font_size
    # MEDIUM_SIZE = font_size
    # BIGGER_SIZE = font_size

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    # sns.set_style("white")
    # sns.set_style("ticks")
    sns.set(style="ticks", font="Times New Roman", font_scale=1.2)

    for bin_end in np.arange(min_FPro_filter, 1.01, 0.1):
        if bin_end not in store_compare_data_df['FPro bin'].values:
            store_compare_data_df = store_compare_data_df.append(
                {'name': 'empty', 'store': None, 'FPro bin': round(bin_end, 2)},
                ignore_index=True)
            pass

    ax = sns.boxplot(x=col_bin, y=col_yaxis, data=store_compare_data_df,
                     hue="store", palette=colors_stores, hue_order=stores, width=width_box
                     )

    # plt.plot([1, 0.1], [6, 0.1], linewidth=2)
    # plt.scatter(6, 0.1, marker='o', s=10)
    # 1 --> 0.5
    # 2 --> 0.6
    # 3 --> 0.7
    # 4 --> 0.8
    # 5 --> 0.9
    # 6 --> 1.0

    nobs = store_compare_data_df.groupby([col_bin, 'store']).size().to_dict()
    box_min_vals = store_compare_data_df.groupby([col_bin, 'store']).agg({col_yaxis: np.min}).to_dict()[col_yaxis]
    # print(nobs)

    stores_in_data = store_compare_data_df['store'].unique()
    if print_n_on_plot:
        for i, label in enumerate(ax.get_xticklabels()):
            label = label.get_text()
            # ['WholeFoods', 'Walmart', 'Target'] ORDER MATERS A LOT!
            for j, store in enumerate(stores):
                if store not in stores_in_data:
                    continue
                # x_pad = 0
                # if j == 0:
                #     x_pad = n_label_x_pad[0]
                # elif j == 2:
                #     x_pad = n_label_x_pad[1]

                x_pad = n_label_x_pad[store]

                # print('xxx', (label, store))
                # print('xx', medians[(float(label), store)])

                nob_val = None
                if (float(label), store) in nobs:
                    nob_val = nobs[(float(label), store)]

                if nob_val is not None:
                    y_loc_txt = box_min_vals[(float(label), store)]

                    ax.text(
                        i + x_pad,
                        y_loc_txt - (y_loc_txt / 2.5),
                        "n:{}".format(nob_val),
                        horizontalalignment='center', size='x-small', color='black',
                        # weight='semibold'
                    )
            pass

    # ax.set_ylim([-0.1, 1.1])
    if True:
        y_min = store_compare_data_df[col_yaxis].min()
        y_max = store_compare_data_df[col_yaxis].fillna(-np.inf).replace(np.inf, -np.inf).max()

        ax.set_ylim([y_min - (y_min / 2), y_max + (y_max * 2)])

    # ax.set_xlim(0, 1)
    # ax.set_xticks(np.arange(0, 1.1, 0.1))
    # ax.set_xticks([0.8, 0.9, 1.0])

    # ax.set_xlim([-0.0, 1.0])
    ax.set_yscale('log')
    ax.set(ylabel='Price Per Calorie')

    if print_n_on_plot is False:
        ax.set(xlabel='')

    # plt.setp(ax.get_legend().get_texts(), fontsize='10')
    ax.legend(
        fontsize=10,
        # bbox_to_anchor=(0.85, 1.0)
    )

    if remove_legend:
        plt.legend([], [], frameon=False)

    title = '{} | corr: {} num product: {}/{}'.format(
        title,
        round(corr, 3),
        len(store_compare_data_df),
        num_initial_products
    )
    print(title)

    if verbose:
        plt.title(title)

    plt.tight_layout()

    return {
        'ax': ax,
        'store_compare_data_df': store_compare_data_df,
        'filt_df': filt_df
    }


def plot_all_categories_FPro(
        data_df, col_FPro, col_category, figsize,
        verbose, export_path, font_scale=1.2
):
    #     SMALL_SIZE = font_size
    #     MEDIUM_SIZE = font_size
    #     BIGGER_SIZE = font_size

    plt.figure(figsize=figsize, dpi=100)

    #     plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    #     plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    #     plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    #     plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    #     plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    #     plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    #     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    # sns.set_style("white")
    # sns.set_style("ticks")

    data_df = data_df.copy()

    data_df[col_category] = data_df[col_category].str.title()

    cat_stats = (
        data_df.groupby(col_category)
            .agg(
            FPro_median=(col_FPro, np.median),
            count_products=(col_FPro, np.size)
        )
            .sort_values(by='FPro_median')
            .reset_index()
    )

    if verbose:
        cat_stats['cat_name'] = cat_stats[col_category] + '|' + cat_stats['count_products'].astype(int).astype(str)
    else:
        cat_stats['cat_name'] = cat_stats[col_category]

    data_df = pd.merge(data_df, cat_stats, on=col_category)

    sns.set(style="ticks", font="Times New Roman", font_scale=font_scale)

    ax = sns.boxplot(x=col_FPro, y='cat_name', data=data_df, order=cat_stats['cat_name'])

    ax.set(ylabel='', xlabel='FPro')

    title = 'Num categories: {}'.format(len(cat_stats['cat_name']))
    print(title)

    if verbose:
        plt.title(title)

    del data_df

    plt.tight_layout(pad=0.5)
    plt.savefig(export_path, dpi=300)

    return ax


def plot_categories_FPro(
        col_FPro, col_categories, col_store, n_label_x_pad, figsize, dpi, data_df,
        order_xaxis, show_counts, remove_legend, xlabel, ylabel, verbose, export_path=None,
        legend_bbox_to_anchor=None, labels_xticks=None, labels_xticks_rotation=None
):
    if labels_xticks is not None:
        labels_xticks = [l for l in labels_xticks]
    #
    num_initial_products = len(data_df)

    data_df[col_categories] = data_df[col_categories].str.title()
    order_xaxis = [c.title() for c in order_xaxis]

    data_df = (
        data_df.replace([np.inf, -np.inf], np.nan)
        [~(data_df[col_FPro].isnull())]
            .reset_index(drop=True)
    )

    # SMALL_SIZE = font_size
    # MEDIUM_SIZE = font_size
    # BIGGER_SIZE = font_size

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    # sns.set_style("white")
    # sns.set_style("ticks")
    sns.set(style="ticks", font="Times New Roman", font_scale=1.05)

    ax = sns.boxplot(x=col_categories, y=col_FPro, hue=col_store, palette=colors_stores,
                     data=data_df, hue_order=stores, order=order_xaxis
                     )

    if labels_xticks is not None and len(labels_xticks) > 0:
        ax.set_xticklabels(labels_xticks, rotation=labels_xticks_rotation)

    nobs = data_df.groupby([col_categories, col_store]).size().to_dict()
    box_min_vals = data_df.groupby([col_categories, col_store]).agg({col_FPro: np.min}).to_dict()[col_FPro]
    # print(nobs)

    stores_in_data = data_df[col_store].unique()
    if show_counts:
        for i, label in enumerate(ax.get_xticklabels()):
            label = label.get_text()
            # ['WholeFoods', 'Walmart', 'Target'] ORDER MATERS A LOT!
            for j, store in enumerate(stores):
                if store not in stores_in_data:
                    continue

                x_pad = n_label_x_pad[store]

                nob_val = None
                if (label, store) in nobs:
                    nob_val = nobs[(label, store)]

                if nob_val is not None:
                    y_loc_txt = box_min_vals[(label, store)]

                    ax.text(
                        i + x_pad,
                        y_loc_txt - (y_loc_txt / 2.5),
                        "n:{}".format(nob_val),
                        horizontalalignment='center', size='x-small', color='black',
                        # weight='semibold'
                    )
            pass

    # y_min = data_df[col_FPro].min()
    # y_max = data_df[col_FPro].fillna(-np.inf).replace(np.inf, -np.inf).max()
    # ax.set_ylim([y_min - (y_min / 2), y_max + (y_max * 2)])
    # ax.set_yscale('log')

    ax.set_ylim([-0.1, 1.1])

    ax.set(xlabel=xlabel, ylabel=ylabel)

    ax.legend(
        fontsize=15,
        bbox_to_anchor=legend_bbox_to_anchor
    )

    if remove_legend:
        plt.legend([], [], frameon=False)

    title = 'Choices | num product: {}/{}'.format(
        len(data_df),
        num_initial_products
    )

    if verbose:
        plt.title(title)

    if export_path is not None:
        # plt.tight_layout()
        plt.savefig(export_path, bbox_inches='tight')

    plt.tight_layout()

    return fig


def plot_pie_category(cat_df, title, figsize, font_size=8):
    #     plt.rcParams.update({'font.size': 25})
    SMALL_SIZE = font_size
    MEDIUM_SIZE = font_size
    BIGGER_SIZE = font_size

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

    fig = plt.figure(figsize=figsize, dpi=300)

    counts = cat_df['count'].values

    #     plt.pie(Tasks,labels=my_labels,autopct='%1.1f%%')
    p, tx, autotexts = plt.pie(
        counts,
        labels=cat_df['label'].values,
        autopct='%1.1f%%'  # , colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    )

    for i, a in enumerate(autotexts):
        perc_text = "{}% ({})".format(
            round((counts[i] / sum(counts)) * 100, 1),
            counts[i])

        tx[i].set_text('{}: {}'.format(
            tx[i].get_text().title(),
            perc_text
        ))

        # a.set_text(perc_text)
        a.set_text('')
        pass

    #     plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    #     fig.set_size_inches(18.5, 10.5)

    # plt.savefig('Plots Food Processing/pie_{}.png'.format(title), bbox_inches="tight")
    plt.show()
    return plt


def plot_bar_classified(gdb_classification_status_df, figsize, font_size=8):
    #     plt.rcParams.update({'font.size': 25})
    SMALL_SIZE = font_size
    MEDIUM_SIZE = font_size
    BIGGER_SIZE = font_size

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

    # fig = plt.figure(figsize=figsize, dpi=300)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    width = 0.35

    tmp_prev_df = None

    for status in gdb_classification_status_df['status'].unique():

        bottom = None
        if tmp_prev_df is not None:
            bottom = tmp_prev_df['count']
            pass

        print(status, bottom)

        tmp_df = gdb_classification_status_df[gdb_classification_status_df['status'] == status]

        ax.bar(tmp_df['store'], tmp_df['count'], width, label=status, linewidth=0, bottom=bottom)

        tmp_prev_df = tmp_df

        pass

    del tmp_prev_df, tmp_df, bottom

    ax.legend(bbox_to_anchor=(0.08, 1.0))
    # ax.set_ylim([0, 27000])

    plt.show()
    return plt


def plot_bar_category_classified(
        gdb_classification_status_df,
        column_cat, bar_width, title,
        figsize, font_size=8
):
    '''
    PLOTTING STACKED BARS IGAVE ME SO MUCH TROUBLE...... DONT USE THIS FUNCTION!
    I used plotly instead!
    '''
    #     plt.rcParams.update({'font.size': 25})
    SMALL_SIZE = font_size
    MEDIUM_SIZE = font_size
    BIGGER_SIZE = font_size

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

    # fig = plt.figure(figsize=figsize, dpi=300)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    tmp_prev_df = None

    for status in gdb_classification_status_df['status'].unique():

        tmp_df = gdb_classification_status_df[gdb_classification_status_df['status'] == status]

        bottom = None
        if tmp_prev_df is not None:
            '''MUST DO THIS IN CASE a category has 0 number of missing nuts!'''
            tmp_prev_df = pd.merge(
                tmp_prev_df.rename(columns={'count': 'count_prev'}),
                tmp_df,
                on=column_cat, how='right'
            )

            # print(status)
            # print(tmp_prev_df['count_prev'])
            # print(tmp_df['count'])
            bottom = tmp_prev_df['count_prev']
            pass

        ax.barh(tmp_df[column_cat], tmp_df['count'], bar_width, label=status, linewidth=0, left=bottom)
        # ax.bar(tmp_df[column_cat], tmp_df['count'], width, label=status, linewidth=0, bottom=bottom)

        tmp_prev_df = tmp_df

        pass

    del tmp_prev_df, tmp_df, bottom

    ax.legend()
    # ax.legend(bbox_to_anchor=(0.08, 1.0))

    # ax.set_ylim([0, 27000])
    plt.suptitle(title)

    plt.show()
    return plt


def plotly_bar_category_classified(
        gdb_status_count_df, column_cat
):
    gdb_status_count_df = gdb_status_count_df[gdb_status_count_df['count'] > 100]

    gdb_status_count_order_df = gdb_status_count_df.groupby(['status', column_cat]).agg(
        count=('count', np.sum)).reset_index()
    gdb_status_count_order_df = (
        gdb_status_count_order_df[gdb_status_count_order_df['status'] == 'Missing Nutrients']
            .sort_values(by='count', ascending=False)
    )

    fig = px.bar(gdb_status_count_df, x="count", y=column_cat, color="status",
                 barmode=['relative', 'stack', "group"][0],
                 facet_row="store",
                 category_orders={column_cat: list(gdb_status_count_order_df[column_cat].values)},
                 width=1000, height=1200
                 )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        font=dict(
            family="Times New Roman",
            size=12,
            #         color="RebeccaPurple"
        )
        #     'xaxis': {'title': 'x-label',
        #                         'visible': True,
        #                         'showticklabels': True},
        #               yaxis={'title': 'y-label',
        #                         'visible': False,
        #                         'showticklabels': False}
    )

    return fig


def plot_hirearchy_measure(
        h_measures_df,
        fpro_bin_list,
        binwidth,
        func_names,
        col_fpro_bin,
        figsize,
        ylim_dict=None
):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 28

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

    fig, axes = plt.subplots(nrows=len(fpro_bin_list), ncols=2, constrained_layout=True,
                             sharex=False, sharey=True, dpi=100, figsize=figsize)
    axes = axes.flatten()

    fig.suptitle('{} | binwidth: {}'.format(
        ' & '.join(func_names).replace('_', ' ').title(), binwidth
    ))

    ax_index = 0
    for row, fpro_bin in tqdm(enumerate(fpro_bin_list)):

        h_levels_tmp = h_measures_df[h_measures_df[col_fpro_bin] == fpro_bin]

        g = sns.histplot(
            data=h_levels_tmp, x=func_names[0],
            # ax=axes[row, 0],
            ax=axes[ax_index],
            binwidth=binwidth
        )
        ax_index += 1

        g.set_title('{}'.format(fpro_bin), loc='center')

        g.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

        # write xlabel only for last plot in column
        if fpro_bin != fpro_bin_list[-1]:
            g.set(xlabel='')

        if False:
            g.set(ylim=(0, ylim_dict[binwidth])  # , xlim=(0, 1),
                  # xlabel=covars_latex_dicts[x_col], ylabel=covars_latex_dicts[y_col]
                  )
            pass
        #         g.xaxis.set_major_locator(plt.MultipleLocator(1))
        #         g.yaxis.set_major_locator(plt.MultipleLocator(100))
        #         print(ylim_dict[binwidth])
        '''-------------------------------'''
        g = sns.histplot(
            data=h_levels_tmp, x=func_names[1],
            # ax=axes[row, 1],
            ax=axes[ax_index],
            binwidth=binwidth
        )
        ax_index += 1

        g.set_title('{}'.format(fpro_bin), loc='center')

        g.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

        # write xlabel only for last plot in column
        if fpro_bin != fpro_bin_list[-1]:
            g.set(xlabel='')

        if False:
            g.set(ylim=(0, ylim_dict[binwidth])  # , xlim=(0, 1),
                  # xlabel=covars_latex_dicts[x_col], ylabel=covars_latex_dicts[y_col]
                  )
            pass
        #         g.xaxis.set_major_locator(plt.MultipleLocator(1))
        #         g.yaxis.set_major_locator(plt.MultipleLocator(100))

        pass
    pass


def plot_correlation_matrix(
        cat_corr_df,
        cols_heatmap,
        filter_min_num_items_in_category,
        figsize,
        title,
        font_size,
        add_count_items,
        vmin_vmax=[None, None],
        xticklabels='auto',
        fmt='.2g',
        export_path=None
):
    #
    cat_corr_df = cat_corr_df[cat_corr_df['count items'] > filter_min_num_items_in_category].reset_index()

    if add_count_items:
        cat_corr_df['cat count'] = cat_corr_df['cat'].str.title() + '|' + cat_corr_df['count items'].astype(str)
    else:
        cat_corr_df['cat count'] = cat_corr_df['cat'].str.title()

    matplotlib.rcParams['font.serif'] = "Times New Roman"
    matplotlib.rcParams['font.family'] = "serif"
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

    SMALL_SIZE = font_size
    MEDIUM_SIZE = font_size
    BIGGER_SIZE = font_size

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=figsize, dpi=150)

    g = sns.heatmap(
        cat_corr_df.set_index('cat count')[cols_heatmap],
        vmin=vmin_vmax[0],
        vmax=vmin_vmax[1],
        cmap=['BrBG', 'RdYlBu_r'][-1],
        xticklabels=xticklabels,
        annot=True,
        fmt=fmt
    )

    if title is not None:
        g.set_title(
            '{} | Num Categories (had over {} items): {}'.format(
                title,
                filter_min_num_items_in_category,
                len(cat_corr_df)),
            fontdict={'fontsize': font_size + 2},
            pad=12
        )
        pass
    g.set(ylabel='')

    if export_path is not None:
        # plt.tight_layout()
        plt.savefig(export_path, bbox_inches='tight')

    return {
        'g': g,
        'cat_corr_df': cat_corr_df
    }


def analyze_ingredient_tree(
        gdb_df,
        categorizing_strategy,
        use_weakly_connected_component,
        strategy_categories,
        limit_depth,
        inherit_from_parent_at_depth,
        stop_placing_order_at_depth,
        use_depth_as_order,
        col_fpro,
        col_novaclass,
        col_store
):
    ingred_fpro_dict = {}

    for strategy_cat in strategy_categories:  # np.arange(0.1, 1.001, 0.1)

        #     fpro_bin_end = np.round(fpro_bin_end, 4)
        #     fpro_bin = (fpro_bin_begin, fpro_bin_end)

        print('--------- strategy:', categorizing_strategy, '=', strategy_cat, '---------')

        if categorizing_strategy in ['custom-bins', 'all-in-one']:
            agg_ingreds = aggregate_ingredient_trees(
                gdb_list=gdb_df[
                    #             (gdb_df[col_fpro] >= fpro_bin_begin) & (gdb_df[col_fpro] < fpro_bin_end)
                    (gdb_df[col_fpro] >= strategy_cat[0]) & (gdb_df[col_fpro] < strategy_cat[1])
                    ].to_dict('records')
            )
        elif categorizing_strategy == 'novaclass':
            agg_ingreds = aggregate_ingredient_trees(
                gdb_list=gdb_df[gdb_df[col_novaclass] == strategy_cat].to_dict('records')
            )
        elif categorizing_strategy == 'store-compare-cereal':
            agg_ingreds = aggregate_ingredient_trees(
                gdb_list=gdb_df[gdb_df[col_store] == strategy_cat].to_dict('records')
            )
        else:
            raise Exception('Unknown Category: ' + categorizing_strategy)

        agg_net_df = pd.DataFrame(
            agg_ingreds.generate_network(
                limit_depth=limit_depth,
                inherit_from_parent_at_depth=inherit_from_parent_at_depth,
                stop_placing_order_at_depth=stop_placing_order_at_depth,
                use_depth_as_order=use_depth_as_order
            ))
        # stop placing order at depth

        agg_net_df = agg_net_df.fillna('ROOT')

        agg_net_df = (
            agg_net_df.groupby(['from', 'to'])
                .agg(freq=('from', np.size))
                .reset_index()
                .sort_values(by='freq', ascending=False)
        )

        '''Create network'''

        G = nx.from_pandas_edgelist(
            agg_net_df, source='from', target='to', edge_attr=True,
            create_using=nx.DiGraph()
        )
        G.remove_node('ROOT')

        if use_weakly_connected_component is True:
            print('[Aggregated Network] Num Nodes::', G.number_of_nodes(), 'Num Edges:', G.number_of_edges(), type(G))

            Gcc = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)
            # Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True) # bad not working!

            G = G.subgraph(Gcc[0])

            print('[Strongly Connected Comp Network] Num Nodes:', G.number_of_nodes(), 'Num Edges:',
                  G.number_of_edges(),
                  type(G))
            pass

        print('Num Nodes:{} Edges:{}'.format(
            G.number_of_nodes(),
            G.number_of_edges()
        ))

        if len(ingred_fpro_dict) > 0:
            prev_key = list(ingred_fpro_dict.keys())[-1]
            print('prev key', prev_key)
            G_prev = ingred_fpro_dict[prev_key]['G']

            print('Num new nodes compared to prev:', len(set(G.nodes()) - set(G_prev.nodes())))
            print('Num intersection of nodes compared to prev:',
                  len(set(G.nodes()).intersection(set(G_prev.nodes())))
                  )

            pass

        ingred_fpro_dict[strategy_cat] = {
            'aggregate_trees': agg_net_df,
            'G': G
        }

        #     fpro_bin_begin = fpro_bin_end
        #     break
        pass

    '''
    PART 2
    '''

    for strategy_cat, ingreds_dict in tqdm(ingred_fpro_dict.items()):
        print('-----------', strategy_cat, '--------------')

        G = ingreds_dict['G']

        print('Num Nodes:', G.number_of_nodes(), 'Edges:', G.number_of_edges())

        tmp_df = pd.concat([
            pd.Series(
                list(G.nodes),
                name='node'
            ),

            pd.Series(
                gh.hierarchical_levels(G, weight='freq'),
                name='hierarchical_levels'
            )
        ], axis=1
        )

        f_hierarchical_levels, f_influence_centrality, f_hierarchical_diff_adj_sparse, f_democracy_coefficient, f_hierarchical_incoherence = gh.forward_hierarchical_metrics(
            G, weight='freq')

        b_hierarchical_levels, b_influence_centrality, b_hierarchical_diff_adj_sparse, b_democracy_coefficient, b_hierarchical_incoherence = gh.backward_hierarchical_metrics(
            G, weight='freq')

        tmp_df['f_hierarchical_levels'] = f_hierarchical_levels
        tmp_df['b_hierarchical_levels'] = b_hierarchical_levels

        tmp_df['f_influence_centrality'] = f_influence_centrality
        tmp_df['b_influence_centrality'] = b_influence_centrality

        tmp_df['f_democracy_coefficient'] = f_democracy_coefficient
        tmp_df['b_democracy_coefficient'] = b_democracy_coefficient

        tmp_df['f_hierarchical_incoherence'] = f_hierarchical_incoherence
        tmp_df['b_hierarchical_incoherence'] = b_hierarchical_incoherence

        tmp_df = pd.merge(
            pd.DataFrame(G.out_degree(), columns=['node', 'out_degree']),
            tmp_df,
            on='node', how='right'
        )

        tmp_df = pd.merge(
            pd.DataFrame(G.in_degree(), columns=['node', 'in_degree']),
            tmp_df,
            on='node', how='right'
        )

        tmp_df = pd.merge(
            pd.DataFrame(G.degree(), columns=['node', 'degree']),
            tmp_df,
            on='node', how='right'
        )

        tmp_df.insert(1, 'strategy_cat', [strategy_cat for r in range(0, len(tmp_df))])

        #     tmp_df.insert(1, 'bin', [fpro_bin for r in range(0, len(tmp_df))])
        #     tmp_df.insert(1, 'bin_end', fpro_bin[1])

        tmp_df = tmp_df.sort_values(by='f_hierarchical_levels', ascending=True).reset_index(drop=True)

        ingreds_dict['h_df'] = tmp_df
        pass
    '''
    Part 3
    '''

    fpro_h_df = []
    for strategy_cat, ingreds_dict in ingred_fpro_dict.items():
        fpro_h_df.append(ingreds_dict['h_df'])
        pass

    fpro_h_df = pd.concat(fpro_h_df).reset_index(drop=True)

    return {
        'fpro_h_df': fpro_h_df,
        'ingred_fpro_dict': ingred_fpro_dict
    }


def load_data_ingre_tree_run_command():
    DB = query_builder.get_mongo_databases()

    p_ingreds = DB['GroceryDB']['ProductIngredients'].find({})

    p_ingreds_list = []
    for p in tqdm(p_ingreds):
        p_ingreds_list.append(p)

    p_ingreds_df = pd.DataFrame(p_ingreds_list)
    del p_ingreds, p_ingreds_list
    len(p_ingreds_df)

    '''
    '''

    products = DB['GroceryDB'][query_builder.mongoDB_CleanedData].aggregate([{
        "$project": {
            'original_ID': 1,
            'name': 1,
            'harmonized single category': 1,
            'isConverted': 1,
            'has10P': 1,
            '12P FPro': 1,
            '12P min_FPro': 1,
            '12P class': 1,
            'url': 1
        }
    }])

    products_list = []
    for p in tqdm(products):
        products_list.append(p)

    gdb_all_df = pd.DataFrame(products_list)
    del products, products_list
    len(gdb_all_df)

    '''
    '''

    gdb_df = gdb_all_df[(gdb_all_df['isConverted'] == 1) & (gdb_all_df['has10P'] == 1)].reset_index(drop=True)

    gdb_df = pd.merge(
        gdb_df,
        p_ingreds_df[['original_ID', 'ingredient_tree']],
        on='original_ID'
    )

    len(gdb_df)

    return gdb_df, gdb_all_df, p_ingreds_df


def run_hierarchy_commandline():
    gdb_df, gdb_all_df, p_ingreds_df = load_data_ingre_tree_run_command()
    ''''''

    categorizing_strategy = ['custom-bins', 'all-in-one', 'novaclass'][1]

    if categorizing_strategy == 'custom-bins':
        strategy_categories = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    elif categorizing_strategy == 'all-in-one':
        strategy_categories = [(0, 1.01)]
    elif categorizing_strategy == 'novaclass':
        strategy_categories = [0, 1, 2, 3]

    print(categorizing_strategy, '-->', strategy_categories)

    limit_depth = -1
    inherit_from_parent_at_depth = -1
    stop_placing_order_at_depth = 2

    res_ingred_analysis = analyze_ingredient_tree(
        gdb_df=gdb_df,
        categorizing_strategy=categorizing_strategy,
        strategy_categories=strategy_categories,
        limit_depth=limit_depth,
        inherit_from_parent_at_depth=inherit_from_parent_at_depth,
        stop_placing_order_at_depth=stop_placing_order_at_depth,
        col_fpro='12P FPro',
        col_novaclass='12P class'
    )

    fpro_h_df = res_ingred_analysis['fpro_h_df']
    ingred_fpro_dict = res_ingred_analysis['ingred_fpro_dict']

    path_export = 'output/fpro_hierarchy_LD_{}_IOP_{}_SOD_{}'.format(
        limit_depth,
        inherit_from_parent_at_depth,
        stop_placing_order_at_depth
    )

    save_csv(fpro_h_df, path_export + '.csv', open=False, index=False)

    ''''''

    max_row_count = 65535
    print(ingred_fpro_dict.keys())

    if True:
        save_xls(
            {str(fpro_bin): h_dict['h_df'][0:max_row_count] for fpro_bin, h_dict in ingred_fpro_dict.items()},
            path_export + '.xls',
            open=True, index=False)

    pass


def get_gsheet_category_df():
    g_sheets = get_google_sheet()
    sheet = g_sheets.worksheet('category_order_desc').get_all_records()
    category_order_desc_df = pd.DataFrame(sheet)
    category_order_desc_df = category_order_desc_df[category_order_desc_df['Category'] != '']

    category_order_desc_df['Category'] = category_order_desc_df['Category'].str.lower()

    mask = category_order_desc_df['Label'].str.strip() == ''
    category_order_desc_df.loc[mask, 'Label'] = category_order_desc_df.loc[mask, 'Category'].str.title()

    # label = cat.title()
    # if isinstance(cat_record['Label'], str) and cat_record['Label'] != '':
    #     label = cat_record['Label'].strip()
    #     pass

    return category_order_desc_df


if __name__ == '__main__':
    # run_hierarchy_commandline()
    price_per_cal_df = pd.read_csv('price_per_cal.csv')

    data_df = price_per_cal_df[price_per_cal_df['harmonized single category'] == 'soup-stew'].reset_index(drop=True)

    ax = sns.regplot(x='12P FPro', y='price percal', data=data_df, scatter=False,
                     robust=True,  # x_estimator=np.mean,
                     color='red', ci=None)
    plt.show()
    print(ax)
    pass
