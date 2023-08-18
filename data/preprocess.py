from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
from utils.utils import get_feilds_attributes, get_Discretizer_attributes
from utils.utils import *

def POOL_preprocess(df, N_BINS = 100):
    '''
    Preprocess the DataFrame 
    Args:
        df: DataFrame
        N_BINS: number of bins for each numerical column (will not be the exact number of bins, differ by distribution)
    Return:
        X_trans: DataFrame after preprocessing
        ct: ColumnTransformer object, for inference and inverse transform
        NUM_vs_CAT: tuple, (number of numerical columns, number of categorical columns - 1) "in feature field, do not include label column"
        existing_values: dict, {column name: sorted list of existing values}
    '''
    # df.drop(columns = ['workclass','education','marital-status','occupation','relationship','race','gender','native-country'], inplace = True, errors = 'ignore')
    NUM, CAT, TARGET = get_feilds_attributes()
    quantile, uniform = get_Discretizer_attributes()
    
    num_CAT = len(CAT)
    num_NUM = len(NUM)  
    
        # feature num of each type
    if TARGET in NUM:
        if get_task() != 'regression':
            raise ValueError('TARGET is numerical, but task is not regression')
        NUM_vs_CAT = (num_NUM - 1, num_CAT)
    elif TARGET in CAT:
        if get_task() != 'classification':
            raise ValueError('TARGET is categorical, but task is not classification')
        NUM_vs_CAT = (num_NUM, num_CAT - 1)
    else:
        raise ValueError('TARGET is not in NUM or CAT')
    
    pipe_uniform = KBinsDiscretizer(n_bins = N_BINS, encode='ordinal', strategy='uniform', subsample=None)
    pipe_quantile = KBinsDiscretizer(n_bins = N_BINS, encode='ordinal', strategy='quantile', subsample=None)
    
    ColumnTransformers_list = [('pass', 'passthrough', make_column_selector(dtype_include=object))]
    # print(uniform, quantile)
    # print(not isinstance(uniform, list), not isinstance(quantile, list))
    if not isinstance(uniform, list) and not isinstance(quantile, list):
        raise ValueError('quantile and uniform can not be empty at the same time')
    if isinstance(uniform, list):
        for column in uniform:
            ColumnTransformers_list.append((column, pipe_uniform, [column]))
    if isinstance(quantile, list):
        for column in quantile:
            ColumnTransformers_list.append((column, pipe_quantile, [column]))
    
    ct = ColumnTransformer(
        ColumnTransformers_list
         ,remainder = 'passthrough', verbose_feature_names_out = False) # make sure columns are unique
    
    ct.set_output(transform = 'pandas')
    X_trans = ct.fit_transform(df)
    X_trans = reorder_dataframe(X_trans)
    # print(X_trans[NUM])
    C_pool = np.array([])

    for column in NUM :
        if column == TARGET:# if the column is the label column, skip it(do not add it to C_pool)
            continue
        values = X_trans[column].to_numpy().reshape(-1,1)
        # values = ct.named_transformers_[column].inverse_transform(values)
        values = (np.unique(values).flatten())
        # values = np.arange(len(values))
        # values = values / max(values)
        # min max scaling
        # values = (values - values.min()) / (values.max() - values.min())
        C_pool = np.concatenate((C_pool, values, np.array([-1])))
        # print(values)
    catagory_count = 0
    for column in CAT:
        if column == TARGET: # if the column is the label column, skip it(do not add it to C_pool)
            continue
        catagory_count += len(X_trans[column].unique()) + 1
    C_pool = np.concatenate((C_pool, np.arange(catagory_count)))
    print('train',C_pool)
    # store the numrical columns' existing values for identifying unseen values
    existing_values = {}
    for column in NUM:
        existing_values[column] = sorted(X_trans[column].unique().astype(int))
    for column in CAT:
        existing_values[column] = sorted(X_trans[column].unique().astype(str))
    
    # apply Ordinal encoding on columns
    from sklearn.preprocessing import OrdinalEncoder
    OE_list = {}
    for column in NUM + CAT:
        OE = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
        X_trans[column] = OE.fit_transform(X_trans[[column]])
        OE_list[column] = OE
    
    # make all columns' catagory unique
    # 7/19: each NUM column has its own number of unique values, plus 1 for unseen values
    # each column has it's own number of unique values. '+1' is for unseen values
    offset = 0
    for column in NUM + CAT:
        # if column == TARGET: # if the column is the label column, skip it(do not add it to C_pool)
        #     continue
        X_trans[column] = X_trans[column].apply(lambda x: x + offset)
        offset += (X_trans[column].max() - X_trans[column].min() + 1) + 1
    
    X_trans = X_trans.astype(int).reset_index(drop = True)
    # print(X_trans)
    # print(check_DataFrame_distribution(X_trans))
    return X_trans, (ct, OE_list, NUM, CAT, existing_values), NUM_vs_CAT, C_pool
    # -1 is for the label column 

def POOL_preprocess_inference(df: pd.DataFrame,
                              inference_package: tuple,
                                # ct: ColumnTransformer,
                                # OE_list: dict,
                                # NUM: list,
                                # CAT: list,
                                # existing_values: dict,
                              ):
    '''Preprocess the DataFrame when inference
    
    Args:
        `df`: DataFrame to be processed.\n
        `inference_package`: tuple, containing the following objects.
            `ct`: ColumnTransformer object required for inference, which makes sure values are in the same range as training data
            `OE_list`: dict, {column name: OrdinalEncoder object}\n
            `NUM`: list of numerical columns \n
            `CAT`: list of categorical columns\n
            `existing_values`: dict, {column name: sorted list of existing values}
    '''
    (ct, OE_list, NUM, CAT, existing_values) = inference_package
    X_trans_ori = ct.transform(df)
    X_trans_ori = reorder_dataframe(X_trans_ori)
    
    import copy
    TEST_POOL_VALUES = copy.copy(X_trans_ori)
    # print(TEST_POOL_VALUES)
    
    # caculate the loaction of unseen values
    unseen_node_indexs = {}
    offset = 0
    for col in NUM + CAT:
        if col == get_label_colunm():
            continue
        unseen_node_indexs[col] = (int(len(existing_values[col])) + offset )
        offset += int(len(existing_values[col])) + 1
    
    X_trans = X_trans_ori
    
    # apply Ordinal encoding on columns, and make all columns' catagory unique
    offset = 0
    for column in NUM + CAT:
        if column == get_label_colunm(): # if the column is the label column, skip it(do not add it to C_pool)
            continue
        OE = OE_list[column]
        X_trans[column] = OE.transform(X_trans[[column]]) # use fitted OE to transform, the unseen values will be encoded as -1
        if -1 in X_trans[column].tolist():
            print('[preprocess]: detected unseen values in column', column)
        X_trans[column] = X_trans[column].apply(lambda x: x + offset if x != -1 else unseen_node_indexs[column])
        offset = unseen_node_indexs[column] + 1  
    # produce OE on TARGET column last
    for column in [get_target()]:
        OE = OE_list[column]
        X_trans[column] = OE.transform(X_trans[[column]]) # use fitted OE to transform, the unseen values will be encoded as -1
        if -1 in X_trans[column].tolist():
            print('[preprocess]: detected unseen values in column', column)
        X_trans[column] = X_trans[column].apply(lambda x: x + offset if x != -1 else unseen_node_indexs[column])
    
    X_trans = X_trans.astype(int).reset_index(drop = True) 
    # check_DataFrame_distribution(X_trans)
    return X_trans, unseen_node_indexs, TEST_POOL_VALUES
