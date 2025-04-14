import numpy as np
import pandas as pd
import structureboost as stb

def test_shap_simple_reg_conc():
    # Load the data
    df = pd.read_csv('tests/data_for_tests/concrete_simplified.csv')
    df.columns = ['cement', 'blast_furnace_slag', 'fly_ash', 
                       'water', 'superplasticizer', 'coarse_agg', 
                       'fine_agg', 'age_days','strength']
    np.random.seed(999)
    foldnum = np.random.randint(0,10,df.shape[0])

    df_train = df.loc[foldnum<=7,:]
    df_test = df.loc[foldnum>7,:]
    features = df.columns[:-1].tolist()
    target = 'strength'
    X_train = df_train.loc[:,features]
    y_train = df_train[target].to_numpy()
    X_test = df_test.loc[:,features]
    y_test = df_test[target].to_numpy()

    fc = stb.get_basic_config(X_train, stb.default_config_dict())
    stb1 = stb.StructureBoost(mode='regression', feature_configs=fc, num_trees=100,
                             learning_rate=.05, max_depth=3)
    stb1.fit(X_train, y_train)
    sv = np.round(stb1.predict_shap(X_test.iloc[:5,:]), decimals=2)

    result = np.array([[ 2.5 ,  1.38,  0.1 , -3.75, -3.35, -0.15,  0.56, 10.84, 35.67],
       [-0.15,  1.69,  0.11, -3.11, -2.78,  0.12,  4.28, 10.67, 35.67],
       [11.27, -2.83,  0.05, -4.54, -2.8 ,  0.07,  0.72,  2.25, 35.67],
       [-3.7 ,  1.79,  0.1 , -2.44, -3.09, -0.01, -1.51, 10.37, 35.67],
       [ 2.78, -0.03,  0.1 , -3.48, -2.36, -0.01,  4.39, 12.09, 35.67]])
    assert(np.allclose(sv, result, rtol=.01, atol=.01))

def test_shap_simple_clf_conc():
    # Load the data
    df = pd.read_csv('tests/data_for_tests/concrete_simplified.csv')
    df.columns = ['cement', 'blast_furnace_slag', 'fly_ash', 
                       'water', 'superplasticizer', 'coarse_agg', 
                       'fine_agg', 'age_days','strength']
    df['high_strength'] = (df['strength']>50).astype(int)
    np.random.seed(999)
    foldnum = np.random.randint(0,10,df.shape[0])

    df_train = df.loc[foldnum<=7,:]
    df_test = df.loc[foldnum>7,:]
    features = df.columns[:-1].tolist()
    target = 'high_strength'
    X_train = df_train.loc[:,features]
    y_train = df_train[target].to_numpy()
    X_test = df_test.loc[:,features]
    y_test = df_test[target].to_numpy()

    fc = stb.get_basic_config(X_train, stb.default_config_dict())
    stb1 = stb.StructureBoost(feature_configs=fc, num_trees=200,
                             learning_rate=.05, max_depth=3)
    stb1.fit(X_train, y_train)
    sv = np.round(stb1.predict_shap(X_test.iloc[:5,:]), decimals=2)

    result = np.array([[ 0.01, -0., -0.01, -0.07, -0.01,  0.,  0.,  0.15, -2.78,
        -5.14],
       [-0.01, -0.01, -0.01, -0.09, -0.02,  0.  ,  0.01, -0.  , -2.07,
        -5.14],
       [ 0.01, -0.  , -0.01, -0.08, -0.01,  0.  , -0.  , -0.  , -2.84,
        -5.14],
       [-0.02, -0.  , -0.01, -0.06, -0.01,  0.  , -0.01, -0.  , -2.86,
        -5.14],
       [ 0.04,  0.01, -0.  , -0.09, -0.02,  0.01,  0.01,  0.03, 11.35,
        -5.14]])
    assert(np.allclose(sv, result, rtol=.01, atol=.01))

def test_categ_shap1():
    df_ca_PRCP = pd.read_csv('tests/data_for_tests/CA_PRCP_sm.csv')
    CA_graph = stb.graphs.CA_county_graph()
    county_int_graph, map_dict = stb.graphs.integerize_graph(CA_graph)
    df_ca_PRCP['county_int'] = df_ca_PRCP.county.apply(lambda x: map_dict[x])

    X = df_ca_PRCP.loc[:,['county_int','month']]
    y = df_ca_PRCP.rained.values
    X_train = X.iloc[:20000,:]
    y_train = y[:20000]
    X_test = X.iloc[20000:,:]
    y_test = y[20000:]
    month_graph = stb.graphs.cycle_int_graph(1,12)
    feature_configs_county = {}
    feature_configs_county['feature_type'] = 'categorical_int'
    feature_configs_county['graph'] = county_int_graph
    feature_configs_county['split_method'] = 'span_tree'
    feature_configs_county['num_span_trees'] = 1
    feature_configs_month = {}
    feature_configs_month['feature_type'] = 'categorical_int'
    feature_configs_month['graph'] = month_graph
    feature_configs_month['split_method'] = 'contraction'
    feature_configs_month['contraction_size'] = np.inf
    feature_configs_month['max_splits_to_search'] = np.inf
    feature_configs = {}
    feature_configs['county_int'] = feature_configs_county
    feature_configs['month'] = feature_configs_month

    stboost_CA = stb.StructureBoost(num_trees = 50, learning_rate=.02, feature_configs=feature_configs, 
                                 max_depth=4, mode='classification')
    stboost_CA.fit(X_train, y_train)

    sv = np.round(stboost_CA.predict_shap(X_test.iloc[:10,:].astype(float)), decimals=3)
    resmat = np.array([[ 0.399, -0.698, -1.631],
       [ 0.063,  0.564, -1.631],
       [ 0.035, -0.678, -1.631],
       [ 0.395, -0.136, -1.631],
       [ 0.107,  0.393, -1.631],
       [ 0.087,  0.59 , -1.631],
       [ 0.063,  0.587, -1.631],
       [ 0.069,  0.641, -1.631],
       [ 0.059,  0.638, -1.631],
       [-0.013,  0.368, -1.631]])
    assert(np.allclose(sv, resmat, atol=.01, rtol=.01))


