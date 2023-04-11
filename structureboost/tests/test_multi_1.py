import numpy as np
import pandas as pd
import structureboost as stb

def test_multi_random_part():
    # Load the data
    df = pd.read_csv('tests/data_for_tests/seattle_house_sample_data.csv')

    # make the adjacency graph for Seattle zipcodes
    seattle_zipcode_list = [98177, 98133, 98155,98125,98117, 98103, 98107, 98115, 98105,
                           98199, 98119, 98109, 98102, 98112, 98122, 98121, 98101, 98154,
                           98104, 98144, 98134, 98116, 98136, 98126, 98106, 98108, 98118,
                           98146, 98168, 98178]

    seattle_zipcode_edges = [ 
                            [98177, 98133], [98177, 98117],
                            [98133, 98155], [98133, 98125], [98133, 98103],
                            [98155, 98125], [98125, 98115], [98117, 98103],
                            [98117, 98107], [98117, 98199], [98103, 98107],
                            [98103, 98115], [98103, 98105], [98103, 98109],
                            [98107, 98199], [98107, 98119], [98115, 98105],
                            [98105, 98102], [98105, 98112], [98199, 98119],
                            [98119, 98109], [98119, 98121], [98109, 98121],
                            [98109, 98102], [98102, 98112], [98102, 98122],
                            [98112, 98122], [98122, 98121], [98122, 98101],
                            [98122, 98154], [98122, 98144], [98121, 98101],
                            [98121, 98122],
                            [98101, 98154], [98101, 98122], [98154, 98122],
                            [98154, 98104],
                            [98104, 98134], [98104, 98144], [98144, 98134],
                            [98144, 98108], [98134, 98106], [98134, 98108],
                            [98116, 98136], [98116, 98126], [98136, 98126],
                            [98136, 98146], [98126, 98146], [98126, 98106],
                            [98106, 98146], [98106, 98108], [98108, 98118],
                            [98108, 98168], [98118, 98178], [98146, 98168], 
                            [98168, 98178]
                            ]
    seattle_zipcode_graph = stb.graphs.graph_undirected(seattle_zipcode_edges)

    # Map the graph and correspoding columns to integers (starting at 0)
    zipcode_int_graph, zipcode_map_dict = stb.graphs.integerize_graph(seattle_zipcode_graph)
    df['zipcode_int'] = np.array([zipcode_map_dict[i] for i in df.zipcode])

    feature_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'yr_built', 'price']


    ind_list_train = ((np.arange(df.shape[0]) % 2)==0)
    ind_list_test= ((np.arange(df.shape[0]) % 2)==1)
    X_train = df.loc[ind_list_train,feature_list]
    X_test = df.loc[ind_list_test,feature_list]
    y_train = df.zipcode_int[ind_list_train]
    y_test = df.zipcode_int[ind_list_test]





    dcd = stb.default_config_dict()
    dcd['default_numerical_max_splits_to_search']=50

    fc = stb.get_basic_config(X_train, dcd)

    ts1 = {}
    ts1['singleton_weight'] = .5
    ts1['partition_type'] = 'variable'
    ts1['random_partition_size'] = 5
    ts1['target_graph']= zipcode_int_graph


    stb1 = stb.StructureBoostMulti(num_trees = 350, target_structure=ts1, 
                                   num_classes=len(zipcode_int_graph.vertices),
                              feature_configs=fc, learning_rate=.05, max_depth=3)

    stb1.fit(X_train, y_train)
    # Create another dataframe, with columns in different order
    X_test_scr = X_test.loc[:,feature_list[5::-1]].copy()

    # Predict on 4 different variants of the test set: 2 dataframes, 2 numpy arrays
    preds1 = stb1.predict(X_test)
    preds2 = stb1.predict(X_test_scr, same_col_pos=False)
    preds3 = stb1.predict(X_test.to_numpy())

    stb1.remap_predict_columns(X_test_scr)
    preds4 = stb1.predict(X_test_scr.to_numpy())

    equality_cond = (np.allclose(preds1, preds2) and
                     np.allclose(preds2, preds3) and
                     np.allclose(preds3, preds4))
    ll = stb.log_loss(y_test, preds1)

    assert(equality_cond)
    assert((ll<2.325) and (ll>2.28))


def test_multi_fixed_part():
    # Load the data
    df = pd.read_csv('tests/data_for_tests/seattle_house_sample_data.csv')

    # make the adjacency graph for Seattle zipcodes
    seattle_zipcode_list = [98177, 98133, 98155,98125,98117, 98103, 98107, 98115, 98105,
                           98199, 98119, 98109, 98102, 98112, 98122, 98121, 98101, 98154,
                           98104, 98144, 98134, 98116, 98136, 98126, 98106, 98108, 98118,
                           98146, 98168, 98178]

    seattle_zipcode_edges = [ 
                            [98177, 98133], [98177, 98117],
                            [98133, 98155], [98133, 98125], [98133, 98103],
                            [98155, 98125], [98125, 98115], [98117, 98103],
                            [98117, 98107], [98117, 98199], [98103, 98107],
                            [98103, 98115], [98103, 98105], [98103, 98109],
                            [98107, 98199], [98107, 98119], [98115, 98105],
                            [98105, 98102], [98105, 98112], [98199, 98119],
                            [98119, 98109], [98119, 98121], [98109, 98121],
                            [98109, 98102], [98102, 98112], [98102, 98122],
                            [98112, 98122], [98122, 98121], [98122, 98101],
                            [98122, 98154], [98122, 98144], [98121, 98101],
                            [98121, 98122],
                            [98101, 98154], [98101, 98122], [98154, 98122],
                            [98154, 98104],
                            [98104, 98134], [98104, 98144], [98144, 98134],
                            [98144, 98108], [98134, 98106], [98134, 98108],
                            [98116, 98136], [98116, 98126], [98136, 98126],
                            [98136, 98146], [98126, 98146], [98126, 98106],
                            [98106, 98146], [98106, 98108], [98108, 98118],
                            [98108, 98168], [98118, 98178], [98146, 98168], 
                            [98168, 98178]
                            ]
    seattle_zipcode_graph = stb.graphs.graph_undirected(seattle_zipcode_edges)

    # Map the graph and correspoding columns to integers (starting at 0)
    zipcode_int_graph, zipcode_map_dict = stb.graphs.integerize_graph(seattle_zipcode_graph)
    df['zipcode_int'] = np.array([zipcode_map_dict[i] for i in df.zipcode])

    part1 = [[98177, 98133,  98125, 98155,98117,98115,  98107,98103,  98105],
             [98118, 98119, 98121,
           98122,  98126, 98104,  98134, 98199, 98136, 98144,
           98146, 98154,  98101, 98102,  98168,  98106,98112, 
            98108, 98116,98178,98109]]

    part2 = [[98177, 98133,  98125, 98155],[98117,98115,  98107,98103,  98105],
             [98102,98112,98122],[98118, 98119, 98121,
             98126, 98104,  98134, 98199, 98136, 98144,
           98146, 98154,  98101,   98168,  98106, 
            98108, 98116,98178,98109]]

    part3 = [[98177, 98133,  98125, 98155],
             [98117,98115,  98107,98103,  98105,98102,98112,98122],
             [98108,98118, 98168,98178],[98119, 98121,
             98126, 98104,  98134, 98199, 98136, 98144,
           98146, 98154,  98101,     98106, 
             98116,98109]]
    part1a = [[zipcode_map_dict[i] for i in sublist] for sublist in part1]
    part2a = [[zipcode_map_dict[i] for i in sublist] for sublist in part2]
    part3a = [[zipcode_map_dict[i] for i in sublist] for sublist in part3]
    p_list = [part1a, part2a, part3a]

    feature_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'yr_built', 'price']


    ind_list_train = ((np.arange(df.shape[0]) % 2)==0)
    ind_list_test= ((np.arange(df.shape[0]) % 2)==1)
    X_train = df.loc[ind_list_train,feature_list]
    X_test = df.loc[ind_list_test,feature_list]
    y_train = df.zipcode_int[ind_list_train]
    y_test = df.zipcode_int[ind_list_test]





    dcd = stb.default_config_dict()
    dcd['default_numerical_max_splits_to_search']=50

    fc = stb.get_basic_config(X_train, dcd)

    ts2 = {}
    ts2['partition_type'] = 'fixed'
    ts2['singleton_weight'] = .6
    ts2['partition_list'] = p_list
    ts2['partition_weight_vec'] = [.4/3,.4/3,.4/3]


    stb1 = stb.StructureBoostMulti(num_trees = 350, target_structure=ts2, 
                                   num_classes=len(zipcode_int_graph.vertices),
                              feature_configs=fc, learning_rate=.05, max_depth=3)

    stb1.fit(X_train, y_train)
    # Create another dataframe, with columns in different order
    X_test_scr = X_test.loc[:,feature_list[5::-1]].copy()

    # Predict on 4 different variants of the test set: 2 dataframes, 2 numpy arrays
    preds1 = stb1.predict(X_test)
    preds2 = stb1.predict(X_test_scr, same_col_pos=False)
    preds3 = stb1.predict(X_test.to_numpy())

    stb1.remap_predict_columns(X_test_scr)
    preds4 = stb1.predict(X_test_scr.to_numpy())

    equality_cond = (np.allclose(preds1, preds2) and
                     np.allclose(preds2, preds3) and
                     np.allclose(preds3, preds4))
    ll = stb.log_loss(y_test, preds1)
    assert(equality_cond)
    assert((ll<2.325) and (ll>2.28))




