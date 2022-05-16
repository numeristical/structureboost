import numpy as np
import pandas as pd
import structureboost as stb

def test_predict_options_seattle():
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

    # Create train and test data
    feature_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'yr_built', 'zipcode_int']
    ind_list_train = ((np.arange(df.shape[0]) % 2)==0)
    ind_list_test= ((np.arange(df.shape[0]) % 2)==1)
    X_train = df.loc[ind_list_train,feature_list]
    X_test = df.loc[ind_list_test,feature_list]
    y_train = df.price[ind_list_train]
    y_test = df.price[ind_list_test]

    # Create structureboost configuration
    dcd = stb.default_config_dict()
    dcd['default_numerical_max_splits_to_search']=50
    fc = stb.get_basic_config(X_train, dcd)

    zc_int_dict = {}
    zc_int_dict['feature_type'] = 'categorical_int'
    zc_int_dict['split_method'] = 'span_tree'
    zc_int_dict['num_span_trees'] = 1
    zc_int_dict['graph'] = zipcode_int_graph
    fc['zipcode_int'] = zc_int_dict

    # Create and fit StructureBoost regression model
    stb1 = stb.StructureBoost(num_trees = 500, feature_configs=fc, mode='regression', learning_rate=.02, max_depth=3)
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
    mse = np.sqrt(np.mean((preds1-y_test)**2))
    perf_cond = (mse<167000) and (mse>160000)
    assert(equality_cond and perf_cond)


