import numpy as np
import pandas as pd
import structureboost as stb



def test_prem_pred_onehot():
    df_train = pd.read_csv('tests/data_for_tests/df_prem_pred_train.csv')
    df_test = pd.read_csv('tests/data_for_tests/df_prem_pred_test.csv')
    feature_list = ['state', 'home_dwell_cov', 'home_pp_cov', 'pp_ratio']
    X_train = df_train.loc[:, feature_list]
    X_test = df_test.loc[:, feature_list]
    y_train = df_train.outcome
    y_test = df_test.outcome

    US_49_graph = stb.graphs.US_48_and_DC_graph()
    state_graph_int, state_int_map = stb.graphs.integerize_graph(US_49_graph)
    reverse_dict = {state_int_map[key]: key for key in state_int_map.keys()}
    X_train['state'] = X_train['state'].apply(lambda x: reverse_dict[x])
    X_test['state'] = X_test['state'].apply(lambda x: reverse_dict[x])

    my_feature_configs = {}
    my_feature_configs_hdc = {}
    my_feature_configs_hdc['feature_type'] = 'numerical'
    my_feature_configs_hdc['max_splits_to_search'] = 1000
    my_feature_configs_ppr = {}
    my_feature_configs_ppr['feature_type'] = 'numerical'
    my_feature_configs_ppr['max_splits_to_search'] = 1000
    my_feature_configs_state = {}
    my_feature_configs_state['feature_type'] = 'categorical_str'
    my_feature_configs_state['split_method'] = 'onehot'
    my_feature_configs_state['feature_vals'] = list(US_49_graph.vertices)
    my_feature_configs['state'] = my_feature_configs_state
    my_feature_configs['home_dwell_cov'] = my_feature_configs_hdc
    my_feature_configs['pp_ratio'] = my_feature_configs_ppr

    my_stboost = stb.StructureBoost(num_trees=200, learning_rate=.02,
                                     feature_configs=my_feature_configs,
                                     max_depth=3, mode='regression',
                                     loss_fn='mse')
    my_stboost.fit(X_train, y_train)
    answers = my_stboost.predict(X_test)
    loss_val = np.mean((y_test-answers)**2)
    assert ((loss_val < 600000) and (loss_val > 560000))


def test_graphical_vor():
    df_train = pd.read_csv('tests/data_for_tests/train_set_vor.csv')
    df_test = pd.read_csv('tests/data_for_tests/test_set_vor.csv')
    feature_list = ['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
    X_train = df_train.loc[:, feature_list]
    X_test = df_test.loc[:, feature_list]
    y_train = df_train.outcome
    y_test = df_test.outcome

    my_feature_configs = {}
    my_feature_configs_vor0 = {}
    my_feature_configs_vor0['feature_type'] = 'graphical_voronoi'
    my_feature_configs_vor0['voronoi_sample_size'] = 8
    my_feature_configs_vor0['voronoi_min_pts'] = 10
    my_feature_configs_vor0['split_method'] = 'span_tree'
    my_feature_configs_vor0['num_span_trees'] = 1
    my_feature_configs_vor0['subfeature_list'] = ['X0', 'Y0', 'Z0']

    my_feature_configs_vor1 = {}
    my_feature_configs_vor1['feature_type'] = 'graphical_voronoi'
    my_feature_configs_vor1['voronoi_sample_size'] = 8
    my_feature_configs_vor1['voronoi_min_pts'] = 10
    my_feature_configs_vor1['split_method'] = 'contraction'
    my_feature_configs_vor1['contraction_size'] = 7
    my_feature_configs_vor1['max_splits_to_search'] = 15
    my_feature_configs_vor1['subfeature_list'] = ['X1', 'Y1', 'Z1']

    my_feature_configs_vor2 = {}
    my_feature_configs_vor2['feature_type'] = 'graphical_voronoi'
    my_feature_configs_vor2['voronoi_sample_size'] = 8
    my_feature_configs_vor2['voronoi_min_pts'] = 10
    my_feature_configs_vor2['split_method'] = 'span_tree'
    my_feature_configs_vor2['num_span_trees'] = 1
    my_feature_configs_vor2['subfeature_list'] = ['X2', 'Y2', 'Z2']

    my_feature_configs['vor_comp_0'] = my_feature_configs_vor0
    my_feature_configs['vor_comp_1'] = my_feature_configs_vor1
    my_feature_configs['vor_comp_2'] = my_feature_configs_vor2

    my_stboost = stb.StructureBoost(num_trees=100, learning_rate=.01,
                                     feature_configs=my_feature_configs,
                                     max_depth=3, mode='classification',
                                     loss_fn='entropy')
    my_stboost.fit(X_train, y_train)
    answers = my_stboost.predict(X_test)
    loss_val = stb.log_loss(y_test, answers)
    assert ((loss_val < .41) and (loss_val > .37))


def test_prem_pred_int():
    df_train = pd.read_csv('tests/data_for_tests/df_prem_pred_train.csv')
    df_test = pd.read_csv('tests/data_for_tests/df_prem_pred_test.csv')
    feature_list = ['state', 'home_dwell_cov', 'home_pp_cov', 'pp_ratio']
    X_train = df_train.loc[:, feature_list]
    X_test = df_test.loc[:, feature_list]
    y_train = df_train.outcome
    y_test = df_test.outcome

    US_49_graph = stb.graphs.US_48_and_DC_graph()
    state_graph_int, state_int_map = stb.graphs.integerize_graph(US_49_graph)

    my_feature_configs = {}
    my_feature_configs_hdc = {}
    my_feature_configs_hdc['feature_type'] = 'numerical'
    my_feature_configs_hdc['max_splits_to_search'] = 1000
    my_feature_configs_ppr = {}
    my_feature_configs_ppr['feature_type'] = 'numerical'
    my_feature_configs_ppr['max_splits_to_search'] = 1000
    my_feature_configs_state = {}
    my_feature_configs_state['feature_type'] = 'categorical_int'
    my_feature_configs_state['graph'] = state_graph_int
    my_feature_configs_state['split_method'] = 'span_tree'
    my_feature_configs_state['num_span_trees'] = 1
    my_feature_configs['state'] = my_feature_configs_state
    my_feature_configs['home_dwell_cov'] = my_feature_configs_hdc
    my_feature_configs['pp_ratio'] = my_feature_configs_ppr

    my_stboost = stb.StructureBoost(num_trees=200, learning_rate=.02,
                                     feature_configs=my_feature_configs,
                                     max_depth=3, mode='regression',
                                     loss_fn='mse')
    my_stboost.fit(X_train, y_train)
    answers = my_stboost.predict(X_test)
    loss_val = np.mean((y_test-answers)**2)
    assert ((loss_val < 560000) and (loss_val > 500000))


def test_prem_pred_str():
    df_train = pd.read_csv('tests/data_for_tests/df_prem_pred_train.csv')
    df_test = pd.read_csv('tests/data_for_tests/df_prem_pred_test.csv')
    feature_list = ['state', 'home_dwell_cov', 'home_pp_cov', 'pp_ratio']
    X_train = df_train.loc[:, feature_list]
    X_test = df_test.loc[:, feature_list]
    y_train = df_train.outcome
    y_test = df_test.outcome

    US_49_graph = stb.graphs.US_48_and_DC_graph()
    state_graph_int, state_int_map = stb.graphs.integerize_graph(US_49_graph)
    reverse_dict = {state_int_map[key]: key for key in state_int_map.keys()}
    X_train['state'] = X_train['state'].apply(lambda x: reverse_dict[x])
    X_test['state'] = X_test['state'].apply(lambda x: reverse_dict[x])

    my_feature_configs = {}
    my_feature_configs_hdc = {}
    my_feature_configs_hdc['feature_type'] = 'numerical'
    my_feature_configs_hdc['max_splits_to_search'] = 1000
    my_feature_configs_ppr = {}
    my_feature_configs_ppr['feature_type'] = 'numerical'
    my_feature_configs_ppr['max_splits_to_search'] = 1000
    my_feature_configs_state = {}
    my_feature_configs_state['feature_type'] = 'categorical_str'
    my_feature_configs_state['graph'] = US_49_graph
    my_feature_configs_state['split_method'] = 'contraction'
    my_feature_configs_state['max_splits_to_search'] = 50
    my_feature_configs_state['contraction_size'] = 9
    my_feature_configs['state'] = my_feature_configs_state
    my_feature_configs['home_dwell_cov'] = my_feature_configs_hdc
    my_feature_configs['pp_ratio'] = my_feature_configs_ppr

    my_stboost = stb.StructureBoost(num_trees=200, learning_rate=.02,
                                     feature_configs=my_feature_configs,
                                     max_depth=3, mode='regression',
                                     loss_fn='mse')
    my_stboost.fit(X_train, y_train)
    answers = my_stboost.predict(X_test)
    loss_val = np.mean((y_test-answers)**2)
    assert ((loss_val < 565000) and (loss_val > 500000))


def test_prem_pred_3():
    df_train = pd.read_csv('tests/data_for_tests/df_prem_pred_train.csv')
    df_test = pd.read_csv('tests/data_for_tests/df_prem_pred_test.csv')
    feature_list = ['state', 'home_dwell_cov', 'home_pp_cov', 'pp_ratio']
    X_train = df_train.loc[:, feature_list]
    X_test = df_test.loc[:, feature_list]
    y_train = df_train.outcome
    y_test = df_test.outcome

    US_49_graph = stb.graphs.US_48_and_DC_graph()
    state_graph_int, state_int_map = stb.graphs.integerize_graph(US_49_graph)

    my_feature_configs = {}
    my_feature_configs_hdc = {}
    my_feature_configs_hdc['feature_type'] = 'numerical'
    my_feature_configs_hdc['max_splits_to_search'] = 1000
    my_feature_configs_ppr = {}
    my_feature_configs_ppr['feature_type'] = 'numerical'
    my_feature_configs_ppr['max_splits_to_search'] = 1000
    my_feature_configs_state = {}
    my_feature_configs_state['feature_type'] = 'categorical_int'
    my_feature_configs_state['graph'] = state_graph_int
    my_feature_configs_state['split_method'] = 'contraction'
    my_feature_configs_state['max_splits_to_search'] = 50
    my_feature_configs_state['contraction_size'] = 9
    my_feature_configs['state'] = my_feature_configs_state
    my_feature_configs['home_dwell_cov'] = my_feature_configs_hdc
    my_feature_configs['pp_ratio'] = my_feature_configs_ppr

    my_stboost = stb.StructureBoost(num_trees=200, learning_rate=.02,
                                     feature_configs=my_feature_configs,
                                     max_depth=3, mode='regression',
                                     loss_fn='mse')
    my_stboost.fit(X_train, y_train)
    answers = my_stboost.predict(X_test)
    loss_val = np.mean((y_test-answers)**2)
    assert ((loss_val < 565000) and (loss_val > 500000))


def test_ins_data_na():
    df_ins = pd.read_csv('tests/data_for_tests/ins_data.csv')
    features_1a_stb = ['auto_premium', 'home_dwell_cov', 'state_abbrev']
    X_train_1a_stb = df_ins.loc[df_ins['fold_num_2'] >= 2, features_1a_stb]
    X_test_1a_stb = df_ins.loc[df_ins['fold_num_2'] == 0, features_1a_stb]
    y_train = df_ins.has_umbrella[df_ins['fold_num_2'] >= 2]
    y_test = df_ins.has_umbrella[df_ins['fold_num_2'] == 0]
    start_config = stb.default_config_dict()
    feat_configs_1a = stb.get_basic_config(X_train_1a_stb, start_config)
    feat_configs_1a['state_abbrev']['graph'] = stb.graphs.US_50_and_DC_graph()

    stboost_def = stb.StructureBoost(num_trees=400,
                                      learning_rate=.02,
                                      feature_configs=feat_configs_1a,
                                      max_depth=2, subsample=.8, replace=False,
                                      mode='classification',
                                      loss_fn='entropy')
    stboost_def.fit(X_train_1a_stb, y_train)

    pred_probs_1a_stb = stboost_def.predict(X_test_1a_stb)
    loss_val = stb.log_loss(y_test, pred_probs_1a_stb)
    assert ((loss_val < .274) and (loss_val > .270))


def test_ins_data_na_2():
    df_ins = pd.read_csv('tests/data_for_tests/ins_data.csv')
    features_2_stb = ['auto_premium', 'homeown_premium', 'home_dwell_cov',
                      'home_pers_prop_cov', 'num_home_pol', 'yob_policyholder',
                      'min_vehicle_year', 'max_vehicle_year', 'num_vehicles',
                      'max_driver_yob', 'min_driver_yob',
                      'median_household_income', 'median_house_value',
                      'state_abbrev']
    X_train_2_stb = df_ins.loc[df_ins['fold_num_2'].isin(
                                [0, 1, 3]), features_2_stb]
    X_test_2_stb = df_ins.loc[df_ins['fold_num_2'] == 4, features_2_stb]

    y_train = df_ins.has_umbrella[df_ins['fold_num_2'].isin([0, 1, 3])]
    y_test = df_ins.has_umbrella[df_ins['fold_num_2'] == 4]

    start_config = stb.default_config_dict()
    start_config['default_numerical_max_splits_to_search'] = 20
    feat_configs_2 = stb.get_basic_config(X_train_2_stb, start_config)
    feat_configs_2['state_abbrev']['graph'] = stb.graphs.US_50_and_DC_graph()
    feat_configs_2['state_abbrev']['num_span_trees'] = 2
    feat_configs_2['auto_premium']['max_splits_to_search'] = 5

    stboost_2 = stb.StructureBoost(num_trees=800,
                                    learning_rate=.02,
                                    feature_configs=feat_configs_2,
                                    max_depth=3, feat_sample_by_tree=.66,
                                    feat_sample_by_node=.5,
                                    subsample=.8,
                                    mode='classification',
                                    loss_fn='entropy',
                                    reg_lambda=5, gamma=10)
    stboost_2.fit(X_train_2_stb, y_train)

    pred_probs_2 = stboost_2.predict(X_test_2_stb)
    loss_val = stb.log_loss(y_test, pred_probs_2)
    assert ((loss_val < .269) and (loss_val > .262))


def test_ins_data_na_3():
    df_ins = pd.read_csv('tests/data_for_tests/ins_data.csv')
    features_2_stb = ['auto_premium', 'homeown_premium', 'home_dwell_cov',
                      'home_pers_prop_cov', 'num_home_pol',
                      'yob_policyholder', 'min_vehicle_year',
                      'max_vehicle_year', 'num_vehicles',
                      'max_driver_yob', 'min_driver_yob',
                      'median_household_income',
                      'median_house_value', 'state_abbrev']
    X_train_2_stb = df_ins.loc[df_ins['fold_num_2'].isin(
                                    [0, 1, 3]), features_2_stb]
    X_test_2_stb = df_ins.loc[df_ins['fold_num_2'] == 4, features_2_stb]

    y_train = df_ins.has_umbrella[df_ins['fold_num_2'].isin([0, 1, 3])]
    y_test = df_ins.has_umbrella[df_ins['fold_num_2'] == 4]

    start_config = stb.default_config_dict()
    start_config['default_numerical_max_splits_to_search'] = 20
    feat_configs_2 = stb.get_basic_config(X_train_2_stb, start_config)
    feat_configs_2['state_abbrev']['graph'] = stb.graphs.US_50_and_DC_graph()
    feat_configs_2['state_abbrev']['num_span_trees'] = 2
    feat_configs_2['auto_premium']['max_splits_to_search'] = 5

    stboost_2 = stb.StructureBoost(num_trees=700,
                                    learning_rate=.02,
                                    feature_configs=feat_configs_2,
                                    max_depth=3, feat_sample_by_tree=.66,
                                    feat_sample_by_node=.5,
                                    subsample=.8,
                                    mode='classification',
                                    loss_fn='entropy', reg_lambda=2, gamma=1)
    stboost_2.fit(X_train_2_stb, y_train)

    pred_probs_2 = stboost_2.predict(X_test_2_stb)
    loss_val = stb.log_loss(y_test, pred_probs_2)
    assert ((loss_val < .264) and (loss_val > .257))
