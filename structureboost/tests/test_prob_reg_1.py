import numpy as np
import pandas as pd
import structureboost as stb

def test_prob_reg_simple_conc():
    # Load the data
    df = pd.read_csv('tests/data_for_tests/concrete_simplified.csv')

    np.random.seed(999)
    foldnum = np.random.randint(0,10,df.shape[0])

    df_train = df.loc[foldnum<=7,:]
    df_test = df.loc[foldnum>7,:]
    X_train = df_train.iloc[:,:-1]
    y_train = df_train.strength.to_numpy()
    X_test = df_test.iloc[:,:-1]
    y_test = df_test.strength.to_numpy()

    fc = stb.get_basic_config(X_train, stb.default_config_dict())
    stb1 = stb.PrestoBoost(num_forests = 10,
                            num_trees = 200,
                            feature_configs= fc,
                            learning_rate=0.05,
                            )
    stb1.fit(X_train, y_train)
    test_dists_conc = stb1.predict_distributions(X_test)
    ll_val = test_dists_conc.log_loss(y_test)
    assert(ll_val<3.3)

def test_prob_reg_evalset_conc():
    # Load the data
    df = pd.read_csv('tests/data_for_tests/concrete_simplified.csv')

    np.random.seed(999)
    foldnum = np.random.randint(0,10,df.shape[0])

    df_train = df.loc[foldnum<7,:]
    df_valid = df.loc[foldnum==7,:]
    df_test = df.loc[foldnum>7,:]
    X_train = df_train.iloc[:,:-1]
    y_train = df_train.strength.to_numpy()
    X_valid = df_valid.iloc[:,:-1]
    y_valid = df_valid.strength.to_numpy()
    X_test = df_test.iloc[:,:-1]
    y_test = df_test.strength.to_numpy()

    fc = stb.get_basic_config(X_train, stb.default_config_dict())
    stb1 = stb.PrestoBoost(num_forests = 10,
                            num_trees = 2000,
                            feature_configs= fc,
                            learning_rate=0.05,
                            )
    stb1.fit(X_train, y_train, eval_set=(X_valid, y_valid), 
            early_stop_past_steps=3)
    test_dists_conc = stb1.predict_distributions(X_test)
    ll_val = test_dists_conc.log_loss(y_test)
    assert(ll_val<3.3)

