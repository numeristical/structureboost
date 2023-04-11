import numpy as np
import pandas as pd
import structureboost as stb

def test_corsage_conc_auto():
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
    stbnew = stb.Coarsage(num_trees = 2000, feature_configs= fc, num_coarse_bins=50, 
                             structure_block_size=15,
                             max_resolution=3000, learning_rate=0.02, max_depth=10)    
    stbnew.fit(X_train, y_train, eval_set=(X_test, y_test), early_stop_past_steps=5)
    preds = stbnew.predict_distributions(X_test)
    llval = preds.log_loss(y_test)
    assert(llval<2.93)

def test_corsage_conc_fixed():
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
    stbnew = stb.Coarsage(num_trees = 2000, feature_configs= fc, num_coarse_bins=50,
                            structure_block_size=15, 
                             max_resolution=3000, learning_rate=0.02, max_depth=10)
    stbnew.fit(X_train, y_train, eval_set=(X_test, y_test), early_stop_past_steps=5)
    preds = stbnew.predict_distributions(X_test)
    llval = preds.log_loss(y_test)
    assert(llval<3.55)

def test_corsage_conc_fixed_rss():
    # Load the data
    df = pd.read_csv('tests/data_for_tests/concrete_simplified.csv')

    np.random.seed(999)
    foldnum = np.random.randint(0,10,df.shape[0])
    df_train = df[foldnum<=6]
    df_valid = df[(foldnum>6) & (foldnum<=7)]
    df_test = df[foldnum>7]

    X_train = df_train.iloc[:,:-1]
    y_train = df_train.strength.to_numpy()

    X_valid = df_valid.iloc[:,:-1]
    y_valid = df_valid.strength.to_numpy()

    X_test = df_test.iloc[:,:-1]
    y_test = df_test.strength.to_numpy()

    fc = stb.get_basic_config(X_train, stb.default_config_dict())
    stbnew = stb.Coarsage(num_trees = 2000, feature_configs= fc, num_coarse_bins=50, 
                             max_resolution=1000, learning_rate=0.05, max_depth=10)    
    stbnew.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stop_past_steps=5)
    preds = stbnew.predict_distributions(X_test)
    llval = preds.log_loss(y_test)
    assert(llval<3.38)

