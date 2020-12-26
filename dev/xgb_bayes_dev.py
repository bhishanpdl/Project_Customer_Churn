pbounds = {
    'learning_rate': (0.01, 1.0),
    'n_estimators': (100, 1000),
    'max_depth': (3,10),
    'subsample': (1.0, 1.0),  # Change for big datasets
    'colsample': (1.0, 1.0),  # Change for datasets with lots of features
    'gamma': (0, 5)}

def xgb_cval(learning_rate,
                        n_estimators,
                        max_depth,
                        subsample,
                        colsample,
                        gamma):

    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    clf = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma)
    return np.mean(cross_val_score(clf, df_Xtrain, ytrain, cv=3, scoring='roc_auc'))

optimizer = BayesianOptimization(
    f=xgb_cval,
    pbounds=pbounds,
    random_state=1,
)

# optimizer.maximize(n_iter=10)
# print("Final result:", optimizer.max)

#========================================================================
def xgb_cval(learning_rate, n_estimators, max_depth, subsample, colsample, gamma):
    # params
    global df_train, ser_ytrain
    np.random.seed(SEED)
    n_splits = 5

    # make integer values as integer
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    estimator = xgb.XGBClassifier(
        eval_metric='auc',
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma)

    # initialize KFold and progress bar
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    arr_eval = np.zeros((n_estimators, n_splits))
    oof_predictions = np.empty(len(df_train))

    progress_bar = tqdm(
        enumerate(skf.split(ytrain, ytrain)),
        total=n_splits,
        leave=False
    )

    # cross-validation
    for fold, (idx_tr, idx_vd) in progress_bar:
        # data to fit
        Xtr, Xvd = df_train.iloc[idx_tr], df_train.iloc[idx_vd]
        ytr, yvd = ser_ytrain.iloc[idx_tr], ser_ytrain.iloc[idx_vd]

        # model
        model = estimator
        eval_set = [(Xtr, ytr), (Xvd, yvd)]
        model.fit(Xtr, ytr, eval_set=eval_set, verbose=False)

        # best fold
        arr_eval[:, fold] = model.evals_result_["validation_1"]['auc']
        best_round = np.argsort(arr_eval[:, fold])[0]

        # progress bar update
        txt = 'Fold #{}:   {:.5f}'.format(fold, arr_eval[:, fold][best_round])
        progress_bar.set_description(txt, refresh=True)

    # find best round from best mean
    mean_eval, std_eval = np.mean(arr_eval, axis=1), np.std(arr_eval, axis=1)
    best_round = np.argsort(mean_eval)[0]

    val_maximize = mean_eval[best_round]
    return val_maximize


# =============================================
pbounds = {
    'learning_rate': (0.01, 1.0),
    'n_estimators': (100, 1000),
    'max_depth': (3, 10),
    'subsample': (1.0, 1.0),
    'colsample': (1.0, 1.0),
    'gamma': (0, 5)}

optimizer = BayesianOptimization(
    f=xgb_cval,
    pbounds=pbounds,
    random_state=SEED,
)

# optimizer.maximize(n_iter=2)
# print("Final result:", optimizer.max)