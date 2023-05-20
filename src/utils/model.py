import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


class CVModel():
    import numpy as np
    def __init__(self, models_folder, num_class=3):
        from os import listdir
        import lightgbm
        model_files = [f for f in listdir(models_folder) if '.model' in f]

        self.models = []
        for model_file in model_files:             
            self.models.append(lightgbm.Booster(model_file= models_folder + model_file, params={'n_jobs':1}))
        self.num_class = num_class

    def predict(self, Y):
        import numpy as np
        import pandas as pd
        prediction = np.zeros((Y.shape[0], self.num_class))
        for model in self.models:

            if(isinstance(Y, pd.DataFrame)):
                prediction += model.predict(Y[model.feature_name()])
            else:
                prediction += model.predict(Y)

        return prediction / len(self.models)
    
    def predict_labels(self, Y):
        predictions = self.predict(Y)
        labels = np.argmax(predictions, axis=1)
        
        return labels

def base_cv(
    train,
    target,
    train_features,
    cat_features=None,
    random_state=42,
    n_folds=5,
    model_folder="models",
    model_name="model",
    model_params=None,
    metric=None
):
    cat_feats_ind = [i for i, j in enumerate(train_features) if j in cat_features]
    model_params["categorical_column"] = cat_feats_ind
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    preds = []
    scores = []
    for i, (train_idx, test_idx) in enumerate(skf.split(train[train_features], train[target])):
        X_train, y_train = train[train_features].iloc[train_idx], train[target].iloc[train_idx]
        X_test, y_test = train[train_features].iloc[test_idx], train[target].iloc[test_idx]
        model = lgb.LGBMModel(**model_params)
        e_stop = round(5 / model.get_params()['learning_rate'])
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test), 
            early_stopping_rounds=e_stop,
            eval_metric=model.metric,
            verbose=False
        )
        model.booster_.save_model(os.path.join(model_folder, model_name, f"fold_{i}.model",))
        fold_preds = model.predict(X_test)
        fold_labels = np.argmax(fold_preds, axis=1)
        fold_score = metric(y_test, fold_labels)
        preds.append(fold_preds)
        scores.append(fold_score)
        print(
            i,
            "it:", model.best_iteration_,
            "score:", fold_score
        )
    return preds, scores

