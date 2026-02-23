## package imports
from pickle import dump, load
import os
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import recall_score, precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
import skops.io as sio
import shap
import numpy as np

## perform modelling and build shap explanation
## solely for grading purpose
## More detail on modelling and shap_explanation notebooks

def modelling():
    """
    Perform modelling. Dataset considered is from feature_selection folder
    """
    ## data loading
    with open('data/feature_sel/X_train.pkl', 'rb') as file:
        X_train = load(file)

    with open('data/feature_sel/y_train.pkl', 'rb') as file:
        y_train = load(file)

    with open('data/feature_sel/X_test.pkl', 'rb') as file:
        X_test = load(file)

    with open('data/feature_sel/y_test.pkl', 'rb') as file:
        y_test = load(file)

    ## define CV
    skf = StratifiedKFold(shuffle=True, random_state=42)
    skf.split(X_train, y_train)

    ## Fit model
    main_clf = HistGradientBoostingClassifier(max_iter=1000,
                                            categorical_features=['Geography', 'Gender'],
                                            early_stopping=True, ## force early stopping
                                            random_state=42,
                                            class_weight='balanced') ## weighting the class due to imbalanced
                                                                    ## data

    main_param = {'learning_rate': [0.01, 0.05, 0.1, 0.12, 0.2, 0.3],
                'max_leaf_nodes': [(2**i)-1 for i in range(2,8)]}

    best_main = GridSearchCV(main_clf, main_param,
                            scoring=['recall', 'precision'], ## report both recall and precision
                            refit='recall', ## optimize recall
                            cv=skf)

    best_main.fit(X_train, y_train) ## NOTE: this already count as model

    print(f"Best Strategy: {best_main.best_params_}")
    print(f"Best Recall (cv average): {best_main.best_score_}")

    ## save model
    mdl = best_main.best_estimator_ ## load best model from Tuning object
    if not os.path.exists('data/modelling'):
        os.makedirs('data/modelling')

    obj_skops = sio.dump(mdl, "data/modelling/model.skops")

    return None

def shap_explanation():
    """
    build shap explanation
    """
    ## data and model loading
    with open('data/feature_sel/X_test.pkl', 'rb') as file:
        X_test = load(file)

    with open('data/feature_sel/y_test.pkl', 'rb') as file:
        y_test = load(file)

    unknown_types = sio.get_untrusted_types(file="data/modelling/model.skops")
    model = sio.load("data/modelling/model.skops", trusted=unknown_types)
    unknown_types = sio.get_untrusted_types(file="data/feature_eng/encoder.skops")
    encoder = sio.load("data/feature_eng/encoder.skops", trusted=unknown_types)

    ## prepare explanation
    ## define log proba for model
    def predict_log_proba(z):
        p = model.predict_proba(z)
        return np.log(p[:,1] / p[:,0])

    background_data = shap.maskers.Independent(X_test, max_samples=100)
    explainer = shap.Explainer(predict_log_proba, background_data)

    ## get explanation
    shap_values = explainer(X_test) ## recompute shap values for entire dataset

    ## no plotting performed here. Refer for Get Explanation section on shap_explanation.ipynb

    ## save explanation result
    if not os.path.exists('data/explainer'):
        os.makedirs('data/explainer')

    with open('data/explainer/explainer.pkl', 'wb') as file:
        dump(explainer, file)

    with open('data/explainer/shap_values.pkl', 'wb') as file:
        dump(shap_values, file)

    return None

modelling()
shap_explanation()