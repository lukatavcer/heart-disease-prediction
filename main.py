import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz  # plot tree
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error, accuracy_score
from math import sqrt
from sklearn.metrics import classification_report  # for model evaluation
from sklearn.metrics import confusion_matrix  # for model evaluation
from sklearn.model_selection import train_test_split  # for data splitting
import eli5  # for permutation importance
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, info_plots  # for partial plots
from subprocess import call
from IPython.display import Image
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# This method is called when RandomState is initialized. It can be called again to re-seed the generator
np.random.seed(0)

# Controls the randomness of the estimator. If int, random_state is the seed used by the random number generator;
# If RandomState instance, random_state is the random number generator; If None, the random number generator
# is the RandomState instance used by np.random.
random_state = 10


def rename_df2(df):
    # df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
    df.columns = ['age', 'sex', 'chest_pain_type',
    #               'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'target']
                  'max_heart_rate_achieved', 'exercise_induced_angina', 'target']

    df['sex'][df['sex'] == 0] = 'female'
    df['sex'][df['sex'] == 1] = 'male'

    df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'
    df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'
    df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'
    df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomatic'

    df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'
    df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'

    return df


def rename_df(df):
    # Rename columns
    df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                  'rest_ecg', 'max_heart_rate_achieved',
                  'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

    df['sex'][df['sex'] == 0] = 'female'
    df['sex'][df['sex'] == 1] = 'male'

    df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'
    df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'
    df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'
    df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomatic'

    df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
    df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

    df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'
    df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'
    df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'

    df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'
    df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'

    df['st_slope'][df['st_slope'] == 1] = 'upsloping'
    df['st_slope'][df['st_slope'] == 2] = 'flat'
    df['st_slope'][df['st_slope'] == 3] = 'downsloping'

    df['thalassemia'][df['thalassemia'] == 1] = 'normal'
    df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'
    df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'

    return df


def eval(conf_matrix):
    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    print(f'Sensitivity: {round(sensitivity, 2)}')

    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    print(f'Specificity: {round(specificity, 2)}')

    TP = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TN = conf_matrix[1][1]
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    print(f'    FPR: {round(FPR, 2)} {FP} of {(FP + TN)}')
    print(f'    FNR: {round(FNR, 2)} {FN} of {(TP + FN)}')


def calc_metrics(y_test, y_predict):
    mae = round(mean_absolute_error(y_test, y_predict), 2)
    rmse = round(sqrt(mean_squared_error(y_test, y_predict)), 2)
    acc = round(accuracy_score(y_test, y_predict), 2)
    print(f"Accuracy: {acc}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    return mae, rmse, acc


def split_data(df):
    """
        x_train is the training part of the matrix of features.
        x_test is the test part of the matrix of features.
        y_train is the training part of the dependent variable that is associated to X_train here.
        y_test is the test part of the dependent variable that is associated to X_train here.
    """
    # Feature scaling between 0 and 1?
    # Sometimes machine models are not based on Euclidean distances, we will still need to do features scaling because
    # the algorithm will converge much faster. That will be the case for Decision Tree which are not based on ED but,
    # if we do not do feature scaling then they will run a very long time.
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop('target', 1), df['target'],
        test_size=0.2,
        random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def plot_tree(model, x_train, y_train, filename):
    y_train_str = y_train.astype('str')
    y_train_str[y_train_str == '0'] = 'no disease'
    y_train_str[y_train_str == '1'] = 'disease'
    y_train_str = y_train_str.values

    # Extract single tree
    feature_names = [i for i in x_train.columns]
    export_graphviz(model, out_file='tree.dot',
                    feature_names=feature_names,
                    class_names=y_train_str,
                    rounded=True, proportion=True,
                    label='root',
                    precision=2, filled=True)

    call(['dot', '-Tpng', 'tree.dot', '-o', filename, '-Gdpi=600'])
    Image(filename=filename)


def decision_tree(x_train, x_test, y_train, y_test, max_leaf_nodes=10):
    """
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    # Grow a tree with max_leaf_nodes in best-first fashion.
    # Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=random_state)

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    mae, rmse, acc = calc_metrics(y_test, y_predict)

    # Visualize the tree
    # plot_tree(model, x_train, y_train, 'dec_tree.png')

    # Get the probability for the classes of the data.
    y_pred_quant = model.predict_proba(x_test)[:, 1]

    # Plot ROC
    plot_roc(y_test, y_predict, y_pred_quant, "decision tree")

    return round(mae, 2), round(rmse, 2), round(acc, 2)


def random_forest(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(max_depth=4)
    model.fit(x_train, y_train)

    # Visualize one tree from the random forest
    # estimator = model.estimators_[1]
    # plot_tree(estimator, x_train, y_train, "rf_tree.png")

    y_predict = model.predict(x_test)

    # Get the probability for the classes of the data.
    y_pred_quant = model.predict_proba(x_test)[:, 1]
    # Plot ROC
    plot_roc(y_test, y_predict, y_pred_quant, "random forest")
    calc_metrics(y_test, y_predict)


def svm(x_train, x_test, y_train, y_test):
    model = SVC(random_state=random_state, kernel='linear', probability=True)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    # Get the probability for the classes of the data.
    y_pred_quant = model.predict_proba(x_test)[:, 1]

    # Plot ROC
    plot_roc(y_test, y_predict, y_pred_quant, "SVM")
    calc_metrics(y_test, y_predict)


def plot_roc(y_test, y_pred_bin, y_pred_quant, classifier=None):
    conf_matrix = confusion_matrix(y_test, y_pred_bin)

    # Evaluate
    eval(conf_matrix)

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title(f'ROC curve for the {classifier} classifier')
    plt.xlabel('FP rate (1 - Specificity)')
    plt.ylabel('TP rate (Sensitivity)')
    plt.grid(True)
    plt.show()

    auc_score = round(auc(fpr, tpr), 2)
    print(f"AUC: {auc_score}")
    return auc_score


def get_redundant_pairs(df):
    # Drop diagonal pairs and lower triangle
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def plot_correlations(df):
    plt.figure(figsize=(16, 8), dpi=300)
    g = df.corr()
    data = g.index
    sns.heatmap(df[data].corr(), annot=True)

    drop_labels = get_redundant_pairs(df)
    corrs = g.unstack().drop(labels=drop_labels).sort_values(ascending=False)
    # print(corrs)
    return corrs


def main():
    df = pd.read_csv("data/heart.csv")

    # Create Pandas Profile report.
    # profile = ProfileReport(df, title="Pandas Profiling Report")
    # profile.to_file("profile_report_renamed.html")

    # Top correlations to target:
    # positive: cp, thalach
    # negative: oldpeak, exang
    # plot_correlations(df)
    # only_top_corrs = True
    only_top_corrs = False

    # Try predicting with only these 4, most highly correlated attributes.
    # + age, sex
    # + target
    if only_top_corrs:
        # no oldpeak
        # df = df.filter(['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang', 'target'])
        # with oldpeak
        # df = df.filter(['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang', 'oldpeak', 'target'])
        # df = df.filter(['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'target'])

        # without oldpeak
        df = df.filter(['age', 'sex', 'cp', 'thalach', 'exang', 'target'])
        # df = df.filter(['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang', 'target'])
        df = rename_df2(df)
    else:
        df = rename_df(df)

    # Convert categorical attributes.
    # Drop female, as we only need one. If user is not female, it is male. Value is binary.
    # Other is redundant and we don't want redundant variables.
    df = pd.get_dummies(df, drop_first=True)

    # Split into validation and training data
    x_train, x_test, y_train, y_test = split_data(df)

    print("Decision tree:")
    # compare MAE with different values of max_leaf_nodes
    # best_leaf_nodes_mae = None
    # best_leaf_nodes_rmse = None
    # min_mae = 10000000000000
    # min_rmse = 10000000000000
    # for max_leaf_nodes in range(2, 30):
    #     mae, rmse, acc = decision_tree(x_train, x_test, y_train, y_test, max_leaf_nodes)
    #     if mae < min_mae:
    #         min_mae = mae
    #         best_leaf_nodes_mae = max_leaf_nodes
    #     if rmse < min_rmse:
    #         min_rmse = rmse
    #         best_leaf_nodes_rmse = max_leaf_nodes
    # print(f"MAE:  {min_mae}   max leaf nodes: {best_leaf_nodes_mae}")
    # print(f"RMSE:  {min_rmse}  max leaf nodes: {best_leaf_nodes_rmse}")

    # Best decision tree is with 11 max_leaf_nodes
    decision_tree(x_train, x_test, y_train, y_test, 11)

    print("\nRandom forest:")
    random_forest(x_train, x_test, y_train, y_test)

    print("\nSVM:")
    svm(x_train, x_test, y_train, y_test)

    print("\n--------------\nDONE")


if __name__ == '__main__':
    main()
