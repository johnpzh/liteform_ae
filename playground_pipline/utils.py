import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import dump, load
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import dgl
from ogb.nodeproppred import DglNodePropPredDataset


estimators = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNeighbors": KNeighborsClassifier(),
    "Linear_SVM": SVC(kernel="linear", random_state=42),
    "RBF_SVM": SVC(gamma=2, C=1, random_state=42),
    "GaussianProcess": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NeuralNet": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=42),
    "NaiveBayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
}


def get_file_name(feature_file: str):
    return os.path.splitext(os.path.basename(feature_file))[0]


def get_X_and_y_v0_density(feature_file: str):
    """get features X and truth y. Use density nnz per row as the feature set

    Args:
        feature_file (str): features csv file

    Returns:
        2d-array, 1d-array: features X, truth y
    """
    df = pd.read_csv(feature_file)
    features = ['num_rows', 
                'num_cols',
                'nnz',
                'avg_nnz_density_per_row',
                'min_nnz_density_per_row',
                'max_nnz_density_per_row',
                'std_dev_nnz_density_per_row']
    X = df[features].to_numpy()

    speedup = df[['speed_to_naive']].to_numpy()
    y = [ 1 if x >= 1.1 else 0 for x in speedup ]

    # # test
    # print(F"X: {X}")
    # print(F"y: {y}")
    # # end test

    return X, y

def get_X_and_y_v1_value(feature_file: str):
    """get features X and truth y. Use value nnz per row as the feature set

    Args:
        feature_file (str): features csv file

    Returns:
        2d-array, 1d-array: features X, truth y
    """
    df = pd.read_csv(feature_file)
    features = ['num_rows', 
                'num_cols',
                'nnz',
                'avg_nnz_per_row',
                'min_nnz_per_row',
                'max_nnz_per_row',
                'std_dev_nnz_per_row']
    X = df[features].to_numpy()

    speedup = df[['speed_to_naive']].to_numpy()
    y = [ 1 if x >= 1.1 else 0 for x in speedup ]

    # # test
    # print(F"X: {X}")
    # print(F"y: {y}")
    # # end test

    return X, y


def get_X_value_and_y_num_parts(feature_file: str):
    """get features X and truth y. Use value nnz per row as the feature set

    Args:
        feature_file (str): features csv file

    Returns:
        2d-array, 1d-array: features X, truth y
    """
    df = pd.read_csv(feature_file)
    features = ['num_rows', 
                'num_cols',
                'nnz',
                'avg_nnz_density_per_row',
                'min_nnz_density_per_row',
                'max_nnz_density_per_row',
                'std_dev_nnz_density_per_row',
                'K']
    X = df[features].to_numpy()

    # speedup = df[['speed_to_naive']].to_numpy()
    # y = [ 1 if x >= 1.1 else 0 for x in speedup ]
    y = df[['best_num_partitions']].to_numpy().reshape(-1)

    # # test
    # print(F"X: {X}")
    # print(F"y: {y}")
    # # end test

    return X, y


def get_X_value_and_y_max_bucket_size(feature_file: str):
    """get features X and truth y. Use value nnz per row as the feature set

    Args:
        feature_file (str): features csv file

    Returns:
        2d-array, 1d-array: features X, truth y
    """
    df = pd.read_csv(feature_file)
    features = ['num_rows', 
                'num_cols',
                'nnz',
                'avg_nnz_per_row',
                'min_nnz_per_row',
                'max_nnz_per_row',
                'std_dev_nnz_per_row']
    X = df[features].to_numpy()

    y = df[['best_max_bucket_width']].to_numpy().reshape(-1)

    # # test
    # print(F"X: {X}")
    # print(F"y: {y}")
    # # end test

    return X, y


def evaluate(truth,
             predict):
    stat = {
        "accuracy": [accuracy_score(y_true=truth, y_pred=predict)],
        "precision": [precision_score(y_true=truth, y_pred=predict)],
        "recall": [recall_score(y_true=truth, y_pred=predict)],
        "f1": [f1_score(y_true=truth, y_pred=predict)]
    }

    return stat


def evaluate_into_stat(truth: list,
                       predict: list,
                       stat: dict):
    stat["accuracy"].append(accuracy_score(y_true=truth, y_pred=predict))
    stat["precision"].append(precision_score(y_true=truth, y_pred=predict, average='micro'))
    stat["recall"].append(recall_score(y_true=truth, y_pred=predict, average='micro'))
    stat["f1"].append(f1_score(y_true=truth, y_pred=predict, average='micro'))


def save_ref_perf(ref_perf: dict,
                  feature_file: str,
                  output_dir: str,
                  desc: str):
    basename = get_file_name(feature_file)
    ref_file = os.path.join(output_dir, F"{basename}.{desc}.csv")
    df = pd.DataFrame(data=ref_perf)
    df.to_csv(ref_file, index=False)
    print(df)


def save_model(classifier,
               feature_file: str,
               output_dir: str,
               desc: str):
    basename = get_file_name(feature_file)
    model_file = os.path.join(output_dir, F"{basename}.{desc}.joblib")
    dump(classifier, model_file)
    print(F"Saved model {classifier} to {model_file} .")


def save_train_perf(train_perf: dict,
                    feature_file: str,
                    output_dir: str,
                    desc: str):
    basename = get_file_name(feature_file)
    train_file = os.path.join(output_dir, F"{basename}.{desc}.csv")
    df = pd.DataFrame(data=train_perf)
    df.to_csv(train_file, index=False)
    print(df)


def get_dataset(dataset_name: str):
    """Return DGLGraph object from dataset name.

    Parameters
    ----------
    dataset_name: str
        The dataset name.

    Returns
    -------
    DGLGraph
        The graph object.
    """
    home = os.path.expanduser("~")
    ogb_path = os.path.join(home, "ogb")
    if not os.path.exists(ogb_path):
        os.makedirs(ogb_path)
    if dataset_name == "pubmed":
        pubmed = dgl.data.PubmedGraphDataset()
        return pubmed[0].int()
    elif dataset_name == "cora":
        cora = dgl.data.CoraGraphDataset()
        return cora[0].int()
    elif dataset_name == "citeseer":
        citeseer = dgl.data.CiteseerGraphDataset()
        return citeseer[0].int()
    elif dataset_name == "arxiv":
        arxiv = DglNodePropPredDataset(name="ogbn-arxiv", root=ogb_path)
        return arxiv[0][0].int()
    elif dataset_name == "proteins":
        proteins = DglNodePropPredDataset(name="ogbn-proteins", root=ogb_path)
        return proteins[0][0].int()
    elif dataset_name == "products":
        products = DglNodePropPredDataset(name="ogbn-products", root=ogb_path)
        return products[0][0].int()
    elif dataset_name == "ppi":
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi).int()
        return g
    elif dataset_name == "reddit":
        reddit = dgl.data.RedditDataset()
        return reddit[0].int()
    else:
        raise KeyError("Unknown dataset: {}".format(dataset_name))

