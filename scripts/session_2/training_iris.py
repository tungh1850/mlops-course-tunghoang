from sklearn.datasets import load_iris  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import LogisticRegression

#export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
"""
setup sqlite server for mlflow
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    -h 127.0.0.1 \
    -p 5000
"""
mlflow.set_tracking_uri('http://127.0.0.1:5000')

def train() -> None:
    mlflow.set_experiment("My Server Experiment")
    iris = load_iris()
    iris = pd.DataFrame(np.c_[iris["data"], iris["target"]], columns=iris.feature_names + ["target"])
    iris.head()

    species = []

    for i in range(len(iris['target'])):
        if iris['target'][i] == 0:
            species.append("setosa")
        elif iris['target'][i] == 1:
            species.append('versicolor')
        else:
            species.append('virginica')


    iris['species'] = species

    # Droping the target and species since we only need the measurements
    X = iris.drop(['target','species'], axis=1)
    X_features = X.columns.to_list()
    # converting into numpy array and assigning petal length and petal width
    X = X.to_numpy()[:, (2,3)]
    y = iris['target']

    # Splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)
    
    log_reg = LogisticRegression()
    with mlflow.start_run(run_name="iris_training_2") as run:
        mlflow.log_param("random_state",42)
        log_reg.fit(X_train,y_train)
        # add logging for metrics
        mlflow.log_metric("train_accuracy",log_reg.score(X_train,y_train))
        mlflow.log_metric("test_accuracy",log_reg.score(X_test,y_test))
        mlflow.log_param("model_params",log_reg.get_params())
        mlflow.log_param("features",X_features)
        mlflow.log_param("target",y.name)
        mlflow.sklearn.log_model(log_reg, name="iris_model",registered_model_name="iris_model")

if __name__ == "__main__":
    train()