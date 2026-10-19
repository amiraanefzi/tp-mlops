from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
 
from src.train import ( 
    build_pipeline, 
    load_config, 
    validate_dataset, 
) 
 
 
def test_dataset_is_valid(): 
    dataset = load_breast_cancer(as_frame=True) 
    validate_dataset(dataset.data, dataset.target) 
 
 
def test_pipeline_can_fit_and_predict(): 
    config = load_config() 
    dataset = load_breast_cancer(as_frame=True) 
 
    X_train, X_test, y_train, _ = train_test_split( 
        dataset.data, 
        dataset.target, 
        test_size=config["data"]["test_size"], 
        random_state=config["seed"], 
        stratify=dataset.target, 
    ) 
 
    pipeline = build_pipeline(config) 
    pipeline.fit(X_train, y_train) 
 
    predictions = pipeline.predict(X_test) 
    assert len(predictions) == len(X_test) 
    assert set(predictions).issubset(set(dataset.target.unique()))