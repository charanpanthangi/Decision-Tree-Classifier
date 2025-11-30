from app.data import load_data
from app.model import build_model, train_model
from app.preprocess import train_test_split_data


def test_model_training_and_prediction():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.3)
    model = build_model(max_depth=3)
    trained_model = train_model(model, X_train, y_train)
    preds = trained_model.predict(X_test)
    assert len(preds) == len(y_test)
