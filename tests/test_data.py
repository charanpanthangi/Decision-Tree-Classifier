from app.data import load_data


def test_load_data_shape():
    X, y = load_data()
    assert not X.empty
    assert len(X) == len(y)
    assert X.shape[1] == 4  # four features in iris dataset
