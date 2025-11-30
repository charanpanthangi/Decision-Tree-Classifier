# Decision Tree Classifier Template

A beginner-friendly yet production-ready template for training and evaluating a CART-style Decision Tree classifier using the classic scikit-learn Iris dataset.

## What is a Decision Tree?
A decision tree splits the feature space into simple decision rules. At each node, the algorithm selects the feature and threshold that best separates the classes using an impurity measure such as **Gini impurity** (default) or **entropy** (information gain). Splitting continues until stopping criteria are met.

### Why trees are easy to use
- **Interpretability:** you can trace predictions by following branches.
- **Minimal preprocessing:** trees are scale-invariant, so feature scaling is optional.
- **Handles non-linear boundaries:** recursive splits capture complex patterns.

### When trees can struggle
- **Overfitting:** very deep trees memorize training data.
- **Instability:** small data changes can alter splits.
- **Limited extrapolation:** trees partition the observed space only.

## Key Hyperparameters
- `criterion`: `"gini"` (default) or `"entropy"`; controls how splits are chosen.
- `max_depth`: maximum depth of the tree; smaller values reduce overfitting.
- `min_samples_split`: minimum samples required to split an internal node; larger values make the tree more conservative.
- `min_samples_leaf`: minimum samples required at a leaf node; useful for smoothing and reducing overfitting.
- `random_state`: ensures reproducible splits.

## Dataset
- **Iris** dataset from scikit-learn (`load_iris`).
- 150 samples, 4 numeric features, 3 target classes.

## Project Structure
```
app/
  data.py          # load data
  preprocess.py    # train/test split (no scaling needed)
  model.py         # DecisionTreeClassifier with sane defaults
  evaluate.py      # accuracy, precision, recall, F1, confusion matrix
  visualize.py     # confusion matrix, feature importance, PCA scatter
  main.py          # orchestrates pipeline
notebooks/
  demo_decision_tree_classifier.ipynb
examples/
  README_examples.md
requirements.txt
Dockerfile
LICENSE
README.md
```

## Pipeline Overview
1. **Load** the Iris dataset (`app/data.py`).
2. **Split** into training/testing (`app/preprocess.py`).
3. **Train** a DecisionTreeClassifier (`app/model.py`).
4. **Evaluate** with accuracy, precision, recall, F1, and confusion matrix (`app/evaluate.py`).
5. **Visualize** confusion matrix, feature importance, and PCA scatter (`app/visualize.py`).

## How to Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```
Figures are saved as SVGs in the `figures/` folder.

## Jupyter Notebook Demo
Launch Jupyter and open `notebooks/demo_decision_tree_classifier.ipynb` for an interactive walkthrough including tree intuition, impurity measures, overfitting demonstration, feature importance, and evaluation visuals.

## Docker Usage
```bash
docker build -t decision-tree-classifier .
docker run --rm decision-tree-classifier
```

## Future Improvements
- Cost-complexity pruning examples.
- Automated hyperparameter tuning (GridSearchCV/RandomizedSearchCV).
- Comparisons with ensemble methods like Random Forests and XGBoost.

## License
MIT License. See [LICENSE](LICENSE).
