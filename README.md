# Data Science and Machine Learning

This repository contains a small machine learning package and a collection of example notebooks for CMOR 438 / Data Science and Machine Learning coursework. The package implements common supervised and unsupervised learning algorithms, while the notebooks demonstrate how to apply those implementations to real datasets.

The project is organized so that the reusable source code lives in `src/`, the demonstrations live in `examples/`, and tests live in `tests/`. The goal is both instructional and practical: each algorithm can be inspected in code, tested independently, and then explored interactively in a notebook.

## Repository Structure

```text
.
+-- docs/                  Supporting documentation
+-- examples/              Jupyter notebooks and datasets by algorithm
+-- src/                   Python package source code
+-- tests/                 Unit tests for algorithms and utilities
+-- pyproject.toml         Package metadata and build configuration
+-- requirements.txt       Python dependencies
+-- LICENSE                MIT license
```

## Source Code

The package is located in `src/ml_package/` and is split into three main areas:

- `supervised_learning/`: implementations of linear regression, logistic regression, perceptron, neural networks, KNN, decision trees, random forests, and voting ensembles.
- `unsupervised_learning/`: implementations of PCA, K-Means clustering, and DBSCAN.
- `utils/`: preprocessing helpers, train/test splitting, cross-validation search utilities, and classification/regression metrics.

See [`src/README.md`](src/README.md) for a more detailed source-code map.

## Example Notebooks

The `examples/` directory contains notebook walkthroughs with local datasets. Each example folder includes an algorithm-specific `README.md` that goes into detail about each algorithm, an `.ipynb` notebook that demonstates how to analyze real data with these algorithms, and any data file needed for the demo.

Supervised learning examples include:

- Linear Regression
- Logistic Regression
- K-Nearest Neighbors
- Decision Trees
- Random Forests
- Perceptron
- Neural Networks
- Voting Models

Unsupervised learning examples include:

- Principal Component Analysis
- K-Means Clustering
- DBSCAN

See [`examples/README.md`](examples/README.md) for a full notebook guide.

## Getting Started

Create and activate a virtual environment, then install the package dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e .
```

After installation, the package can be imported as `ml_package`:

```python
from ml_package import KNN, LinearRegression, PCA
```

To explore the demonstrations, open any notebook in `examples/` with Jupyter or VS Code.

## Running Tests

The test suite is organized to match the package structure. Run all tests from the repository root with:

```bash
pytest
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
