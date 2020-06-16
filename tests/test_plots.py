import matplotlib.pyplot as graph
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as some

from rosey_graph import plot_ecdf, plot_confusion_probability_matrix, plot_barplot, plot_forest


# If True then execution stops when the plots are drawn
def show_graph(block=False):
    graph.show(block=block)


@given(data=some.lists(
    some.floats(min_value=-1000, max_value=1000),
    min_size=3, max_size=10000
))
def test_plot_ecdf(data):
    data = np.array(data)
    plot_ecdf(data, {'label': 'hey'})
    show_graph()


def test_confusion_probability_matrix():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    x, y = load_breast_cancer(True)
    model = LogisticRegression()
    model.fit(x, y)

    plot_confusion_probability_matrix(
        y, model.predict(x), model.predict_proba(x)
    )
    show_graph()

    plot_confusion_probability_matrix(
        y, model.predict(x), model.predict_proba(x),
        labels=['Malignant', 'Benign']
    )
    show_graph()


@pytest.mark.parametrize('orient', ['h', 'v'])
def test_barplot(orient):
    plot_barplot(
        {'dogs': 10, 'cats': 4, 'birbs': 8},
        orient=orient,
    )
    show_graph()


def test_plot_forest():
    point_estimate = [1., 2., 3.]
    labels = ['one', 'two', 'three']

    # Test only point estimate
    plot_forest(point_estimate)
    show_graph()

    # Test Intervals
    lower_bound = [x-2 for x in point_estimate]
    upper_bound = [x+2 for x in point_estimate]
    plot_forest(point_estimate, lower_bound, upper_bound)
    show_graph()

    # Test labels
    plot_forest(point_estimate, lower_bound, upper_bound, labels)
    show_graph()
