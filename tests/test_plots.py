import pytest
import matplotlib.pyplot as graph
from rosey_graph import plot_forest

# If True then execution stops when the plots are drawn
show_plot_kwargs = {'block': True}


def test_plot_forest():
    point_estimate = [1., 2., 3.]
    labels = ['one', 'two', 'three']

    # Test only point estimate
    plot_forest(point_estimate)
    graph.show(**show_plot_kwargs)

    # Test Intervals
    lower_bound = [x-2 for x in point_estimate]
    upper_bound = [x+2 for x in point_estimate]
    plot_forest(point_estimate, lower_bound, upper_bound)
    graph.show(**show_plot_kwargs)

    # Test labels
    plot_forest(point_estimate, lower_bound, upper_bound, labels)
    graph.show(**show_plot_kwargs)
