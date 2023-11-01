__version__ = '1.2023.11.01'  # Major.YYYY.MM.DD

import matplotlib.pyplot as graph
import numpy as np

colors_538 = [
    '#30a2da',
    '#fc4f30',
    '#e5ae38',
    '#6d904f',
    '#8b8b8b',
]


def plot_roc_curve(
        prediction_probability,
        true,
        label='',
        plot_curve_only=False,
        estimate_intervals=False,
        show_graph=False
):
    """
    ROC Curves

    :param prediction_probability:
    :param true:
    :param label:
    :param plot_curve_only:
    :param estimate_intervals: estimates the 95% interval for the AUC
    :param show_graph:
    :return:
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.utils import resample

    fpr, tpr, thres = roc_curve(true, prediction_probability)
    auc = roc_auc_score(true, prediction_probability)

    plot_label = label if label == '' else label + ' '
    plot_label = f'{plot_label}AUC = {auc:.5f}'
    if estimate_intervals:
        auc_dist = []
        for seed in range(1000):
            true_resample, prediction_probability_resample = resample(true, prediction_probability)
            auc_dist.append(roc_auc_score(true_resample, prediction_probability_resample))
        auc_dist = np.array(auc_dist)
        plot_label += f' ({np.percentile(auc_dist, 2.5):.5f}, {np.percentile(auc_dist, 97.5):.5f})'

    graph.plot(fpr, tpr, label=plot_label)
    if not plot_curve_only:
        graph.plot([0, 1], [0, 1], linestyle='--', color='k', label='Guessing')
    graph.xlim([0, 1])
    graph.ylim([0, 1])
    graph.legend(loc=0)
    graph.xlabel('False Positive Rate')
    graph.ylabel('True Positive Rate')

    if show_graph:
        graph.show()

    return {'fpr': fpr, 'tpr': tpr, 'threshold': thres}


def _plot_pr(
        y_pred_proba,
        y_true,
        is_precision_plot: bool,
        class_labels,
        estimate_intervals: bool,
        show_graph
):
    import itertools
    from sklearn.metrics import precision_recall_curve
    from sklearn.utils import resample

    for class_i, color in zip(range(y_pred_proba.shape[1]), itertools.cycle(colors_538)):
        precision, recall, threshold = precision_recall_curve(
            y_true=y_true,
            probas_pred=y_pred_proba[:, class_i]
        )

        graph.plot(
            threshold,
            precision[:-1] if is_precision_plot else recall[:-1],
            linewidth=2,
            color=color,
            label=f'{class_i}' if class_labels is None else f'{class_labels[class_i]}'
        )

        if estimate_intervals:
            for seed in range(1000):
                y_true_resample, y_pred_proba_resample = resample(y_true, y_pred_proba, random_state=seed)

                precision_i, recall_i, threshold_i = precision_recall_curve(
                    y_true=y_true_resample,
                    probas_pred=y_pred_proba_resample[:, class_i]
                )

                graph.plot(
                    threshold_i,
                    precision_i[:-1] if is_precision_plot else recall_i[:-1],
                    linewidth=0.5,
                    color=color,
                    alpha=0.1
                )

        graph.xlabel('Probability Threshold')
        graph.ylabel('Precision' if is_precision_plot else 'Recall')
        graph.legend()

        if show_graph:
            graph.show()


def plot_precision(prediction_probability, true, class_labels=None, estimate_intervals=False, show_graph=False):
    """
    This plots the precision | probability

    Reminder that precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> import matplotlib.pyplot as graph
    >>> x, y = load_breast_cancer(return_X_y=True)
    >>> model = LogisticRegression().fit(x, y)
    >>> ypp = model.predict_proba(x)
    >>> plot_precision(ypp, y)
    >>> graph.show()
    >>> plot_precision(ypp, y, class_labels=['Malignant', 'Benign'])
    >>> graph.show()
    >>> plot_precision(ypp, y, estimate_intervals=True, class_labels=['Malignant', 'Benign'])
    >>> graph.show()

    :return:
    """
    _plot_pr(
        y_pred_proba=prediction_probability,
        y_true=true,
        is_precision_plot=True,
        class_labels=class_labels,
        estimate_intervals=estimate_intervals,
        show_graph=show_graph
    )


def plot_recall(prediction_probability, true, class_labels=None, estimate_intervals=False, show_graph=False):
    """
    Plots the recall | probability

    Reminder that recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> import matplotlib.pyplot as graph
    >>> x, y = load_breast_cancer(return_X_y=True)
    >>> model = LogisticRegression().fit(x, y)
    >>> ypp = model.predict_proba(x)
    >>> plot_recall(ypp, y)
    >>> graph.show()
    >>> plot_recall(ypp, y, class_labels=['Malignant', 'Benign'])
    >>> graph.show()
    >>> plot_recall(ypp, y, estimate_intervals=True, class_labels=['Malignant', 'Benign'])
    >>> graph.show()

    :return:
    """
    _plot_pr(
        y_pred_proba=prediction_probability,
        y_true=true,
        is_precision_plot=False,
        class_labels=class_labels,
        estimate_intervals=estimate_intervals,
        show_graph=show_graph
    )


def plot_biplot(pca, x_axis=0, y_axis=1, data=None, feature_names=None, c=None, show_graph=False):
    """
    Plots the kind of biplot R creates for PCA

    :param pca: SKLearn PCA object
    :param c: Matplotlib scatterplot c param for coloring by class
    :param x_axis:
    :param y_axis:
    :param feature_names:
    :param data: X data you want to see in the biplot
    :param show_graph:
    :return:
    """
    x_axis_upscale_coef, y_axis_upscale_coef = 1, 1

    if data is not None:
        data = pca.transform(data)
        pc_0, pc_1 = data[:, x_axis], data[:, y_axis]
        x_axis_upscale_coef = pc_0.max() - pc_0.min()
        y_axis_upscale_coef = pc_1.max() - pc_1.min()

        graph.scatter(pc_0, pc_1, c=c, alpha=0.66)

    projected_rotation = pca.components_[[x_axis, y_axis], :]
    projected_rotation = projected_rotation.T

    for i_feature in range(projected_rotation.shape[0]):
        graph.scatter(
            [0, projected_rotation[i_feature, x_axis] * x_axis_upscale_coef * 1.2],
            [0, projected_rotation[i_feature, y_axis] * y_axis_upscale_coef * 1.2],
            alpha=0
        )

        graph.arrow(
            0,
            0,
            projected_rotation[i_feature, x_axis] * x_axis_upscale_coef,
            projected_rotation[i_feature, y_axis] * y_axis_upscale_coef,
            alpha=0.7
        )
        graph.text(
            projected_rotation[i_feature, x_axis] * 1.15 * x_axis_upscale_coef,
            projected_rotation[i_feature, y_axis] * 1.15 * y_axis_upscale_coef,
            f'col {i_feature}' if feature_names is None else feature_names[i_feature],
            ha='center', va='center'
        )

    graph.xlabel(f'PC {x_axis}')
    graph.ylabel(f'PC {y_axis}')

    if show_graph:
        graph.show()


def plot_learning_curve(means, stds, xs=None, n=None, show_graph=False):
    """
    Plot learning curve with confidence intervals

    :param xs: What the units on the x-axis should be
    :param n: sample size, usually the number of CV intervals
    :param means:
    :param stds:
    :param show_graph:
    :return:
    """
    import numpy as np
    xs = xs if xs is not None else np.arange(len(means))

    # If N is given, compute the standard error
    stds = stds / np.sqrt(n) if n is not None else stds
    ci95 = stds * 1.96

    graph.plot(xs, means)
    graph.fill_between(
        xs,
        means - ci95, means + ci95,
        alpha=0.4
    )
    if show_graph:
        graph.show()


def plot_confusion_matrix(y_true, y_pred, labels: list = None, axis=1, show_graph=False):
    """
    Normalised Confusion Matrix

    :param y_true:
    :param y_pred:
    :param labels:
    :param axis: 0 if you want to know the probabilities given a predication. 1 if you want to know class confusion.
    :param show_graph:
    :return:
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    labels = True if labels is None else labels

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=axis, keepdims=True)

    sns.heatmap(
        cm,
        annot=True, square=True, cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    graph.xlabel('Predicted')
    graph.ylabel('True')

    if show_graph:
        graph.show()


def plot_ecdf(x, plot_kwargs=None, show_graph=False):
    """
    Create the plot of the empricial distribution function

    >>> import numpy as np
    >>> import matplotlib.pyplot as graph
    >>> plot_ecdf(np.random.normal(100, 15, size=100), {'label': 'blah'})
    >>> graph.show()

    :param x:
    :param plot_kwargs:
    :param show_graph:
    :return:
    """

    def _ecdf(data):
        """
        Empirical CDF (x, y) generator
        """
        import numpy as np
        x = np.sort(data)
        cdf = np.linspace(0, 1, len(x))
        return cdf, x

    cdf, x = _ecdf(x)
    plot_kwargs = dict() if plot_kwargs is None else plot_kwargs

    graph.plot(x, cdf, **plot_kwargs)

    if show_graph:
        graph.show()


def plot_confusion_probability_matrix(
        y_true, y_pred, y_pred_proba,
        labels: list = None, figsize=(8, 8), rug_height=0.05, show_graph=False
):
    """
    Confusion matrix where you can see the histogram of the

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> import matplotlib.pyplot as graph
    >>> x, y = load_breast_cancer(return_X_y=True)
    >>> model = LogisticRegression().fit(x, y)
    >>> ypp = model.predict_proba(x)[:, 1]
    >>> plot_confusion_probability_matrix(y, model.predict(x), model.predict_proba(x))
    >>> graph.show()
    >>> plot_confusion_probability_matrix(y, model.predict(x), model.predict_proba(x), labels=['Malignant', 'Benign'])
    >>> graph.show()

    :param y_true:
    :param y_pred:
    :param y_pred_proba:
    :param labels:
    :param figsize:
    :param rug_height:
    :param show_graph:
    :return:
    """
    import numpy as np
    from itertools import product
    from sklearn.metrics import confusion_matrix

    def solve_n_bins(x):
        """
        Uses the Freedman Diaconis Rule for generating the number of bins required
        https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
        Bin Size = 2 IQR(x) / (n)^(1/3)
        """
        import numpy as np
        from scipy.stats import iqr

        x = np.asarray(x)
        hat = 2 * iqr(x) / (len(x) ** (1 / 3))

        if hat == 0:
            return int(np.sqrt(len(x)))
        else:
            return int(np.ceil((x.max() - x.min()) / hat))

    n_classes = y_pred_proba.shape[1]
    labels = list(range(n_classes)) if labels is None else labels
    cm = confusion_matrix(y_true, y_pred)

    # Create subplots
    figure, box = graph.subplots(n_classes, n_classes, sharex='all', figsize=figsize)

    # Create histograms
    for i, j in product(range(n_classes), range(n_classes)):
        selection_mask = (y_true == i) & (y_pred == j)
        assert selection_mask.sum() == cm[i, j]

        subset_probabilities = y_pred_proba[selection_mask, i]
        box[i, j].set_title(f'N: {cm[i, j]}')
        box[i, j].hist(subset_probabilities, density=True, bins=solve_n_bins(subset_probabilities), alpha=0.7)
        box[i, j].plot(subset_probabilities, np.ones(len(subset_probabilities)) * rug_height, '|', alpha=0.7)
        box[i, j].set_yticks([])

    # Axis labels
    for k in range(n_classes):
        box[-1, k].set_xlabel(f'Pred = {labels[k]}')
        box[k, 0].set_ylabel(f'True = {labels[k]}')

    if show_graph:
        graph.show()


def plot_barplot(d: dict, orient: str = 'h', show_graph=False):
    """
    Create bar plot from a dictionary that maps a name to the size of the bar

    >>> dictionary = {'dogs': 10, 'cats': 4, 'birbs': 8}
    >>> plot_barplot(dictionary, show_graph=True)

    :param d:
    :param orient:
    :param show_graph:
    :return:
    """
    orient = orient.lower()

    if 'h' not in orient and 'v' not in orient:
        raise ValueError('`orient` must be either `h` for horizontal and `v` for vertical')

    bar_plot = graph.barh if 'h' in orient else graph.bar
    bar_plot(
        range(len(d.values())),
        list(d.values()),
        tick_label=list(d.keys())
    )

    if show_graph:
        graph.show()


def plot_2d_histogram(x, y, bins=100, transform=lambda z: z, plot_kwargs: dict = None, show_graph=False):
    """
    Creates a 2D histogram AND allows you to transform the colors of the histogram with the transform function

    Datashader like functionality without all the hassle
    :param x:
    :param y:
    :param bins:
    :param transform: function that takes 1 argument used to transform the histogram
    :param plot_kwargs: arguments to pass to the internal imshow()
    :param show_graph:
    :return:
    """
    import numpy as np

    required_kwargs = {'aspect': 'auto'}
    if plot_kwargs is None:
        plot_kwargs = required_kwargs
    else:
        plot_kwargs.update(required_kwargs)

    h, *_ = np.histogram2d(x, y, bins=bins)
    h = np.rot90(h)

    graph.imshow(transform(h), **plot_kwargs)
    if show_graph:
        graph.show()


def plot_forest(point_estimate, lower_bound=None, upper_bound=None, labels=None, show_graph=False):
    """
    Create forest plot using summary data.

    :param point_estimate: Where the center of the point should be located
    :param lower_bound:
    :param upper_bound:
    :param labels:
    :param show_graph:
    :return:
    """
    # Validate input
    import pandas as pd
    from operator import xor
    if xor(bool(lower_bound), bool(upper_bound)):
        raise ValueError('You must supply both an `upper_bound` and `lower_bound`')

    if not all((len(x) for x in (point_estimate, lower_bound, upper_bound, labels) if x is not None)):
        raise AssertionError('All inputs must be the same lengths')

    # Setup
    indices = list(range(len(point_estimate)))
    labels = labels.values if isinstance(labels, pd.Series) else labels

    # Plot
    graph.plot(point_estimate, indices, 'D', markersize=10, color='seagreen')

    if lower_bound and upper_bound:
        graph.hlines(indices, xmin=lower_bound, xmax=upper_bound, colors='seagreen')

    graph.yticks(indices, labels)

    if show_graph:
        graph.show()


def plot_decision_boundary(model, x, dim_indices=(0, 1), extrapolation=1.2, show_graph=False):
    """
    Create a 2 dimensional decision boundary plot across any 2 dimensions chosen

    :param model: A model with .predict_proba() method implemented
    :param x: 
    :param dim_indices: The indices of the dimensions you want to plot
    :param extrapolation: The ratio of the x and y axis to extrapolate/extend
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError('Model must have a `predict_proba()` method implemented')

    x_min, x_max = x[:, dim_indices[0]].min() * extrapolation, x[:, dim_indices[0]].max() * extrapolation
    y_min, y_max = x[:, dim_indices[1]].min() * extrapolation, x[:, dim_indices[1]].max() * extrapolation

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    pred = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = pred[:, 1]
    pred = pred.reshape(xx.shape)

    graph.contourf(xx, yy, pred, alpha=0.4)

    if show_graph:
        graph.show()


if __name__ == '__main__':
    import doctest

    doctest.testmod()
