import seaborn as sns
import matplotlib.pyplot as plt

from constants import full_figure_width, figure_height, _xlabel_size, _ylabel_size, order, OUTCOME_DICT
from data import load_data_with_gender
from modelling import fit_and_test_best_model
from eli5.sklearn import PermutationImportance
import pandas as pd

import eli5

feature_names = ['NIHSS', 'Age', 'MRS0', 'MRS1', 'MRS2', 'MRS3', 'MRS4', 'MRS5', 'Gender']


def calculate_permutation_importances(model):
    obj = eli5.explain_weights(model, feature_names=feature_names)
    expl_df = eli5.formatters.format_as_dataframe(obj)

    return expl_df


def plot_explanation_for_prediction(model, x, feature_names=feature_names, ax=None):
    explanation = eli5.explain_prediction(model, x, feature_names=feature_names)
    expl = eli5.formatters.as_dataframe.format_as_dataframe(explanation)

    expl = expl.sort_values('feature')
    expl.rename({'feature': 'Feature', 'weight': 'Weight', 'outcome': 'Outcome'}, inplace=True, axis=1)

    expl['Outcome'] = expl['target'].map(OUTCOME_DICT)
    expl['include'] = expl['value'] != 0
    expl['include'][expl['Feature'] == 'Gender'] = True  # Include Gender as explanation even if value = 0
    g = sns.catplot(x="Feature", y="Weight", hue="Outcome",
                    data=expl.loc[expl.include & (expl.Feature.str.contains('<BIAS>') == False)],
                    height=6, kind="bar", ax=ax, hue_order=order)


def plot_prediction(model, x_input, ax=None):
    _pred = model.predict_proba(x_input.reshape(1, -1))
    df_prediction = pd.DataFrame(_pred.T*100, index=pd.Index(OUTCOME_DICT.values()), columns=['Probability'])
    df_prediction['Outcome'] = df_prediction.index
    sns.catplot(data=df_prediction, x='Outcome', y='Probability', order=order, kind='bar', ax=ax)


def plot_figure(model, expl_df, X):
    """expl_df: a df resulting from an eli5 explanation"""

    _plot_style = sns.axes_style()

    _plot_style['axes.spines.right'] = False
    _plot_style['axes.spines.top'] = False

    sns.set_style(_plot_style)
    fig, axes = plt.subplots(ncols=3, figsize=(full_figure_width, figure_height))
    plt.sca(axes[0])
    plt.barh(range(len(expl_df)), expl_df['weight'].values, xerr=expl_df['std'].values)
    # plt.errorbar(range(len(expl_df)), expl_df['weight'].values, expl_df['std'].values, fmt=' ', color='k')
    plt.yticks(range(len(expl_df)), expl_df.feature)
    plt.xlabel('Feature Importance', fontsize=_xlabel_size)
    plt.ylabel('Feature', fontsize=_ylabel_size)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plot_prediction(model, X[3, :], ax=axes[1])
    plt.sca(axes[1])
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Outcome', fontsize=_xlabel_size)
    plt.ylabel('% Probability', fontsize=_ylabel_size)

    plot_explanation_for_prediction(model, X[3, :], ax=axes[-1])
    plt.sca(axes[-1])
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Feature', fontsize=_xlabel_size)
    plt.ylabel('Weight', fontsize=_ylabel_size)

    plt.setp(axes[2].get_legend().get_texts(), fontsize='15')  # for legend text
    plt.setp(axes[2].get_legend().get_title(), fontsize='0')  # for legend title

    sns.despine()

    plt.tight_layout(h_pad=0.9)
    with open('../../figures/fig4.png', 'wb') as fileout:
        fig.savefig(fileout, bbox_inches='tight')


if __name__ == '__main__':
    _, best_model, X_test = fit_and_test_best_model()

    expl_df = calculate_permutation_importances(best_model)
    plot_figure(best_model, expl_df, X_test)