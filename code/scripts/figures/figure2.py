import seaborn as sns
import matplotlib.pyplot as plt

from constants import full_figure_width, figure_height, _xlabel_size, _ylabel_size, order
from data import load_data_with_gender


def plot_figure_2a(df):
    _plot_style = sns.axes_style()

    _plot_style['axes.spines.right'] = False
    _plot_style['axes.spines.top'] = False

    f, axes = plt.subplots(1, 3, figsize=(full_figure_width, figure_height))
    sns.set_style(_plot_style)

    sns.catplot(x="MRS", y="NIHSS", kind="swarm", data=df, ax=axes[0], palette='Blues')
    plt.sca(axes[0])
    plt.xlabel("MRS",fontsize=_xlabel_size)
    plt.ylabel('NIHSS', fontsize=_ylabel_size)
    plt.tick_params(axis='both', which='major', labelsize=10)
    sns.despine()

    sns.regplot(x="NIHSS", y="Age", data=df, ax=axes[1], line_kws={"color": "black"}, scatter_kws={'alpha':0.3})
    plt.sca(axes[1])
    plt.xlabel("Age",fontsize=_xlabel_size)
    plt.ylabel('NIHSS', fontsize=_ylabel_size)
    plt.tick_params(axis='both', which='major', labelsize=10)
    sns.despine()

    sns.catplot(x="MRS", y="Age", kind="swarm", data=df, ax=axes[2], palette='Blues')
    plt.sca(axes[2])
    plt.xlabel("MRS",fontsize=_xlabel_size)
    plt.ylabel('Age', fontsize=_ylabel_size)
    plt.tick_params(axis='both', which='major', labelsize=10)
    sns.despine()

    with open('../../figures/fig2a.png', 'wb') as fileout:
        f.savefig(fileout, bbox_inches='tight')


def plot_figure_2b(df):
    g = sns.FacetGrid(df, col="Outcome", hue="Outcome", col_order=order)
    r_plots = g.map(sns.regplot, "MRS", 'NIHSS')

    for ax in r_plots.axes[:, 0]:
        ax.set_xlim((-.5, 5.5))

    with open('../../figures/fig2bi.png', 'wb') as fileout:
        g.fig.savefig(fileout, bbox_inches='tight')

    g = sns.FacetGrid(df, col="Outcome", hue="Outcome", col_order=order)
    r_plots = g.map(sns.regplot, "Age", 'NIHSS')

    with open('../../figures/fig2bii.png', 'wb') as fileout:
        g.fig.savefig(fileout, bbox_inches='tight')



if __name__ == '__main__':
    df = load_data_with_gender(
        new_data_path="../../../data/Discharge destinations including gender.xlsx",
        old_data_path="../../../data/Discharge destinations 2.xlsx",
    )

    plot_figure_2a(df)
    plot_figure_2b(df)
