import seaborn as sns
import matplotlib.pyplot as plt

from constants import full_figure_width, figure_height, _xlabel_size, _ylabel_size, order
from data import load_data_with_gender

sns.set(style="ticks", color_codes=True)


def plot_figure(df):
    _plot_style = sns.axes_style()

    _plot_style['axes.spines.right'] = False
    _plot_style['axes.spines.top'] = False

    f, axes = plt.subplots(1, 4, figsize=(full_figure_width, figure_height))
    sns.set_style(_plot_style)

    cp1 = sns.catplot(x="Outcome", y="NIHSS", kind="swarm", data=df, ax=axes[0], order=order)
    plt.sca(axes[0])
    plt.xlabel("Outcome", fontsize=_xlabel_size)
    plt.ylabel('NIHSS', fontsize=_ylabel_size)
    plt.tick_params(axis='both', which='major', labelsize=10)
    sns.despine()

    sns.catplot(x="Outcome", y="Age", kind="swarm", data=df, ax=axes[1], order=order)
    plt.sca(axes[1])
    plt.xlabel("Outcome", fontsize=_xlabel_size)
    plt.ylabel('Age', fontsize=_ylabel_size)
    plt.tick_params(axis='both', which='major', labelsize=10)
    sns.despine()

    sns.catplot(x="Outcome", y="MRS", kind="box", data=df, ax=axes[2], order=order)
    plt.sca(axes[2])
    plt.xlabel("Outcome", fontsize=_xlabel_size)
    plt.ylabel('MRS', fontsize=_ylabel_size)
    plt.tick_params(axis='both', which='major', labelsize=10)
    sns.despine()

    sns.countplot(x='Outcome', data=df, hue='Gender', ax=axes[3], order=order)
    plt.sca(axes[3])
    plt.xlabel("Outcome", fontsize=_xlabel_size)
    plt.ylabel('Count', fontsize=_ylabel_size)
    plt.tick_params(axis='both', which='major', labelsize=10)
    sns.despine()

    plt.tight_layout(h_pad=0.9)

    with open('../../figures/fig1.png', 'wb') as fileout:
        f.savefig(fileout)


if __name__ == '__main__':
    df = load_data_with_gender(
        new_data_path="../../../data/Discharge destinations including gender.xlsx",
        old_data_path="../../../data/Discharge destinations 2.xlsx",
    )

    plot_figure(df)
