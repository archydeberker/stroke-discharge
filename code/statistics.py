import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import constants


def covariate_correlations_segregated_by_outcome(df, covariate1, covariate2, outcome):
    r, p = stats.pearsonr(df.loc[df['Outcome'] == outcome, covariate1].values,
                          df.loc[df['Outcome'] == outcome, covariate2].values)

    return dict(r=r, p=p)


def compute_kruskal(df, factor):
    F, p = stats.kruskal(*[df.loc[df['Outcome'] == outcome, factor] for outcome in constants.OUTCOMES])
    return F, p


def compute_pairwise_p(df, factor, outcomes=constants.OUTCOMES):
    p_mtx = np.zeros((len(outcomes), len(outcomes)))
    for i, o1 in enumerate(outcomes):
        for j, o2 in enumerate(outcomes):
            _, p_mtx[i, j] = stats.ttest_ind(df.loc[df['Outcome'] == o1, factor],
                                             df.loc[df['Outcome'] == o2, factor])

    return p_mtx, outcomes


def univariate_stats(df, factor, plot=False):
    F, p = compute_kruskal(df, factor)

    print(f"{factor}, p={p} from Kruskal Wallace")
    p, outcomes = compute_pairwise_p(df, factor)
    if plot:
        mask = np.zeros_like(p)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(p, xticklabels=outcomes, yticklabels=outcomes, mask=mask, annot=True, linewidths=.5)

        plt.title(factor)
        plt.show()

    return p