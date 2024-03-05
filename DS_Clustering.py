import matplotlib.pyplot as plt

# Plotting Function -----------------------------------------------------------------------
def clustering_plot(X, labels=[], centers=[], title=None, figsize='auto', alpha=1,
                    xscale='linear', yscale='linear'):
    import itertools

    combination_set = list(itertools.combinations(X.columns,2))
    len_comb = len(combination_set)

    if len_comb == 1:
        dim = 1;        nrows = 1;        ncols = 1;
    elif len_comb == 2:
        dim = 1;        nrows = 1;        ncols = 2;
    elif len_comb == 3:
        dim = 1;        nrows = 1;        ncols = 3;
    elif len_comb % 3 == 0:
        dim = 2;        nrows = len_comb // 3;        ncols = 3;
    elif len_comb % 2 == 0:
        dim = 2;        nrows = len_comb // 2;        ncols = 2;

    # print(dim, nrows, ncols)
    if list(labels):        # label
        unique_label = np.unique(labels)

    if list(centers):   #centers:
        center_frame = pd.DataFrame(centers, columns=X.columns)

    if figsize == 'auto':
        figsize = (ncols*5, nrows*4)
    fig, axe = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ci, (x1, x2) in enumerate(combination_set):
        # print(ci, x1, x2)
        if nrows==1 and ncols==1:
            if list(labels):
                for l in unique_label:
                    cluster_mask = (labels == l)
                    X_mask = X.iloc[cluster_mask, :]
                    axe.scatter(X_mask[x1], X_mask[x2], label=str(l), edgecolors='white', alpha=alpha)
                axe.legend()
            else:
                axe.scatter(X[x1], X[x2], c='skyblue', edgecolors='white', alpha=alpha)
            axe.set_xlabel(x1)
            axe.set_ylabel(x2)
            axe.set_xscale(xscale)
            axe.set_yscale(yscale)

            if list(centers):
                axe.scatter(center_frame[x1], center_frame[x2], marker='*', c='r', s=200, edgecolors='white')

        elif dim == 1:
            if list(labels):
                for l in unique_label:
                    cluster_mask = (labels == l)
                    X_mask = X.iloc[cluster_mask, :]
                    axe[ci].scatter(X_mask[x1], X_mask[x2], label=str(l), edgecolors='white', alpha=alpha)
                axe[ci].legend()
            else:
                axe[ci].scatter(X[x1], X[x2], c='skyblue', edgecolors='white', alpha=alpha)
            axe[ci].set_xlabel(x1)
            axe[ci].set_ylabel(x2)
            axe[ci].set_xscale(xscale)
            axe[ci].set_yscale(yscale)
            if list(centers):
                axe[ci].scatter(center_frame[x1], center_frame[x2], marker='*', c='r', s=200, edgecolors='white')

        else: # dim == 2:
            if list(labels):
                for l in unique_label:
                    cluster_mask = (labels == l)
                    X_mask = X.iloc[cluster_mask, :]
                    axe[ci//ncols-1][ci%ncols].scatter(X_mask[x1], X_mask[x2], label=str(l), edgecolors='white', alpha=alpha)
                axe[ci//ncols-1][ci%ncols].legend()
            else:
                axe[ci//ncols-1][ci%ncols].scatter(X[x1], X[x2], c='skyblue', edgecolors='white', alpha=alpha)
            axe[ci//ncols-1][ci%ncols].set_xlabel(x1)
            axe[ci//ncols-1][ci%ncols].set_ylabel(x2)
            axe[ci//ncols-1][ci%ncols].set_xscale(xscale)
            axe[ci//ncols-1][ci%ncols].set_yscale(yscale)

            if list(centers):
                axe[ci//ncols-1][ci%ncols].scatter(center_frame[x1], center_frame[x2], marker='*', c='r', s=200, edgecolors='white')

    if title:
        fig.suptitle(title, fontsize=15)
    plt.show()
# -----------------------------------------------------------------------------------------------------
