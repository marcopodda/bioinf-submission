import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


PALETTE = {
    "Standard RV": "#888888",
    "Feature-based": "#7781CE",
    "PSE-based": "#33A4AC"
}


def format_species(species):
    try:
        n1, n2 = species.split(" ")
        return f"{n1[0]}. {n2}"
    except:
        return species


def plot_scores(
    df,
    row=None,
    col="Species",
    metric="AUROC",
    hue="Method",
    height=4,
    aspect=1.0,
    ylim=None,
    sharey=False,
    sharex=False,
    methods=["Baseline", "Features", "PSEs"],
    filename="scores.png",
):
    df = df.copy()
    df["Dummy"] = "Dummy"
    df.Species = df.Species.map(format_species)

    g = sns.FacetGrid(
        df,
        row=row,
        col=col,
        sharex=sharex,
        sharey=sharey,
        despine=False,
        height=height,
        aspect=aspect,
    )

    g = g.map(
        sns.barplot,
        "Dummy",
        metric,
        hue,
        order=["Dummy"],
        hue_order=methods,  # type: ignore
        palette=PALETTE,
        # ci="sd",
    )

    g.set_titles(row_template="", col_template="{col_name}")
    g.set(xticks=[], ylim=ylim)
    g.set_axis_labels("", metric)
    g.add_legend(title=None)

    if filename:
        plt.savefig(filename)


def plot_cumulative_antigen_sum(
    df,
    hue="Method",
    row=None,
    col="Species",
    metric="Antigens Discovered",
    height=4,
    aspect=1.0,
    sharex=False,
    sharey=False,
    methods=["Baseline", "Ranking"],
    filename="antigen_counts.png",
):
    df = df.copy()
    df.Species = df.Species.map(format_species)

    g = sns.FacetGrid(
        df,
        col=col,
        col_wrap=5,
        height=height,
        aspect=aspect,
        sharex=sharex,
        sharey=sharey,
        despine=False,
        legend_out=False
    )

    g = g.map(
        sns.lineplot,
        "N. of in-vivo tests",
        metric,
        hue,
        hue_order=methods,
        # ci="sd",
        palette=PALETTE,
    )

    g.set_titles(template="{col_name}")

    axes = g.axes.ravel()
    for a in axes:
        a.yaxis.set_major_locator(MaxNLocator(integer=True))

    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2)
    g.fig.subplots_adjust(top=0.88, bottom=0.1)

    if filename:
        plt.savefig(filename)


def plot_box(
    df,
    row=None,
    col="Species",
    metric="Ranking Position",
    hue="Method",
    height=4,
    aspect=1.0,
    sharex=False,
    sharey=False,
    methods=["Baseline", "Ranking"],
    filename="boxplot.png",
):
    df = df.copy()
    df.Species = df.Species.map(format_species)

    g = sns.FacetGrid(
        df,
        row=row,
        col=col,
        height=height,
        aspect=aspect,
        sharex=sharex,
        sharey=sharey,
        despine=False,
    )

    g = g.map(
        sns.boxplot,
        metric,
        hue,
        order=methods,
        palette=PALETTE,
        ci="sd",
    )
    g.set_axis_labels(metric, "")

    if row is None:
        g.set_titles(template="{col_name}")
    else:
        g.set_titles(template="{col_name}  |  {row_name}")

    if filename:
        plt.savefig(filename)
