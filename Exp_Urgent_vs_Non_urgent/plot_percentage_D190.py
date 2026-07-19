import glob
import matplotlib.pyplot as plt

fontsize = 28

base = dict(
    linewidth=2,
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=1,
)

styles = [
    ("s", "-"),
    ("o", ":"),
    ("^", "--"),
    ("D", "-."),
    ("v", (0, (3, 1, 1, 1))),
]

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
]

for filepath in sorted(
    glob.glob(
        "Jobs_Served_Multiple_Job_Counts_190.txt"
    )
):
    variables = {}

    with open(filepath, "r") as f:
        exec(f.read(), variables)

    x = variables["x"]

    plot_data = {
        "21 Jobs": (variables["jobs_21"], 21),
        "23 Jobs": (variables["jobs_23"], 23),
        "25 Jobs": (variables["jobs_25"], 25),
        "27 Jobs": (variables["jobs_27"], 27),
        "29 Jobs": (variables["jobs_29"], 29),
    }

    fig, ax = plt.subplots(
        figsize=(6, 4.5),
        dpi=300
    )

    for ((label, (y, total_jobs)), (marker, linestyle), color) in zip(
        plot_data.items(),
        styles,
        colors
    ):
        y_norm = [val / total_jobs for val in y]
        y=y_norm
        ax.plot(
            x,
            y,

            label=label,

            marker=marker,
            linestyle=linestyle,

            color=color,

            **base
        )

    ax.set_xlabel(
        "Mandatory Jobs (%)",
        fontsize=fontsize
    )

    ax.set_ylabel(
        "Optional Jobs (%)",
        fontsize=fontsize
    )

    ax.set_xlim(-0.015, 1.0)
    ax.set_ylim(-0.02, 0.61)

    ax.set_xticks([0.0, 0.3, 0.6, 0.9])
    ax.set_yticks([0.0, 0.3,0.6])

    ax.tick_params(
        axis="both",
        labelsize=fontsize - 3
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5
    )


    fig.tight_layout(rect=[0, 0, 1, 1])

    outname = filepath.replace(".txt", ".pdf")

    fig.savefig(
        outname,
        bbox_inches="tight"
    )

    plt.show()