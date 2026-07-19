import pandas as pd
import matplotlib.pyplot as plt


files = [
    "solver_overhead/mean_M5_D250.csv",
    "solver_overhead/mean_M10_D250.csv",
    "solver_overhead/mean_M15_D250.csv",
]


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
    ("^", "-."),
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

labels = [
    "5 Models",
    "10 Models",
    "15 Models",
]


fig, ax = plt.subplots(
    figsize=(6, 3.6),
    dpi=300
)


for filepath, label, (marker, linestyle), color in zip(
    files,
    labels,
    styles, colors
):

    df = pd.read_csv(filepath)

    ax.plot(
        df["jobNumber"],
        df["mean_runtime_ms"],

        label=label,

        linestyle=linestyle,
        marker=marker,

        color=color,

        **base
    )


ax.set_xlabel(
    "Jobs",
    fontsize=fontsize
)

ax.set_ylabel(
    "Runtime (ms)",
    fontsize=fontsize
)


ax.set_xlim(1, 30)
ax.set_ylim(0, 11)

ax.set_xticks([1, 10, 20, 30])
ax.set_yticks([0,  5, 10])


ax.tick_params(
    axis="both",
    labelsize=fontsize - 3
)


ax.grid(
    True,
    linestyle=":",
    linewidth=0.5
)


fig.tight_layout(
    rect=[0, 0, 1, 1]
)


fig.savefig(
    "Deadline_250.pdf",
    bbox_inches="tight"
)

plt.show()
