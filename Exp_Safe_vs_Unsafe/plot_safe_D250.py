import pandas as pd
import matplotlib.pyplot as plt



file = "Deadline_250_safe_unsafe_results.csv"


df = pd.read_csv(file)

avg_df = (
    df.groupby("Jobs")
    .mean(numeric_only=True)
    .reset_index()
)

# Keep only Jobs <= 20
avg_df = avg_df[avg_df["Jobs"] <= 20]

fontsize = 30



plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.linewidth": 1.5,
})

fig, ax = plt.subplots(
    figsize=(7, 4.5),
    dpi=300
)
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
]

base = dict(
    linewidth=2,
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=1,
)


ax.plot(
    avg_df["Jobs"],
    avg_df["Optimal_Safe"],
    marker="s",
    linestyle="-",
    label="RETINA",
    **base
)

ax.plot(
    avg_df["Jobs"],
    avg_df["Low_Safe"],
    marker="o",
    linestyle=":",
    label="RETINA-Low",
    **base
)

ax.plot(
    avg_df["Jobs"],
    avg_df["High_Safe"],
    marker="^",
    linestyle="--",
    label="RETINA-High",
    **base
)

ax.plot(
    avg_df["Jobs"],
    avg_df["CA_MOT_Safe"],
    marker="D",
    linestyle="-.",
    label="CA-MOT",
    **base
)


ax.set_xlabel(
    "Total Jobs",
    fontsize=fontsize
)

ax.set_ylabel(
    "Safety Condition\nSatisfying Jobs",
    fontsize=fontsize
)


ax.set_ylim(4, 18)
ax.set_xlim(10, 21)

ax.set_yticks([5, 11, 17])
ax.set_xticks([11,  14, 17, 20])


ax.tick_params(
    axis='both',
    labelsize=fontsize-3,
    width=1.5,
    length=6
)


ax.grid(
    True,
    linestyle="--",
    linewidth=0.8,
    alpha=0.4
)


plt.tight_layout()


outname = file.replace(".csv", ".pdf")

fig.savefig(outname,bbox_inches="tight")


plt.show()