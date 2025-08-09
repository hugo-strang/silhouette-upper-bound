# sequence A000110 in the OEIS
bell_numbers = [
    1,
    1,
    2,
    5,
    15,
    52,
    203,
    877,
    4140,
    21147,
    115975,
    678570,
    4213597,
    27644437,
    190899322,
    1382958545,
    10480142147,
    82864869804,
    682076806159,
    5832742205057,
    51724158235372,
    474869816156751,
    4506715738447323,
    44152005855084346,
    445958869294805289,
    4638590332229999353,
    49631246523618756274,
]


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create DataFrame
df = pd.DataFrame({"n": range(len(bell_numbers)), "B_n": bell_numbers})

# Set seaborn theme
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)

# Initialize figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with seaborn
sns.scatterplot(
    data=df,
    x="n",
    y="B_n",
    ax=ax,
    color=sns.color_palette("deep")[0],
    s=100,
    edgecolor="black",
    linewidth=0.5,
)

# Log scale
ax.set_yscale("log")

ax.tick_params(axis="y", which="major")

sns.despine(ax=ax)

ax.set_xlabel(r"$n$", fontsize=14, labelpad=10)
ax.set_ylabel(r"$B_n$", fontsize=14, labelpad=10)

plt.tight_layout()
plt.savefig("bell_numbers.pdf")
