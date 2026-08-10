import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("data/train.csv")


# Create plots folder if it doesn't exist
import os
os.makedirs("plots", exist_ok=True)


# ============================================================
# 1. Survival Distribution
# ============================================================

df["Survived"].value_counts().plot(kind="bar")

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.xticks(
    ticks=[0, 1],
    labels=["Did Not Survive", "Survived"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "plots/survival_distribution.png"
)

plt.close()


# ============================================================
# 2. Survival by Gender
# ============================================================

pd.crosstab(
    df["Sex"],
    df["Survived"]
).plot(kind="bar")

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.legend(
    ["Did Not Survive", "Survived"],
    title="Outcome"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "plots/survival_by_gender.png"
)

plt.show()


# ============================================================
# 3. Survival by Passenger Class
# ============================================================

pd.crosstab(
    df["Pclass"],
    df["Survived"]
).plot(kind="bar")

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.legend(
    ["Did Not Survive", "Survived"],
    title="Outcome"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "plots/survival_by_class.png"
)

plt.show()
