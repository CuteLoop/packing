import pandas as pd
import matplotlib.pyplot as plt

# Read header line to get L if you want (simple parse)
with open("best_circles.csv", "r") as f:
    header = f.readline().strip()
# Example header: "# L=10 bestE=... N=..."
L = float(header.split("L=")[1].split()[0])

df = pd.read_csv("best_circles.csv", comment="#")

fig, ax = plt.subplots()
ax.set_aspect("equal")

# Draw container square centered at origin
half = L / 2
ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half])

# Draw circles
for _, row in df.iterrows():
    c = plt.Circle((row["x"], row["y"]), row["r"], fill=False)
    ax.add_patch(c)

ax.set_xlim(-half*1.05, half*1.05)
ax.set_ylim(-half*1.05, half*1.05)
plt.savefig("best_circles.png", dpi=300)
plt.show()
