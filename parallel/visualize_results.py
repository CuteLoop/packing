import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import pandas as pd
from PIL import Image

# --- CONFIG ---
FRAMES_DIR = "frames"
CSV_FILE = "evolution.csv"
OUTPUT_GIF = "evolution.gif"
OUTPUT_GRAPH = "energy_plot.png"


def plot_energy_graph():
    if not os.path.exists(CSV_FILE):
        print(f"Warning: {CSV_FILE} not found. Skipping graph.")
        return

    print(f"Plotting {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
        plt.figure(figsize=(10, 5))
        plt.plot(df["Epoch"], df["BestEnergy"], label="Best Energy", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Energy (Negative is Better)")
        plt.title("Optimization Progress: Collaborative Annealing")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.savefig(OUTPUT_GRAPH)
        plt.close()
        print(f"Saved {OUTPUT_GRAPH}")
    except Exception as e:
        print(f"Error plotting graph: {e}")


def read_geometry_file(filepath):
    """Reads the X, Y, Angle format from the text files."""
    with open(filepath, "r") as f:
        lines = f.readlines()[1:]  # Skip header

    data = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            data.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return data


def create_frame_images():
    txt_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "epoch_*.txt")))
    if not txt_files:
        print("No frame text files found.")
        return []

    print(f"Found {len(txt_files)} snapshots. Rendering images...")

    images = []

    # Fixed bounds (adjust if needed)
    xlim = (-150, 150)
    ylim = (-150, 150)

    for i, txt_file in enumerate(txt_files):
        data = read_geometry_file(txt_file)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")

        patches = []
        for (x, y, _ang) in data:
            size = 10.0
            rect = MplPolygon([
                (x - size, y - size),
                (x + size, y - size),
                (x + size, y + size),
                (x - size, y + size)
            ])
            patches.append(rect)

        p = PatchCollection(patches, alpha=0.7, edgecolor="black", facecolor="cyan")
        ax.add_collection(p)

        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ax.scatter(xs, ys, c="red", s=5)

        epoch_name = os.path.basename(txt_file).replace(".txt", "")
        ax.set_title(f"State: {epoch_name}")

        png_path = txt_file.replace(".txt", ".png")
        plt.savefig(png_path)
        plt.close()

        images.append(Image.open(png_path))
        if i % 10 == 0:
            print(f"Rendered {i}/{len(txt_files)}...")

    return images


def make_gif(images):
    if images:
        print(f"Saving GIF to {OUTPUT_GIF}...")
        images[0].save(
            OUTPUT_GIF,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=200,
            loop=0,
        )
        print("Done.")


if __name__ == "__main__":
    plot_energy_graph()
    imgs = create_frame_images()
    make_gif(imgs)
