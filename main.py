from modeling.neuron import Neuron
import numpy as np
from pathlib import Path

HOME = Path.cwd()


def main():
    dataset_root = HOME / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    neu_store_path = dataset_root / "neuron.npy"
    if not neu_store_path.exists():
        neu = Neuron()
        np.save(str(neu_store_path), neu)
    else:
        neu = np.load(str(neu_store_path), allow_pickle=True).item()
    neu.plot_region(201)


if __name__ == "__main__":
    main()
