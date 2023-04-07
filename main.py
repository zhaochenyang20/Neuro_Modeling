from data_prepare.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def plot_brain_region(data: DataLoader):
    print("======= Neural cells number in different region =======")
    pos = data.global_centers
    cate = data.brain_region_id

    unique, counts = np.unique(cate, return_counts=True)
    for i in range(len(unique)):
        print("Region", unique[i], ":", counts[i])

    cmap = {
        20: 'red',
        26: 'green',
        67: 'blue',
        81: 'black',
        88: 'orange',
        173: 'purple',
        187: 'cyan',
        201: 'yellow',
        300: 'gray',
        327: 'brown',
        348: 'pink',
        0: 'silver',
        53: 'gold',
        95: 'magenta',
        228: 'navy',
        334: 'aqua',
        355: 'fuchsia',
    }
    colors = [cmap[i] for i in cate[:, 0]]
    x = pos[:, 0]
    y = pos[:, 1]
    plt.scatter(x, y, c=colors, s=5)
    plt.title("Neural cells distribution map")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def main():
    data = DataLoader()
    plot_brain_region(data)


if __name__ == "__main__":
    main()