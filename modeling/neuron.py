from data_prepare.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self):
        self.data = DataLoader()
        self.neuron_num, self.obs_len = self.data.global_C.shape
        self.x = self.data.global_centers[:, 0]
        self.y = self.data.global_centers[:, 1]
        self.region_id = np.ravel(self.data.brain_region_id)
        self.fr_0 = self.get_fr(0, 1000)
        self.fr_1 = self.get_fr(1000, 7000)
        self.fr_2 = self.get_fr(7000, self.obs_len)
        self.p_0 = self.get_p(0, 1000)
        self.p_1 = self.get_p(1000, 7000)
        self.p_2 = self.get_p(7000, self.obs_len)
        self.fr_list = np.asarray([self.fr_0, self.fr_1, self.fr_2])
        self.p_list = np.asarray([self.p_0, self.p_1, self.p_2])
        self.cmap = {
            20: "red",
            26: "green",
            67: "blue",
            81: "black",
            88: "orange",
            173: "purple",
            187: "cyan",
            201: "yellow",
            300: "gray",
            327: "brown",
            348: "pink",
            0: "silver",
            53: "gold",
            95: "magenta",
            228: "navy",
            334: "aqua",
            355: "fuchsia",
        }
        self.categories = np.unique(self.region_id)

    def get_fr(self, start, end):
        if start == end:
            return 0.0
        interval_len = (end - start + 1) * 0.1
        spike_num = np.count_nonzero(self.data.global_S[:, start:end], axis=1)
        f_r = spike_num / interval_len
        return f_r

    def get_p(self, start, end):
        p = np.sum(self.data.global_S[:, start:end], axis=1)
        return p


    def plot_region(self, region_id=-1):
        cate = self.region_id
        unique = np.unique(cate)

        fr_max = np.max([np.max(self.fr_0), np.max(self.fr_1), np.max(self.fr_2)])
        p_max = np.max([np.max(self.p_0), np.max(self.p_1), np.max(self.p_2)])


        colors = [self.cmap[i] for i in cate[:, 0]]

        if region_id == -1:
            for stage in range(0, 3):
                fr = self.fr_list[stage]
                p = self.p_list[stage]
                plt.subplot(1, 3, stage + 1)
                plt.scatter(fr, p, c=colors, s=5)
                plt.title(f"Stage {stage+1}")
                plt.xlabel("fr(num/s)")
                plt.ylabel("p")
            plt.show()
        else:
            if region_id not in unique:
                print("wrong region_id")
                quit()

            start = [0, 1000, 7000]
            end = [1000, 7000, self.obs_len]
            for stage in range(0, 3):
                fr = self.fr_list[stage]
                p = self.p_list[stage]

                id = self.region_id[start[stage] : end[stage]] == region_id
                index = np.where(id)[0]

                fr = fr[index]
                p = p[index]
                color = self.cmap[region_id]

                plt.subplot(1, 3, stage + 1)
                plt.scatter(fr, p, c=color, s=5)
                plt.title(f"Stage {stage+1} Region id {region_id}")
                plt.xlim(0, fr_max)
                plt.ylim(0, p_max)
                plt.xlabel("fr(num/s)")
                plt.ylabel("p")
            plt.show()

    def plot_brain_regions(self):
        colors = [self.cmap[c] for c in self.region_id]
        fig, ax = plt.subplots()
        scatter = ax.scatter(self.x, self.y, c=colors)
        legend = ax.legend(*scatter.legend_elements(),
                   loc="lower left", title="Categories")

        # set the title and axes labels
        ax.set_title("Points with Colors")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        plt.show()

    def plot_fr_p(self, index=0, **kwargs):
        category = kwargs.get('category', [])
        if category != []:
            selection = np.full(self.region_id.shape, False)
            selection[np.isin(self.region_id, category)] = True
            assert selection.any(), "no category with that id"
        else:
            selection = np.ones(self.region_id.shape[0], dtype=bool)
        save_pic = kwargs.get('save_pic', False)
        fr = self.fr_list[index][selection]
        p = self.p_list[index][selection]
        colors = list(np.array([self.cmap[c] for c in self.region_id])[selection])
        fig, ax = plt.subplots()
        scatter = ax.scatter(fr, p, c=colors)
        legend = ax.legend(*scatter.legend_elements(),
                loc="lower left", title="Categories")

        # set the title and axes labels
        ax.set_title("Points with Colors")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        if not save_pic:
            plt.show()
        else:
            from pathlib import Path
            pic_root = Path.cwd() / "pics" / "单阶段分区图" / str(index)
            pic_root.mkdir(parents=True, exist_ok=True)
            file_path = pic_root / f"fr_p_{''.join(map(str, category))}.png"
            plt.savefig(file_path)
            print(f"Created {str(file_path)}")


