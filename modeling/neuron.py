from data_prepare.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List


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
        self.c_0 = self.get_c(0, 1000)
        self.c_1 = self.get_c(1000, 7000)
        self.c_2 = self.get_c(7000, self.obs_len)
        self.fr_list = np.asarray([self.fr_0, self.fr_1, self.fr_2]).transpose()
        self.p_list = np.asarray([self.p_0, self.p_1, self.p_2]).transpose()
        self.c_list = np.asarray([self.c_0, self.c_1, self.c_2]).transpose()
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
            #! 稀疏点
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

    def get_p(self,start, end, region_id = -1):
        if start == end:
            return 0.0
    
        if region_id == -1:
            p_sum = np.sum(self.data.global_S[:, start:end], axis=1)
            
            interval_len = (end - start + 1) * 0.1
            p = p_sum / interval_len
            
        return p

    def get_c(self, start, end):
        if start == end:
            return 0.0
        interval_len = (end - start + 1) * 0.1
        c_sum = np.sum(self.data.global_C[:, start:end], axis=1)
        c = c_sum / interval_len
        return c

    def devide_by_regions(self, data):
        region_data = {}
        for region_id in self.categories:
            selection = (self.region_id == region_id)
            region_data[region_id] = data[selection]
        return region_data

    def plot_brain_regions(self, region_ids = None, title = None ,*args, **kwargs):
        if region_ids:
            region_poses = self.devide_by_regions(self.data.global_centers)
            for region_id in region_ids:
                region_pos = region_poses[region_id]
                x = region_pos[:, 0]
                y = region_pos[:, 1]
                plt.scatter(x, y, c=self.cmap[region_id], label=region_id)
        else:
            colors = [self.cmap[c] for c in self.region_id]
            fig, ax = plt.subplots()
            scatter = ax.scatter(self.x, self.y, c=colors)
            legend = ax.legend(
                *scatter.legend_elements(), loc="lower left", title="Categories"
            )
                
        # set the title and axes labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Neurons in {f'Region {region_ids}' if region_ids != None else 'All Regions'}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()

    def plot_self_3d(self, type, **kwargs):
        assert type in ["fr", "p", "c"]
        category = kwargs.get("category", [])
        if category != []:
            selection = np.full(self.region_id.shape, False)
            selection[np.isin(self.region_id, category)] = True
            assert selection.any(), "no category with that id"
        else:
            selection = np.ones(self.region_id.shape[0], dtype=bool)
        save_pic = kwargs.get("save_pic", False)
        if type == "p":
            points = self.p_list[selection]
        elif type == "fr":
            points = self.fr_list[selection]
        else:
            points = self.c_list[selection]
        max_value = np.max(points)
        points = points / max_value # scale all points to the same maximum value
        colors = list(np.array([self.cmap[c] for c in self.region_id])[selection])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
        ax.set_title(f'{type}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if not save_pic:
            plt.show()
        else:
            from pathlib import Path

            pic_root = Path.cwd() / "pics" / f"三维分布图/{type}"
            pic_root.mkdir(parents=True, exist_ok=True)
            file_path = pic_root / f"{type}_{''.join(map(str, category))}.png"
            if category == []:
                file_path = pic_root / f"{type}_all.png"
            plt.savefig(file_path)
            plt.close(fig) # close figure after saving it
            print(f"Created {str(file_path)}")

    def plot_fr_p(self, indices=[0, 1, 2], **kwargs):
        category = kwargs.get("category", [])
        if category != []:
            selection = np.full(self.region_id.shape, False)
            selection[np.isin(self.region_id, category)] = True
            assert selection.any(), "no category with that id"
        else:
            selection = np.ones(self.region_id.shape[0], dtype=bool)
        save_pic = kwargs.get("save_pic", False)

        fig, axs = plt.subplots(
            1, 3, figsize=(10, 5)
        )  # create a figure with three subplots

        fr_max = max(np.max(self.fr_0), np.max(self.fr_1), np.max(self.fr_2))
        p_max = max(np.max(self.p_0), np.max(self.p_1), np.max(self.p_2))
        print(fr_max, p_max)
        for i, index in enumerate(
            indices
        ):  # iterate over the three indices and plot each subplot
            fr = self.fr_list[index][selection]
            p = self.p_list[index][selection]
            colors = list(np.array([self.cmap[c] for c in self.region_id])[selection])
            scatter = axs[i].scatter(fr, p, c=colors)
            axs[i].set_title(f"Index {index}")  # set the subplot title
            axs[i].set_xlabel("fr-axis")
            axs[i].set_ylabel("p-axis")
            axs[i].set_xlim(0, fr_max)
            axs[i].set_ylim(0, p_max)
            if i == 0:  # only show the legend for the first subplot
                legend = axs[i].legend(
                    *scatter.legend_elements(), loc="lower left", title="Categories"
                )

        fig.suptitle("Points with Colors")  # set the figure title
        if not save_pic:
            plt.show()
        else:
            from pathlib import Path

            pic_root = Path.cwd() / "pics" / "三阶段分区图/avg"
            pic_root.mkdir(parents=True, exist_ok=True)
            file_path = pic_root / f"fr_p_{''.join(map(str, category))}.png"
            if category == []:
                file_path = pic_root / f"fr_p_all.png"
            plt.savefig(file_path)
            print(f"Created {str(file_path)}")

    def plot_spikes_time_series(
        self, region_ids: List = [], stage_idx: int = -1, **kwargs
    ):
        """Plot the time series of spikes of neurons in stage(s).

        Args:
            save_pic (bool, optional): Save the picture or not. Defaults to False.
        """
        if region_ids == []:
            region_ids = self.categories
        save_pic = kwargs.get("save_pic", False)

        stages = {
            -1: (0, self.obs_len),
            0: (0, 1000),
            1: (1000, 7000),
            2: (7000, self.obs_len),
        }

        # select by region_ids

        fig, ax = plt.subplots(figsize=(20, 6))
        for region_id in region_ids:
            selection = self.region_id == region_id
            spikes = np.sum(
                self.data.global_S[
                    selection, stages[stage_idx][0] : stages[stage_idx][1]
                ],
                axis=0,
            )
            ax.plot(
                np.arange(spikes.shape[0]) * 0.1,
                spikes,
                c=self.cmap[region_id],
                label=f"Region {region_id}",
                alpha=0.5,
            )
        ax.set_title(
            f"Spikes Time Series of Neurons\nin Regions {region_ids if region_ids != [] else 'All'}\n\
            across {f'Stage {stage_idx+1}' if stage_idx != -1 else 'All Stages'}"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spikes")
        if not save_pic:
            fig.show()
        else:
            from pathlib import Path

            pic_root = Path.cwd() / "pics" / "时区图"
            pic_root.mkdir(parents=True, exist_ok=True)
            file_path = pic_root / f"times_{''.join(map(str, region_ids))}.png"
            plt.savefig(file_path)
            print(f"Created {str(file_path)}")
