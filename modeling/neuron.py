from data_prepare.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import os


class Neuron:
    def __init__(self):
        self.data = DataLoader()
        self.neuron_num, self.obs_len = self.data.global_C.shape
        self.x = self.data.global_centers[:, 0]
        self.y = self.data.global_centers[:, 1]
        self.xlim = (np.min(self.x), np.max(self.x))
        self.ylim = (np.min(self.y), np.max(self.y))
        self.region_id = np.ravel(self.data.brain_region_id)
        self.categories = np.unique(self.region_id)
        self.region_name = self.data.brain_region_name
        self.region_id2name = self.get_region_id2name()
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
        self.first_stage_list = np.asarray([self.fr_0, self.p_0, self.c_0]).transpose()
        self.total_list = np.concatenate(
            (self.fr_list, self.p_list, self.c_list), axis=1
        )
        self.first_stage_list = np.asarray([self.fr_0, self.p_0, self.c_0]).transpose()
        self.total_list = np.concatenate(
            (self.fr_list, self.p_list, self.c_list), axis=1
        )

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

        self.vec_9d = np.zeros((self.categories.shape[0], 9))  # (17, 9)
        self.vec_3d = np.zeros((self.categories.shape[0], 3))  # (17, 3)
        for i, cate in enumerate(self.categories):
            index = np.where(cate == self.region_id)[0]
            total_list_cate = self.total_list[index, :]
            first_stage_cate = self.first_stage_list[index, :]
            self.vec_9d[i, :] = np.average(total_list_cate, axis=0)
            self.vec_3d[i, :] = np.average(first_stage_cate, axis=0)

        self.vec_9d = np.zeros((self.categories.shape[0], 9))  # (17, 9)
        self.vec_3d = np.zeros((self.categories.shape[0], 3))  # (17, 3)
        for i, cate in enumerate(self.categories):
            index = np.where(cate == self.region_id)[0]
            total_list_cate = self.total_list[index, :]
            first_stage_cate = self.first_stage_list[index, :]
            self.vec_9d[i, :] = np.average(total_list_cate, axis=0)
            self.vec_3d[i, :] = np.average(first_stage_cate, axis=0)

        self.e_sim_9d = self.get_Eculid_dis(self.vec_9d)
        self.e_sim_3d = self.get_Eculid_dis(self.vec_3d)

        self.get_sim_mat(self.e_sim_9d)
        self.get_sim_mat(self.e_sim_3d)

    def get_region_id2name(self):
        region_id2name = {}
        for region_id in self.categories:
            selection = self.region_id == region_id
            region_name = np.unique(self.region_name[selection])
            assert region_name.size == 1
            if region_name[-2:] == ' 1':
                region_id2name[region_id] = region_name[0][:-2]
            else:
                region_id2name[region_id] = region_name[0]
        self.region_id2name = region_id2name
        return self.region_id2name

    def get_fr(self, start, end):
        if start == end:
            return 0.0
        interval_len = (end - start + 1) * 0.1
        spike_num = np.count_nonzero(self.data.global_S[:, start:end], axis=1)
        f_r = spike_num / interval_len
        return f_r

    def get_p(self, start, end, region_id=-1):
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
            selection = self.region_id == region_id
            region_data[region_id] = data[selection]
        return region_data

    def plot_brain_regions(self, region_ids=[], labels=[], title=None, *args, **kwargs):
        save_pic = kwargs.get("save_pic", False)
        save_path = kwargs.get("save_path", None)

        fig, ax = plt.subplots(figsize=(10, 10))
        scatter_args = {"alpha": 1, "s": 6}
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        if len(region_ids) == 0:
            region_ids = self.categories
        if len(labels) == 0:
            labels = [
                f"[{region_id}] {self.region_id2name[region_id]}"
                for region_id in region_ids
            ]
        region_poses = self.devide_by_regions(self.data.global_centers)
        for region_id, label in zip(region_ids, labels):
            region_pos = region_poses[region_id]
            x = region_pos[:, 0]
            y = region_pos[:, 1]
            ax.scatter(x, y, c=self.cmap[region_id], label=label, **scatter_args)

        # set the title and axes labels
        if not title:
            title = f"Neurons in {f'Region {region_ids}' if len(region_ids) < len(self.categories) else 'All Regions'}"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(-0.15, 1))
        fig.tight_layout()
        fig.show()
        if save_pic:
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)
            else:
                fig.savefig(f"./{title.lower().replace(' ', '-')}.png")

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
        points = points / max_value  # scale all points to the same maximum value
        colors = list(np.array([self.cmap[c] for c in self.region_id])[selection])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
        ax.set_title(f"{type}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
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
            plt.close(fig)  # close figure after saving it
            print(f"Created {str(file_path)}")

    def plot_fr_p_c(self, indices=[0, 1, 2], **kwargs):
        category = kwargs.get("category", [])
        if category != []:
            selection = np.full(self.region_id.shape, False)
            selection[np.isin(self.region_id, category)] = True
            assert selection.any(), "no category with that id"
        else:
            selection = np.ones(self.region_id.shape[0], dtype=bool)
        save_pic = kwargs.get("save_pic", False)

        num_subplots = len(indices)
        fig, axes = plt.subplots(
            1,
            num_subplots,
            figsize=(5 * num_subplots, 5),
            subplot_kw={"projection": "3d"},
        )
        if num_subplots == 1:
            axes = [axes]  # Convert the single subplot to a list

        fr_max = np.max(self.fr_list)
        p_max = np.max(self.p_list)
        c_max = np.max(self.c_list)
        print(fr_max, p_max, c_max)
        for i, index in enumerate(indices):
            fr = self.fr_list.transpose()[index][selection]
            p = self.p_list.transpose()[index][selection]
            c = self.c_list.transpose()[index][selection]
            colors = list(np.array([self.cmap[c] for c in self.region_id])[selection])
            size = 100 * c / c_max
            scatter = axes[i].scatter(fr, p, c=colors, s=size)
            axes[i].set_title(f"stage {index}")
            axes[i].set_xlabel("fr-axis")
            axes[i].set_ylabel("p-axis")
            axes[i].set_xlim(0, fr_max)
            axes[i].set_ylim(0, p_max)
            if i == 0:
                legend = axes[i].legend(
                    *scatter.legend_elements(), loc="lower left", title="Categories"
                )
                legend.set_zorder(100)
        plotname = (
            f"Brain Regin {category[0]}'s f_r, p, c"
            if category != []
            else "All Brain Regions' f_r, p, c"
        )
        fig.suptitle(plotname)
        if not save_pic:
            plt.show()
        else:
            from pathlib import Path

            pic_root = Path.cwd() / "pics" / "第一阶段分区图"
            pic_root.mkdir(parents=True, exist_ok=True)
            file_path = pic_root / f"fr_p_c_{''.join(map(str, category))}.png"
            if category == []:
                file_path = pic_root / f"fr_p_c_all.png"
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

    def get_sim_mat(self, mat):
        # create a 17x17 numpy matrix
        matrix = np.floor(mat)

        # create row and column labels
        row_labels = ["" + str(self.categories[i]) for i in range(17)]
        col_labels = ["" + str(self.categories[i]) for i in range(17)]

        # convert column labels to string format
        col_labels_str = [str(label) for label in col_labels]

        # convert matrix to latex table format
        latex_table = "\\begin{tabular}{|c|" + "|".join(["c"] * 17) + "|}\n\\hline\n"
        latex_table += " & " + " & ".join(col_labels_str) + " \\\\\n\\hline\n"
        for i in range(17):
            latex_table += (
                row_labels[i]
                + " & "
                + " & ".join([str(int(x)) for x in matrix[i]])
                + " \\\\\n"
            )
        latex_table += "\\hline\n\\end{tabular}"

        print(latex_table)

    def get_Eculid_dis(self, mat):
        matrix = mat

        # 计算每两个类别的特征之间的欧几里得距离
        distances = np.zeros((17, 17))
        for i in range(17):
            for j in range(i + 1, 17):
                distances[i][j] = np.linalg.norm(matrix[i] - matrix[j])
                distances[j][i] = distances[i][j]

        # 得到一个(17, 17)大小的矩阵
        return distances
