from modeling.neuron import Neuron
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import product
from typing import *
from pathlib import Path
from prettytable import PrettyTable, MARKDOWN

HOME = Path.cwd()
dataset_root = HOME / "dataset"
dataset_root.mkdir(parents=True, exist_ok=True)
neu_store_path = dataset_root / "neuron.npy"
if not neu_store_path.exists():
    neu = Neuron()
    np.save(str(neu_store_path), neu)
else:
    neu = np.load(str(neu_store_path), allow_pickle=True).item()


def plot_active_regions_by_steps(
    neuron,
    abs_threshold=None,
    rel_threshold=None,
    step=100,
    unit_sample_num=10,
    *args,
    **kwargs,
):
    """
    绘制每一时间步的活跃区域随时间变化的图像

    Args:
    ----
    neuron: NeuronData
        NeuronData 类型的对象，包含神经元数据
    abs_threshold: float
        绝对阈值，指尖峰强度大于等于该值的被认为是尖峰，单位为微伏（uV）
    rel_threshold: float
        相对阈值，指尖峰强度在所有样本中的排名大于等于该值的被认为是尖峰，
        用于计算活跃区域，取值范围为 [0, 1]，默认为 0.8
    step: int
        时间步长，单位为秒（s），默认为 100
    *args, **kwargs:
        可变参数和关键字参数，用于传递给 neuron.plot_brain_regions() 函数

    Returns:
    -------
    None
    """
    # 获取是否保存 GIF 图像的参数
    save_gif = kwargs.pop("save_gif", False)

    # 将神经元数据按照区域分组，并计算每个区域的平均尖峰强度
    region_spikes = neuron.devide_by_regions(neuron.data.global_S)
    region_all_mean_spikes = {
        region_id: np.mean(region_spikes[region_id], axis=0)
        for region_id in neuron.categories
    }

    # 计算每个区域的平均尖峰强度，并按照时间步长分组
    max_step_num = int(neuron.obs_len / unit_sample_num // step)
    max_obs_len = step * unit_sample_num * max_step_num
    region_step_mean_spikes = {
        region_id: np.mean(
            region_all_mean_spikes[region_id][:max_obs_len].reshape(-1, step * 10),
            axis=1,
        )
        for region_id in neuron.categories
    }

    # pdprint(region_all_mean_spikes[0].shape)
    # pdprint(region_all_mean_spikes[0][:max_obs_len].shape)
    # pdprint(region_all_mean_spikes[0][:max_obs_len].reshape(step * 10, -1).shape)
    # pdtest(region_step_mean_spikes[0].shape, (14,))

    # # 各脑区时间步内平均尖峰强度随时间变化的 3D 图像
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # for region_idx, region_id in enumerate(neuron.categories):
    #     # ax.scatter(np.arange(max_step_num) + 0.5, np.repeat(region_idx, max_step_num), c=neuron.cmap[region_id], s = region_step_mean_spikes[region_id]*1000)
    #     ax.plot(np.arange(max_step_num) + 0.5, np.repeat(region_idx, max_step_num), region_step_mean_spikes[region_id],
    #             c=neuron.cmap[region_id],label=f"{region_id}")
    # ax.set_xticks(np.arange(max_step_num))
    # ax.set_xticklabels(np.arange(max_step_num) * step)
    # ax.set_xlabel("Time (s)")
    # ax.set_yticks(np.arange(len(neuron.categories)))
    # ax.set_yticklabels(neuron.categories)
    # ax.set_ylabel("Region ID")
    # ax.set_title(f"{step} s Time Step Mean Spikes of Regions across Time")
    # fig.show()

    # 在每个时间步中，找到活跃区域
    # 将每个区域的平均尖峰强度按照升序排列，取排名前 20% 的区域作为活跃区域
    region_activities_descend_in_steps = []  # step_idx -> (region_id, mean_spike)
    for i in range(max_step_num):
        region_activities = np.array(
            [
                (region_id, mean_spikes[i])
                for region_id, mean_spikes in region_step_mean_spikes.items()
            ]
        )
        # pdprint(region_mean_spikes) # v
        region_activities_descend = np.flip(
            region_activities[region_activities[:, 1].argsort()], axis=0
        )
        # pdprint(region_ids_mean_spikes_descend)
        region_activities_descend_in_steps.append(region_activities_descend)

    # for step_idx, step_region_activities_descend in enumerate(region_activities_descend_in_steps):

    # 根据 active_regions_in_steps 调用 neuron.plot_brain_regions() 绘制活跃区域随时间步变化的图像
    # 每个时间步都调用 neuron.plot_brain_regions() 函数绘制活跃区域随时间步变化的图像
    if save_gif:
        # 如果要保存为 GIF 图像，首先创建一个空的图像路径序列 image_path_sequence
        image_path_sequence = []
    # pdprint(active_region_ids_in_steps) # x

    for step_idx, step_region_activities_descend in enumerate(
        region_activities_descend_in_steps
    ):
        # 计算阈值对应的区域数量
        # 相对阈值 rel_threshold 为 0.2 时，表示取排名前 20% 的区域作为活跃区域
        if rel_threshold is not None and abs_threshold is None:
            num_rel_active_regions = int(len(neuron.categories) * (1 - rel_threshold))
            step_region_activities_descend_above_threshold = (
                step_region_activities_descend[:num_rel_active_regions]
            )
        elif abs_threshold is not None and rel_threshold is None:
            row_idxes = np.where(
                np.abs(step_region_activities_descend[:, 1]) > abs_threshold
            )[0]
            # pdprint(row_idxes)
            step_region_activities_descend_above_threshold = np.take_along_axis(
                step_region_activities_descend, row_idxes[:, np.newaxis], axis=0
            )

        # pdprint(region_ids_mean_spikes_descend)
        # pdprint(step_region_activities) # x
        # 根据时间步长计算时间段的开始和结束时间，用于设置图像标题
        start_time = step_idx * step
        end_time = (step_idx + 1) * step
        regions_desc = None
        if rel_threshold is not None and abs_threshold is None:
            regions_desc = f"Top {int((1-rel_threshold)*100)}% Active Regions"
        elif abs_threshold is not None and rel_threshold is None:
            regions_desc = f"Regions with Mean Spike above {abs_threshold}"
        else:
            raise NotImplementedError()
        title = None
        if rel_threshold is not None and abs_threshold is None:
            title = f"{regions_desc} in [{start_time},{end_time}) s"
        elif abs_threshold is not None and rel_threshold is None:
            title = f"{regions_desc} in [{start_time},{end_time}) s"
        else:
            raise NotImplementedError()

        labels = [
            f"{int(step_region_activities_descend_above_threshold[idx][0])} ({float(step_region_activities_descend_above_threshold[idx][1]):.6f})"
            for idx in range(step_region_activities_descend_above_threshold.shape[0])
        ]
        region_ids = step_region_activities_descend_above_threshold[:, 0]

        # 为保存图像设置路径和文件名
        threshold_dirname = None
        if rel_threshold is not None and abs_threshold is None:
            threshold_dirname = f"rel={rel_threshold}"
        elif abs_threshold is not None and rel_threshold is None:
            threshold_dirname = f"abs={abs_threshold}"
        else:
            raise NotImplementedError()
        save_dir = Path.cwd() / f"pics/各时段活跃区域图/{threshold_dirname}/step={step}ms/"
        save_path = save_dir / f"{title.lower().replace(' ', '-')}.png"

        # 调用 neuron.plot_brain_regions() 函数绘制活跃区域随时间步变化的图像，并保存图像
        neuron.plot_brain_regions(
            list(region_ids),
            title=title,
            labels=labels,
            save_path=save_path,
            *args,
            **kwargs,
        )

        # 如果要保存为 GIF 图像，则将图像路径添加到 image_path_sequence 中
        if save_gif:
            image_path_sequence.append(save_path)

    fig, ax = plt.subplots()
    for region_idx, region_id in enumerate(neuron.categories):
        ax.scatter(
            np.arange(max_step_num) + 0.5,
            np.repeat(region_idx, max_step_num),
            c=neuron.cmap[region_id],
            s=region_step_mean_spikes[region_id] * 1000,
        )
    ax.set_xticks(np.arange(max_step_num))
    ax.set_xticklabels(np.arange(max_step_num) * step)
    ax.set_xlabel("Time (s)")
    ax.set_yticks(np.arange(len(neuron.categories)))
    ax.set_yticklabels(neuron.categories)
    ax.set_ylabel("Region ID")
    ax.set_title(f"{step} s Time Step Mean Spikes of Regions across Time")
    fig.show()

    # 如果要保存为 GIF 图像，则将图像路径序列中的所有图像打开并保存为 GIF 图像
    if save_gif:
        # 打开所有的 PNG 图像文件并将其添加到图像序列中
        image_sequence = []
        for image_path in image_path_sequence:
            image = Image.open(image_path)
            image_sequence.append(image)

        # 保存图像序列为 GIF 图像文件

        gif_basename = f"{regions_desc} Throughout the Entire Process By {step} s Step".lower().replace(
            " ", "-"
        )
        image_sequence[0].save(
            save_dir / f"{gif_basename}.gif",
            save_all=True,
            append_images=image_sequence[1:],
            duration=200,
            loop=0,
        )


def plot_spikes_time_series(
    neuron,
    region_ids: List = [],
    stage_idx: int = -1,
    step: int = 100,
    unit_sample_num=10,
    **kwargs,
):
    """Plot the time series of spikes of neurons in stage(s).

    Args:
        save_pic (bool, optional): Save the picture or not. Defaults to False.
    """
    if region_ids == []:
        region_ids = neuron.categories
    save_pic = kwargs.get("save_pic", False)
    max_step_num = int(neuron.obs_len / unit_sample_num // step)
    max_obs_len = step * unit_sample_num * max_step_num

    stages = {
        -1: (0, max_obs_len),
        0: (0, 1000),
        1: (1000, 7000),
        2: (7000, max_obs_len),
    }

    # select by region_ids

    region_step_mean_spikes = {}
    for region_id in neuron.categories:
        selection = neuron.region_id == region_id
        region_neuron_num = np.count_nonzero(selection)
        neurons_step_mean_spikes = np.mean(
            neuron.data.global_S[
                selection, stages[stage_idx][0] : stages[stage_idx][1]
            ].reshape(region_neuron_num, -1, step * unit_sample_num),
            axis=2,
        )
        region_step_mean_spikes[region_id] = np.mean(neurons_step_mean_spikes, axis=0)
        # pdtest(region_step_mean_spikes[region_id].size, 14) # v

    # visualize
    all_region_mean_spikes = np.array(list(region_step_mean_spikes.values()))
    # pdprint(all_region_mean_spikes.shape)
    # pdprint(all_region_mean_spikes)
    max_region_mean_spike = np.max(all_region_mean_spikes)
    fig, ax = plt.subplots(figsize=(20, 6))

    ax.set_xticks(np.arange(14 + 1) * 100)
    ax.set_ylim(0, max_region_mean_spike)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spikes")
    title = f"{step}s Step Mean Spikes Time Series of Neurons \
in {f'Region {region_ids}' if len(region_ids) < len(neuron.categories) else 'All Regions'} \
across {f'Stage {stage_idx+1}' if stage_idx != -1 else 'All Stages'}"
    ax.set_title(title)

    for region_id in region_ids:
        spikes = region_step_mean_spikes[region_id]
        ax.plot(
            np.arange(spikes.shape[0]) * step,
            spikes,
            c=neuron.cmap[region_id],
            label=f"{region_id}",
            alpha=0.5,
        )
    ax.legend()
    fig.show()
    if save_pic:
        from pathlib import Path

        pic_root = Path.cwd() / "pics" / "各区域峰值曲线时间序列图"
        pic_root.mkdir(parents=True, exist_ok=True)
        file_path = pic_root / f"{title.lower().replace(' ', '-')}.png"
        plt.savefig(file_path)
        print(f"Created {str(file_path)}")


def create_markdown_table():
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.align = "l"
    return table


def pdprint(*args, **kwargs):
    print("[debug]", *args, **kwargs)


def pdtest(actual, expected, *args, **kwargs):
    print("[test][ actual ]", actual, *args, **kwargs)
    print("[test][expected]", expected, *args, **kwargs)
    if actual != expected:
        print("[test][warning!]", "=== check this! ===", *args, **kwargs)


def cluster_2d(data):
    assert data.shape[1] == 2
    # Perform clustering with KMeans
    kmeans = KMeans(n_clusters=17)
    kmeans.fit(data)

    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plot the data points with different colors for each cluster
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="rainbow")
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", s=300, c="black")
    plt.show()


def plot_fr_p_c():
    for i in neu.categories:
        neu.plot_fr_p_c(indices=[0], category=[int(i)], save_pic=True)
    neu.plot_fr_p_c(indices=[0], category=[], save_pic=True)


def plot_3d_3_field():
    for type, category in product(["fr", "p", "c"], neu.categories):
        neu.plot_self_3d(type, category=[int(category)], save_pic=True)
    for type in ["fr", "p", "c"]:
        neu.plot_self_3d(type, category=[], save_pic=True)


def cluster_3d(data, n_clusters=11):
    assert data.shape[1] == 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(n_clusters):
        cluster_points = data[kmeans.labels_ == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    region_groups = [
        [20, 88, 173, 300, 327, 348],
        [0, 53, 81, 95, 187, 355],
        [26, 201],
        [67, 228],
        [334],
    ]
    for region_id, region_name in neu.region_id2name.items():
        print(f"[{region_id}]: {region_name}")
    for region_group in region_groups:
        title = f"Region {region_group}"
        neu.plot_brain_regions(
            region_group,
            title=title,
            save_pic=True,
            save_path=Path.cwd()
            / "pics"
            / "分组脑区图"
            / (title.lower().replace(" ", "-") + ".png"),
        )

    steps = [100, 50, 10, 1]
    for step in steps:
        plot_spikes_time_series(neu, region_ids=[], step=step, save_pic=True)

    ploted_regions = []
    left_regions = neu.categories
    for region_group in region_groups:
        plot_spikes_time_series(neu, region_ids=region_group, save_pic=True)

        # 判断 arr1 中哪些元素包含在 arr2 中
        ploted_regions += region_group
        mask = np.in1d(left_regions, ploted_regions)

        # 删除 arr1 中已经包含在 arr2 中的元素
        left_regions = left_regions[~mask]
        plot_spikes_time_series(neu, region_ids=left_regions, save_pic=True)

    for region_id in neu.categories:
        plot_spikes_time_series(neu, region_ids=[region_id], save_pic=True)
        # break

    neu.plot_brain_regions()

    neuron_idx = 2
    spike_units = neu.data.global_S[neuron_idx]
    spike_times = np.nonzero(spike_units)
    spike_durations = spike_units[spike_times]

    print(spike_units)
    print(spike_times)
    print(spike_durations)
    print(np.count_nonzero(spike_units))
    plt.plot(spike_units)