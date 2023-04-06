# Math_Modeling

## 4 月 5 日

### Notion文档：
- [2023校内数学建模比赛文档](https://tongyx361.notion.site/SM-a29a6c05a68b47c2815c861c6c8a1dea)

### 分工

- [x]  赵晨阳去搬救兵，找生医背景的同学询问需要考虑的因素和可能的模型；
- [x] 飞翰去写数据脚本，能够 load 数据并且说明每个数据文件的意义；
- [x] 雨轩去研究往年报告，看看老师喜欢什么内容。

这部分内容先做完，不用急着读参考文献，我们把手头内容写完了再去讨论可行模型。

## 其他文档

题干：https://tongyx361.notion.site/SM-a29a6c05a68b47c2815c861c6c8a1dea

数据：https://cloud.tsinghua.edu.cn/d/33ac858407044d48aced/

# 4 月 6 日

## 参考

1. 涉及两个问题：识别不同类型的神经元发放，以及提取时空特征进行差异性识别；
2. 目前神经科学方面还没有看见有公认的 top model，但针对尖峰数据处理有许多成熟的工具，大家可以康康参考论文的 method 部分对数据的处理（研究问题和主题有很大差异，但都是处理尖峰数据，其使用的方法和引用的 package / paper 有参考价值）
3. 时空特征的提取是一个需要有 trade off 的过程，深度学习方面有提取时间特征较好的模型，也用提取空间较好的模型（比如 resnet），我之前有看到尝试融合这两方面的论文，虽然信号类型不同，但或许思想方法可以形成参照

---

我觉得宏观上这个题就是让你分析钙信号的特点 钙信号越强代表神经元活动越强。然后不同的脑区，大部分神经元的活动对应共同的结果，指向一个整体的功能（但一个脑区内也会有一些特殊的神经元的活动存在 有的是噪声 有的是有一些调控功能 这个不好说）生物里 现在有几类神经元研究还挺火的，1.找动物宏观的行为对应的脑区以及具体构成通路神经元 2.开发记录神经元活动的新技术 有个双光子钙成像系统前几年挺火的。我觉得你们这个应该是原始数据已经记录完成了 就是分析不同位置神经元在不同麻醉时间的活动情况

-----

1.  Histaminergic neuron system in the brain: distribution and possible functions
2. Distribution of orexin neurons in the adult rat brain
3. Distribution of cholinergic neurons in rat brain: demonstrated by the immunocytochemical localization of choline acetyltransferase

第一个问题的 neuron distribution 的 可能会有帮助 hhh

## 数据内容

`global_C(N, M): changes of calcium signals(0.1 seconds between adjacent elements)`

钙信号强度

`global_S(N, M): The peak time of calcium activity in neuronal cells. A number is other than 0 means that the activity of the neuron reaches its peak at this time.`

S spike 超过阈值会放电，非负浮点数

`global_centers(N ,2): the space location of neural cells`

细胞的空间位置， id 和前述的 N 对应

`brain_region_id(N ,1): the number of brain regions where neurons are located`

细胞所在的大脑区域，id 和前述的 N 对应，范围不大的 int，在 30 ~ 100 之间

`brain_region_name(N ,1): the name of brain regions where neurons are located`

细胞所在的大脑区域名字

``*_org: elements before clipping(infer_results_1 and infer_results_2 have different M, above elements use min(M1,M2))``

一共有两组数据，第二组稍微舍弃了 1% 左右的数据

N = 4000

M = 14000

## 目标

1. 建立模型解释全脑神经元是否具有多种不同类型的发放规律
2. 对清醒-麻醉-苏醒过程在脑区间的空间差异性和时间差异性进行识别和判断

## 先验知识

这里引入一些先验知识，一般而言我们不会对于 spike 建模，而是对于 neuron 建模。对于每个 neuron，在三段时间内会发生若干多次放电，每次放电具有一个非零数值。于是我们可以得得到某个区间内关于放电的两个特征：

- $f_r$：放电频率（次数），所考虑的区间内非零电位值的个数
- $p$：电位数值和，所考虑区间所有电位的数值和

## 预期结果

这两个特征我们预期得到的结果是：

1. 不同阶段主要活跃的分区分区不同：也即我们对于三个阶段分别聚类，每次聚类选出这个阶段的 $f_r$ 和 $p$，在二维空间进行聚类，得到二位平面上的聚类结果。与其的结论是，在某个区间，活跃的 Neuron 主要来自某些脑区。
2. 同一阶段内，不同脑区受影响的时间先后和长短不同：也即对于某个确定的区间，我们已知活跃的脑区，对于每个脑区，直接将所有 Neuron 的 Spike 图染上同一颜色，进行重叠，然后区分出不同脑区放电的时间先后。

## 思路

因此，我们的思路如下：

1. 首先，对于每个时间阶段，在 $f_r$ 和 $p$ 构成的二维平面上把每个脑区的所有 Neuron 绘图。这样以来，我们得到三个阶段对应的三张图，每个图上不同脑区的 Neuron 聚集的区域有所不同。
2. 其次，仍旧对于每个时间阶段，我们在已经得到了该阶段活跃的脑区后，将每个脑区内所有 Neuron 的 Spike 图染上同一颜色，进行重叠，然后区分出不同脑区放电的时间先后。

如此以来，我们拿着很强的先验结论先去验证，希望这些预期结论都是正确的。而后，如果预期的结论都是正确的，再对于 1. 进行聚类，用实验严格证明，而不是我们简单的可视化算法感性认知。

对于 2.，我们其实简单计算每个脑区在自身活跃时间段内所有 Neuron 的 Spike 的统计学规律即可，也即计算这个脑区所有 Neuron 出现 Spike 的时间的均值和方差。

一个完整的 spike 是 5 维的，时间，平面坐标，放电数值，分区标号

- 首先需要一个 spike 类，记录五维度的特征
- 基本的思路是 omit 某一些维度，然后进行聚类，查看某一簇在忽略的维度上的特征

## 分工

建立 Neuron Class

```python
# 二位坐标
x = 
y = 
# f_r_{} 表示某个时间阶段上的发放率
# p_r_{} 表示某个时间阶段上 spike 之和
f_r_0 = 
p_r_0 = 
f_r_1 = 
p_r_1 = 
f_r_2 = 
p_r_0 = 
# 脑区分布，脑区分布实际上和二维坐标结合，所以聚类时选择全部 omit id, x，y，三者作为聚类结果的解释
id = 
```

1. 脑区分布图 + 写类接口 + 存数据——A

2. 验证思路 1.——B

3. 验证思路 2.——C

## 后记

目前的思路遗漏了两个关健信息：

1. Neuron 的二维坐标：但我目前认为二维坐标仅仅用于划分脑区（这个需要 A 去验证），同一个脑区内的点我们可以先忽略二维坐标。后续如果需要对于某个脑区内部详细讨论，再进一步分析；
2. 我们忽略了钙信号的情况，但目前我们认为钙信号和 Spike 是类似的，故而先讨论完 Spike。
