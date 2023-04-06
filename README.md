# Math_Modeling

## 4 月 5 日

### 分工

- [ ]  赵晨阳去搬救兵，找生医背景的同学询问需要考虑的因素和可能的模型；
- [ ] 飞翰去写数据脚本，能够 load 数据并且说明每个数据文件的意义；
- [ ] 雨轩去研究往年报告，看看老师喜欢什么内容。

这部分内容先做完，不用急着读参考文献，我们把手头内容写完了再去讨论可行模型。

## 其他文档

题干：https://tongyx361.notion.site/SM-a29a6c05a68b47c2815c861c6c8a1dea

数据：https://cloud.tsinghua.edu.cn/d/33ac858407044d48aced/

### 参考

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