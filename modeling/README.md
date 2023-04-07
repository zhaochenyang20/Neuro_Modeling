## 1. Neuron类

> 该类记录了神经元相关的特征信息与接口

### 1.1  变量声明

- `x，y`：横纵坐标
- `fr_{i}`：第 i 个阶段的发放率（次数每秒）
- `p_{i}`：第 i 个阶段的spike值总和
- `fr_list`：三个阶段的发放率组成的列表
- `p_list`：三个阶段的spike值总和组成的列表

### 1.2 函数声明

- `plot_region(region_id=-1)`：

  - 不传入`region_id`的时候，默认绘画三张图，代表了三个阶段。图的横纵坐标为发放率与spike值和，点的颜色代表了其类别。

  - 传入`region_id`的时候，仅绘画出对应大脑区域的细胞的三个阶段的特征图。

  - 可使用的`region_id`与其对应的颜色如下：

    ```python
    {20: "red",
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
    355: "fuchsia",}
    ```

    

