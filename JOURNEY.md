# Playground S6E6 完整复盘：从 0.956 到 0.968 的提升之路

> Kaggle Playground Series S6E6 — Predicting Stellar Class  
> 任务: 三分类 (GALAXY / STAR / QSO) | 指标: Balanced Accuracy  
> 577K train, 247K test, 12 raw features → 50 engineered features

---

## Phase 0: 理解数据 (Day 1)

### 物理背景

SDSS 巡天数据，五种测光波段 ugriz，三种天体：

| 类 | 物理本质 | 红移 | 颜色 |
|------|------|:---:|------|
| GALAXY | 千亿年老恒星集合 | 0.1-0.7 | 偏红 (g-r 大) |
| STAR | 银河系内单颗恒星 | ≈0 | 居中 |
| QSO | 超大质量黑洞吸积盘 | 0.3-7 | 偏蓝 (u-g 小) |

关键认知：这些天体在颜色-颜色图上自然分离，不是随机噪声。

### 基础特征工程

- 颜色指数 (u-g, g-r, r-i, i-z, u-r, g-i, r-z, u-z, g-z, color_curv)
- alpha 循环编码 (sin/cos)
- 光谱型/星系族群 label encoding

发现：log 变换无意义——星等本身就是对数尺度。正态检验：18 个 float 特征无一正态，但对树模型无关。

---

## Phase 1: 树模型的基线 (Day 1-2)

### 三树对比

| 模型 | OOF BA | 单 fold 时间 | 树生长方式 |
|------|------|------|------|
| LightGBM | 0.9565 | 66s | leaf-wise (非对称) |
| XGBoost | 0.9556 | 174s | level-wise (对称) |
| CatBoost | 0.9540 | 585s | symmetric (全层同 split) |

**结论**: LightGBM 全面最优。三树差异 <0.0026——换模型拉不开差距。

### 阈值调优 — 第一个有效提升

```
argmax:    BA = 0.9565
调阈值:    BA = 0.9656 (+0.0091)

最优阈值: GALAXY=2.14, STAR=0.81, QSO=0.53
含义: QSO 最容易选中（降阈值救回边界样本）
```

> **思考**: 标准 argmax 假设所有类等权，但 balanced accuracy 关心每类的 recall，需要针对性调整阈值。三个阈值类似于为三类各自设置"录取分数线"。

### 尝试但失败的方法

- **Binary Chain** (STAR vs 非STAR → GALAXY vs QSO): OOF 看似 0.98，实际上子 CV 导致信息泄漏，LB 仅 0.92。教训：嵌套 CV 必须对齐 split。
- **平衡采样**: LB 反而降 0.002。训练时强行三等权，test 真实分布 GALAXY 65% → 模型过度预测稀有类。
- **Pseudo-Labeling**: 98.9% test 被标上 → 几乎等于复制自己判断，无增益。

### TDA 分析 (GUDHI)

三类天体在特征空间中都是实心团块 (H₁=0)，无环/洞结构。Bottleneck 距离：GALAXY↔STAR=1.24, GALAXY↔QSO=1.13。加入 redshift 后三类均匀分离。

---

## Phase 2: 向排行榜学习 (Day 2-3)

### 研究 Chris Deotte 的方案

排行榜 11th (0.97233)。关键发现：

| 他有的 | 我们缺的 |
|------|------|
| 手写 RealMLP (CV 0.96904) | pytabkit RealMLP (0.945) |
| g/redshift, i/redshift 比值 | ❌ |
| floor 分箱 (数值→类别) | ❌ |
| delta 分位数分箱 | ❌ |
| fold-safe target encoding | ❌ |
| TabM, TabICL | ❌ |
| 19 模型集成 | 3 模型 voting |

### 核心认知转变

**不是模型不够多，是特征工程和训练机制有差距。** 他的 RealMLP 单模 (0.969) 比我们整个 ensemble (0.9656) 还高——说明 MLP 能力被我们严重低估了。

---

## Phase 3: 特征工程升级 (Day 3-4)

### Deotte 特征移植

```
新增 21 个特征:
  g/redshift, i/redshift        ← 红移比值，消除距离效应
  mag_mean, mag_range           ← 亮度统计
  log1p(redshift)               ← 对数红移
  8 个 floor 分箱 (数值取整当类别)  ← 自动分段
  2 个 delta 分位数分箱 (100+500)   ← 等密度分箱
  6 个 target encoding (alpha×delta, u×z 交互)
```

### 效果

| | 旧特征 | 新特征 | Δ |
|------|------|------|------|
| LGB 单模 | 0.9567 | 0.9586 | +0.0019 |
| XGB 单模 | 0.9562 | 0.9517 | **-0.0045** |
| MLP 单模 | 0.9450 | 0.9556 | **+0.0106** |
| 集成 (trees+mlp) | 0.9638 | 0.9642 | +0.0004 |

> **思考**: 新特征对 MLP 帮助最大 (+0.01)，因为神经网络需要这些"捷径"。XGBoost 反而降了——更多特征意味着更多噪音维度，XGB 的 level-wise 生长方式对噪声敏感。LightGBM 的 leaf-wise 自适应忽略弱特征。

### LB 结果

```
S2 (旧):    0.96670
新 trees:   0.96760  (+0.0009)
新 both:    0.96815  (+0.0015)
```

---

## Phase 4: 模型范式扩展 (Day 4-5)

### 三大树模型消融

| 模型 | 变体数 | 最佳单模 BA |
|------|:---:|------|
| LightGBM | ×5 | 0.95864 |
| XGBoost | ×5 | 0.95637 |
| CatBoost | ×2 | 0.95422 |

LGB 的 leaf-wise 生长在不同 leaf 大小/采样率下变化最小 (0.9571-0.9586)，XGB 变化最大 (0.95-0.956)，CB 最低。**树模型调参收益 <0.002。**

### pytabkit RealMLP (JaneStreet 经验移植)

JaneStreet 36 组架构扫描的核心结论：`[512,512,512] + 64 epoch` > `[1024,1024,1024]` > 4 层深度。但移植到 Playground 后发现：

```
pytabkit 最佳: 0.9571  (比树还差 0.0015)
```

差距不在层数宽度——在训练机制。

### FT-Transformer — 新的最强单模

```python
FT-Transformer (d_token=192, 3 layers, 8 heads):
  单模 OOF: 0.95981  ← 比任何树/MLP 高
```

> **思考**: attention 机制在特征间学动态交互，和树的分裂规则互补。FT 在 QSO 识别上尤其好——attention 自发学到了 redshift×颜色 的交互模式。

### 手写 RealMLP — 消融实验

```
A (pytabkit 默认):  OOF 0.95804
B (+β₂=0.98):       OOF 0.95800  Δ=0
C (+分组 lr/wd):    OOF 0.95979  Δ=+0.0018  ← 关键!
D (+scheduler):      OOF 0.95981  Δ=+0.00002
```

> **思考**: β₂ 和 scheduler 几乎无贡献；五组差异化学习率是核心提升来源。PBLD 层需要比 backbone 更低的学习率（0.115×），否则初始化好的傅里叶基被过快更新破坏。

---

## Phase 5: 集成学习 (Day 5-6)

### 简单平均 + 阈值调优 (最可靠)

```
trees (14棵):        OOF 0.95798 → tuned 0.96080
trees+mlp:           OOF 0.95917 → tuned 0.96421
trees+mlp+ft:        OOF 0.96027 → tuned 0.96501   ← 最佳 OOF
trees+mlp+ft+hc:     OOF 0.95983 → tuned 0.96529
```

### LB 对照

```
S2 (旧 3 模型):      0.96670
trees_only (14 树):  0.96760  (+0.0009)  ← 树模型提升
trees+mlp (16 模型):  0.96815  (+0.0015)  ← MLP 加成
trees+mlp+ft (17):   0.96844  (+0.0018)  ← FT 加成  🏆
trees+mlp+ft+hc (18): 0.96832  (-0.0001)  ← 手写 MLP 拖后腿
```

### Stacking 尝试

LR 做 meta-learner: OOF 虚高 (0.96833) 但 LB 反降 (0.96736)。17 个模型的 OOF 高度相关 (r>0.9)，meta-learner 学到的是噪声而非互补信号。

> **教训**: 同范式模型的**预测相关性太高**时，stacking 就是拟合噪声。简单平均 + 阈值调优在这个场景下是最优策略。

### 加权平均

FT 被赋予 50% 权重 → 严重过拟合。原因同上：模型间预测太相似。

---

## 当前成绩与差距分析

| | OOF | LB |
|------|:---:|:---:|
| 我们最佳 (trees+mlp+ft) | 0.96501 | **0.96844** |
| Deotte (11th) | ~0.970 | 0.97233 |
| Top 1 | — | 0.97283 |
| 差距 | — | **0.0044** |

### 差距来源

Deotte 的 RealMLP 单模 CV 0.96904（我们手写版 0.95981）。这 0.009 的单模差距如能补上 → 集成轻松破 0.972。

差距可能来自：
- fold-safe target encoding（我们 CV 前一次性做了）
- 更多未复刻的训练细节（numerical_noise、sample_weight_power 等）
- 特征选择（他可能只用了几十个特征中的部分）

### 已确认无效的方法

| 方法 | 效果 |
|------|------|
| Binary Chain | LB -0.044 |
| 平衡采样 | LB -0.002 |
| Pseudo-Labeling | 无增益 |
| Stacking (LR meta) | LB -0.001 |
| 加权平均 | OOF 更低 |
| 纯树调参 | ±0.001 |
| log 变换 | 无意义 |

### 有效方法排序

| 方法 | 增益 |
|------|:---:|
| 阈值调优 | +0.009 |
| Deotte 特征工程 | +0.002 |
| FT-Transformer 加入集成 | +0.0003 |
| MLP 加入集成 | +0.0005 |
| 更多树模型 | +0.0009 |
| 手写 MLP 分组 lr/wd | +0.0018 |

---

## 服务器与基础设施教训

1. **Python 路径**: 服务器有系统 python3 和 miniconda python3，要明确用 `/miniconda3/bin/python3`
2. **磁盘管理**: 删文件前先 `ls -la`，数据不放 git 仓库里
3. **screen 保活**: 所有长任务用 `screen -dmS`，不用直接 SSH
4. **CLAUDE.md 铁律**: 程序在跑不动它；删文件先检查；速率正常不切换方案
5. **OOF 存盘**: 训完立即 `np.save`，以后任何 blend 秒出
6. **git push 前验证**: `py_compile.compile` 检查语法，避免服务器上 debug

---

## 跨项目通用收获

详细记录在 `shared/task_tool_selection.md`

1. **数据量 > 模型复杂度**: 16M 行简单模型 > 2M 行复杂模型
2. **特征工程是最大杠杆**: 尤其对神经网络 (+0.01 vs +0.002 for trees)
3. **树模型调参收益极小**: LGB leaf/depth 变化 <0.002
4. **跨范式集成 > 同范式堆叠**: 树+MLP+FT > 树×14
5. **阈值调优几乎免费**: 分类任务 5 分钟 +0.01
6. **64 epoch 常优于 128**: 表格数据少训反而泛化好
7. **简单平均 > Stacking**: 当基模型预测高度相关时
