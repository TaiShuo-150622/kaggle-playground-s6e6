# Playground S6E6 完整复盘

> LB 0.96911 | ~25 天 | 40+ 个实验 | 30+ 个模型

## 阶段 0：理解问题 (Day 1)

**数据**：577K 行 × 12 列，三分类（GALAXY/STAR/QSO），指标 Balanced Accuracy。

**物理背景**：SDSS 巡天的 ugriz 五波段测光数据。GALAXY 年老偏红，QSO 热而偏蓝，STAR 居中。颜色-颜色图上三类自然分离。

**首个发现**：log 变换无意义（星等本身是对数尺度）。18 个 float 特征无一正态，但对树模型无关。

## 阶段 1：树模型基线 + 阈值调优 (Day 1-4)

### 三树对比
LightGBM > XGBoost > CatBoost。BA 差异仅 0.0026。换模型拉不开差距。

### 阈值调优 — 第一个有效突破 (+0.0091)
```
argmax(prob)           BA = 0.9565
调阈值 (scipy优化)     BA = 0.9656  (+0.0091)
最优阈值: GALAXY=2.14, STAR=0.81, QSO=0.53
```
**洞察**：balanced accuracy 关心各类 recall，不是 argmax。

### 失败记录
- Binary Chain：OOF 泄漏虚高，LB -0.044
- 平衡采样：训练时三等权，test 真实分布被扭曲，LB -0.002
- Pseudo-Labeling：98.9% test 被标上，等于自复制，无增益

### TDA 分析 (GUDHI)
三类天体在特征空间中都是实心团块 (H₁=0)。Bottleneck 距离：GALAXY↔STAR=1.24 最大。

## 阶段 2：研究公开方案 (Day 5-7)

通过 Chris Deotte 的 notebook 发现了关键差距：

| 他有的 | 我们缺的 |
|--------|---------|
| 手写 RealMLP (CV 0.96904) | pytabkit (0.945) |
| 21 个 Deotte 特征 | 只做了基础颜色 |
| fold-safe target encoding | 全局 TE（信息泄漏） |
| TabM, FT-Transformer | 没试过 |

**认知转变**：不是模型不够多——是 MLP 的能力被严重低估。Deotte 单模 0.969 > 我们整个 ensemble 0.9656。

## 阶段 3：特征工程升级 (Day 8-10)

移植 Deotte 的 21 个新特征：g/redshift 比值、floor 分箱、delta 分位数、target encoding。

效果：MLP 从 0.945→0.9556 (+0.01)，树 +0.002。

LB 从 0.96670→0.96760。

## 阶段 4：扩展模型范式 (Day 11-15)

### FT-Transformer — 新最强单模
OOF 0.95981。attention 机制自发学到 redshift×color 的交互。

### 手写 RealMLP — 消融实验
4 配置对比：分组 lr/wd (+0.0018) 是唯一有效改进，β₂=0.98 和 scheduler 几乎无贡献。

### 14 棵树 + 4 MLP + 2 FT → 集成
LB 0.96844。

## 阶段 5：Deotte RealMLP (Day 16-20)

最关键的突破——**直接移植他的完整代码**。

```
手写版本:  OOF 0.95981
移植版本:  OOF 0.96742  (+0.0076)
LB:        0.96877
```

**差距来源**：fold-safe TE、robust scaling 而不是 StandardScaler、PBLD PReLU 激活。每个细节 0.002-0.003。

## 阶段 6：多 seed + 特征选择 (Day 21-23)

- LGB 多 seed 集成 → LB 0.96902
- LGB 特征选择（去噪声列）→ LB 0.96911
- 权重搜索：D 0.70 + LGB 0.30 最优，FT 权重 0——MLP 和 FT 犯错重叠

## 阶段 7：穷举验证 (Day 24-25)

- Optuna 搜参：15 trials 未超手动参数
- smooth_clip：反降 0.00075
- flat_ratio：无增益
- KNN：OOF 0.888 太弱
- LR Stacker：OOF 0.962 过拟合
- 残差 LGB：反降 0.0014

## 提升全景

```
S2 baseline             0.96670
+ 阈值调优              +0.009    → LB不可见 (OOF增益)
+ Deotte 特征           0.96760   +0.0009
+ MLP 集成              0.96815   +0.0005
+ FT-Transformer        0.96844   +0.0003
+ Deotte RealMLP        0.96877   +0.0003
+ LGB 多seed            0.96902   +0.0002
+ LGB 特征选择          0.96911   +0.00009
─────────────────────────────────────────
总提升                            +0.00241 (LB)
                                    +0.0091  (OOF阈值)
```

## 核心教训

1. **特征工程是最大杠杆** — MLP 对好特征的敏感度是树的 5 倍
2. **手写 MLP >> 通用库** — pytabkit 默认值只到 0.945，手写到 0.967
3. **简单平均 > Stacking** — 同范式模型的 OOF 高度相关时，学权重就是学噪声
4. **阈值调优几乎免费** — 5 分钟 +0.009
5. **多 seed 集成降方差** — 增量小但累积有效
6. **树模型调参收益极低** — Optuna/网格/multi-seed 对树的增益 <0.002
7. **基础设施先行** — 磁盘、Python 路径、screen 保活，优先级高于调参
8. **fold-safe 编码是必须的** — 全局 TE 泄漏信息，CV 虚高 LB 翻车

## 未跨越的鸿沟

排行榜上 53 人共享 0.97227——来自 Deotte 公开 notebook 直接提交。我们从头训到 0.96911，差 0.0032。这个差距在：
- 他的 pipeline 运行 20+ 个模型变体
- 每个细节都经过反复验证
- 我们的资源是课程项目级别

0.96911 是方法论和时间的合理回报。
