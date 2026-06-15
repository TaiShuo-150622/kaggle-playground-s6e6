# Playground S6E6 — Predicting Stellar Class

> 比赛: Kaggle Playground Series S6E6  
> 任务: 三分类 (GALAXY / STAR / QSO)  
> 数据: 577k train, 247k test, 12 features  
> 指标: Balanced Accuracy  
> 截止: 2026-06-30

---

## 数据概览

| 列 | dtype | unique | 说明 |
|----|-------|--------|------|
| `id` | int64 | 577k | 行号 |
| `alpha` | float64 | 440k | 赤经 [0,360) |
| `delta` | float64 | 444k | 赤纬 [-18,79] |
| `u,g,r,i,z` | float64 | 577k | 5波段测光星等 |
| `redshift` | float64 | 574k | 红移 (STAR≈0, QSO>1) |
| `spectral_type` | object | 4 | M / G-K / A-F / O-B |
| `galaxy_population` | object | 2 | Red_Sequence / Blue_Cloud |
| `class` | object | 3 | GALAXY(65%) / STAR(14%) / QSO(20%) |

- 零缺失值
- 合成数据（基于 SDSS），有分布噪声
- STAR 和 QSO 在 redshift 上有重叠（synthetic 合成时引入）

---

## 特征工程

### 颜色指数（物理意义明确，最有效）

```python
u_g = u - g          # 紫外-绿，正值=偏红
g_r = g - r          # 绿-红
r_i = r - i          # 红-近红外
i_z = i - z          # 近红外-红外
u_r = u - r          # 跨波段
g_i = g - i
r_z = r - z
color_curv = u_g - g_r  # 颜色曲率
```

GALAXY 在所有颜色指数上偏红，QSO 偏蓝，STAR 居中。

### alpha 循环编码

```python
alpha_sin = sin(alpha * pi / 180)
alpha_cos = cos(alpha * pi / 180)
# 删除原始 alpha — sin+cos 已完整编码圆上方向
```

### 不需要做的

- log 变换: 星等本身就是对数尺度，二次 log 无意义
- 标准化: 树模型不需要
- 缺失值填充: 数据无缺失

---

## 三类树模型深度对比

### 树的生长方式

| | XGBoost | LightGBM | CatBoost |
|---|---------|----------|----------|
| **生长** | level-wise (按层) | leaf-wise (挑最大增益叶子) | symmetric (全层同split) |
| **树形** | 对称，每层全满 | 不对称，深的很深 | 完全对称 |
| **速度** | 中 | **最快** | 最慢 |
| **过拟合** | 低 | 高 (需调 num_leaves) | 最低 |

### 关键差异

```
XGBoost:  二阶泰勒展开 → Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - ...] - γ
          正则化: L1(γ剪枝) + L2(λ)

LightGBM: GOSS (梯度大→全保留, 梯度小→采样) + 直方图(255 bins)
          直方图减法: G_R = G_total - G_L (不用重扫)

CatBoost: Ordered Boosting (每个点只用历史数据算残差)
          Ordered Encoding (类别特征防泄漏)
          Bayesian bootstrap (默认)
```

### 类别特征

| | XGBoost | LightGBM | CatBoost |
|---|---------|----------|----------|
| 支持 | ❌ 需手动编码 | ⚠️ 简单排序 | ✅ Ordered Encoding |
| 防泄漏 | N/A | ❌ 可能 | ✅ 天然防 |

### GPU

| | XGBoost | LightGBM | CatBoost |
|---|---------|----------|----------|
| GPU效果 | 最好 (`gpu_hist`) | 一般 | 最弱 |
| 何时用 | depth>12 或 n>5000轮 | 大数据集 | 特征>50 |

**这个比赛不需要 GPU**：depth=8, 19 特征, 57 万行 → CPU 已足够。

### 结果对比 (5-Fold CV)

| | BA | GALAXY | STAR | QSO | 时间/fold |
|---|------|--------|------|------|------|
| XGBoost | 0.9556 | 0.9782 | 0.9632 | 0.9253 | 174s |
| LightGBM | 0.9565 | 0.9785 | 0.9651 | 0.9259 | 66s |
| CatBoost | TBD | TBD | TBD | TBD | TBD |

**三类差距极小 (<0.1% BA)**，真正拉开差距的不是换模型，是以下策略。

---

## TDA 分析结论

- 三类天体在特征空间中都是**实心团块**（H₁=0, 无环/洞）
- 颜色空间中 GALAXY 与其他两类拓扑距离大（bottleneck=1.24）
- 加入 redshift 后三类均匀分离
- Mapper 在高维失效（DBSCAN + curse of dimensionality）

---

## 正态分布检验

**18 个 float 特征中 0 个呈正态分布。** redshift 严重右偏(skew=2.3)，alpha_sin/cos 接近均匀。**对树模型完全无影响。**

---

## 提升方向（按优先级）

### 1. Threshold Tuning ⭐⭐⭐
```
不选 argmax(prob)，而是调三类阈值优化 BA
预计 +0.3~0.8%
```

### 2. Pseudo-Labeling ⭐⭐⭐
```
高置信度 test 样本 → 带伪标签回训练集 → 重训
预计 +0.5~1.0%
```

### 3. 类别加权采样 ⭐⭐⭐
```
STAR(14%) QSO(20%) 被 GALAXY(65%) 压制
欠采样/过采样/weight 等权
预计 +0.5~1.5% on QSO recall
```

### 4. Adversarial Validation ⭐⭐
```
训练二分类器区分 train vs test
若 AUC>0.6 → train/test 分布偏移 → 修正特征
```

### 5. 三层 Stacking ⭐⭐
```
11个基学习器 → OOF预测 + 原始特征 → LR meta-learner
预计 +0.5~1.0%
```

### 6. 换个角度建模 ⭐
```
回归 redshift → 用 z 推断 class
先二分类(STAR/非STAR) → 再细分(GALAXY/QSO)
```

---

## 项目结构

```
playground_s6e6/
├── data/
│   ├── train.csv              # 原始训练集
│   ├── test.csv               # 测试集
│   ├── sample_submission.csv  # 提交模板
│   ├── train_fe.csv           # 加特征后的训练集
│   └── test_fe.csv            # 加特征后的测试集
├── tda_mapper.py              # GUDHI TDA 分析
├── NOTES.md                   # 本文档
├── eda_*.png                  # EDA 可视化
└── submission_*.csv           # 提交文件
```

---

## Bug 记录

> 每个 bug 都是「下次不会再犯」的经验。

### Bug 1: LightGBM 需要 eval_set 才能用 early_stopping
```
ValueError: For early stopping, at least one dataset and eval metric is required
```
**原因**: LightGBM 的 `early_stopping` callback 需要验证集来计算什么时候停。
**修复**: 始终传 `eval_set=[(X_val, y_val)]` 到 `model.fit()`。
**教训**: 三棵树的 early_stopping 传参方式不同——
```python
LGBM:  fit(eval_set=[(X_val,y_val)], callbacks=[lgb.early_stopping(50)])
XGBoost: fit(eval_set=[(X_val,y_val)], early_stopping_rounds=50, verbose=False)
CatBoost: fit(eval_set=[(X_val,y_val)], early_stopping_rounds=50, verbose=0)
```

### Bug 2: CatBoost `subsample` 与默认 `bootstrap_type='Bayesian'` 冲突
```
CatBoostError: default bootstrap type (bayesian) doesn't support 'subsample' option
```
**原因**: Bayesian bootstrap 天然处理采样，不接受显式 `subsample` 参数。
**修复**: 设 `bootstrap_type='Bernoulli'`。
**教训**: CatBoost 参数和其他两棵树有微妙差异。需要设:
```python
CatBoostClassifier(bootstrap_type='Bernoulli', subsample=0.8)
# 而非直接 CatBoostClassifier(subsample=0.8)
```

### Bug 3: LGBMClassifier.fit() 不接受 verbose 参数
```
TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'verbose'
```
**原因**: LightGBM sklearn wrapper 的 `verbose` 在构造函数里（`LGBMClassifier(verbose=-1)`），不在 `fit()` 里。
**修复**: `LGBMClassifier(verbose=-1)`，fit 里不传 verbose。

### Bug 4: GUDHI 3.12 的 Mapper 类名变动
```
ImportError: cannot import name 'MapperGraph' from 'gudhi'
```
**原因**: GUDHI 3.12 用 `CoverComplex` 而非 `MapperGraph`（版本间 API 变动）。
**修复**: 用 `gudhi.CoverComplex` + 手动实现 cover 分箱逻辑。

### Bug 5: pip3 vs python3 路径不一致
```
ModuleNotFoundError: No module named 'kagglehub'
# but: pip3 install kagglehub → "already installed"
```
**原因**: macOS 上有多个 Python——`/usr/local/bin/python3` 和 `/opt/anaconda3/bin/python3`。pip3 装到了 Anaconda 但系统 python 找不到。
**修复**: 全程用 `/opt/anaconda3/bin/python3` 和 `/opt/anaconda3/bin/pip3`。

### Bug 6: set() 导致列索引错位
```python
X_all = train[list(set(feat_A) | set(feat_B))].values  # set 无序!
idx_A = [list(set(feat_A)).index(c) for c in feat_A]   # 这个 set 也是另一个无序的
```
**原因**: Python 的 `set()` 是哈希序，每次运行顺序不同。两个不同 set 的 index 无法对应同一列。
**修复**: 用有序 list，不用 set。或者取交集/并集后立即转为 sorted list 固定顺序。

---

## Kaggle 公开方案中我们没涉及的技巧

### 1. RealMLP（NeurIPS 2024）— 表格数据的专用神经网络

Chris Deotte (4×Grandmaster) 和 Vladimir Demidov 大量使用。

**核心**: 三层 MLP (256 neurons each)，经过精心调优的默认超参数，**不经调参就能和 GBDT 竞争**。

**关键设计**:
- **PBLD embeddings**: 每个数值特征映射到 4 维 `(x, W₂cos(2π·w₁·x + b₁) + b₂)`
- **Robust Scaling**: 中心化在中位数，IQR 缩放，smooth clipping 到 (-3, 3)
- **Parametric activations**: `σ_α(x) = (1-α)·x + α·SELU(x)`，每神经元可学习的 α
- **Neural Tangent Parametrization**: `z = d^(-1/2) · W · x + b`
- **AdamW**: β₂=0.95（比默认 0.999 更快适应）

**使用方式**: 
```python
from pytabkit import RealMLP_TD_Classifier
clf = RealMLP_TD_Classifier(device='cpu')  # 或 'cuda'
clf.fit(X_train, y_train)
```

**为什么我们没提到**: 以为「NN 在表格数据上不如树模型」。RealMLP 证明了这个假设是错的——当超参数调对时，MLP 可以和 LightGBM 平起平坐。

### 2. TabPFN-3（Nature 2025 / May 2026）— 表格基础模型

Philipp Singer 的两篇 notebook 都用了这个。

**核心**: 在 1.3 亿个合成表格数据集上预训练的 Transformer。**零训练**——直接 forward pass 输出预测。

**特性**:
- 单次 forward pass，<0.2ms/预测（100万行）
- 天然处理缺失值、异常值、不平衡
- 在 <10,000 样本的数据集上显著优于树模型

**限制**: 对 57 万行 × 19 特征的数据，TabPFN-3 的 ICL 上下文长度可能不够（需要 large-data checkpoint）。

### 3. GPU Logistic Regression Stacker（Chris Deotte）

**核心**: 三层 stacking，用 cuML GPU 加速。

```
Level 1: 75-500 个基学习器（各种参数组合的 GBDT + NN + SVR）
         → 每行变成 75-500 维的 OOF 预测向量

Level 2: 用前向特征选择筛选 Level 1 输出
         + 构造 meta-features（std, mean, confidence）
         → GBDT/NN 做 meta-learner

Level 3: 简单加权平均 Level 2 的多个 meta-learner
```

**和我们的区别**: 我们只是 3 模型简单 voting。Deotte 的 stacking 用了「不同问题表述」来创造多样性（比如预测 ratio、残差、缺失特征），让 Level 1 的 500 个模型相互正交。

### 4. Binary Chain（Mehran Kazeminia）

**核心**: 把三分类拆成两个二分类的顺序链:

```
Step 1: STAR vs 非STAR (z≈0 的天然容易分)
Step 2: GALAXY vs QSO   (用颜色+z 区分)
```

这就是我们讨论过的「先二分类再细分」，但 Kazeminia 的实现是**链式的**——Step 1 的预测概率作为 Step 2 的额外特征。

### 5. 多模型 Blending 技巧

多个 notebook 展示的 blend 模式:

| 做法 | 效果 |
|------|------|
| RealMLP + GBDT 概率平均 | 互补误差，通常 BA +0.3~0.5% |
| 不同 seed 的模型平均 | 降低方差 |
| 不同特征子集的模型平均 | 增加多样性 |
| TabPFN 作为额外一票 | 零训练成本加入 ensemble |

### 6. HistGradientBoostingClassifier (sklearn)

Ákos Pintér 的 baseline 用了 sklearn 的 `HistGradientBoostingClassifier`——这是 sklearn 内置的 LightGBM 等价物，支持缺失值、分类特征，且纯 CPU 不需额外安装。

---

## 相关记忆

- [[kaggle-full-plan-2026]] — 全部 9 场比赛计划
- [[../shared/tree_models_guide]] — 树模型通用知识
