# S6E6 实验日志

> **Goal**: Deotte 水平（单模 OOF ≥0.969，集成 LB ≥0.97227）
> **当前最佳**: LB 0.96911 | 差距: 0.00316
> **规则**: OOF +≥0.0005 → 提交 Kaggle；3 次无效 → 换方向

## 单模目标

| 模型 | 当前 OOF | 目标 OOF | 差距 | 状态 |
|------|---------|---------|------|------|
| Deotte RealMLP | 0.96742 | **0.96900** | 0.00158 | ⬜ |
| FT-Transformer | 0.95981 | **0.96200** | 0.00219 | ⬜ |
| LightGBM (best) | 0.95864 | **0.95950** | 0.00086 | ⬜ |
| XGBoost (best) | 0.95637 | **0.95800** | 0.00163 | ⬜ |
| TabPFN-3 (new) | — | **0.96000** | — | ⬜ |

> Handcrafted RealMLP 已删除 — 移植版 (0.96742) 远超手写版 (0.95981)，不再追手写版。

## 实验列表

| # | 日期 | 方向 | 假设 | 模型 | OOF before | OOF after | Δ | 决策 |
|---|------|------|------|------|-----------|----------|---|------|
| 1 | 6/23 | 特征工程 V3 | 252 特征(移植cat-v3) → 所有树模型大幅提升 | CB/LGB/XGB | — | — | — | 运行中 |
| 2 | 6/23 | One-vs-Rest | 3个二分类XGB > 1个三分类XGB | XGB-OVR | — | — | — | 运行中 |

## 关键发现

- cat-v3 notebook: 252 特征(含 Flux/Color ratios/Redshift/Sky/Hash combos) + 原始SDSS数据
- 我们的 43 特征 vs 社区 252 特征 = 树模型差距的主因
- 社区 CatBoost 参数: class_weights=[1,3.25,5], bagging_temperature=0.2, max_ctr_complexity=3

## 方向状态

| 方向 | 优先级 | 实验数 | 累计Δ | 状态 |
|------|--------|--------|-------|------|
| A: fold-safe TE | ⭐⭐⭐ | 0 | — | 待做 |
| B: RealMLP 细节补全 | ⭐⭐⭐ | 0 | — | 待做 |
| C: FT-Transformer 改进 | ⭐⭐⭐ | 1 | — | 进行中 |
| D: TabPFN-3 | ⭐⭐ | 0 | — | 待做 |
| E: 新特征探索 | ⭐ | 0 | — | 待做 |

## 已确认无效（不复现）

- Binary Chain: LB -0.044（信息泄漏）
- 平衡采样: LB -0.002
- Pseudo-Labeling: 自复制无增益
- Stacking (LR meta): 过拟合
- 加权平均: OOF更低
- 残差LGB: 反降0.0014
- smooth_clip: 反降0.00075
- KNN: OOF 0.888
