# Playground S6E6 — 完整时间线

> 每天做了什么、提交了什么、舍弃了什么

## Day 1 (6/13)

- [x] EDA：颜色-颜色图、正态检验、Q-Q plots
- [x] 基础特征工程：颜色指数、alpha sin/cos、u_z/g_z
- [x] TDA：GUDHI 持久同调（三类都是实心团块）
- [x] LightGBM/XGBoost/CatBoost 三树对比——LGB 最优
- [x] **S1 提交**: voting_argmax → **LB 0.95662**

## Day 2 (6/14)

- [x] 阈值调优：scipy.optimize 搜 3 阈值
- [x] **S2 提交**: voting_tuned → **LB 0.96670** (+0.010)
- [x] 平衡采样尝试 → ~~LB 0.95423~~ 舍弃
- [x] Pseudo-Labeling → ~~LB 0.96680~~ 无增益 舍弃
- [x] Binary Chain → ~~LB 0.92217~~ 泄漏 舍弃
- [x] 研究 Chris Deotte 公开 notebook，发现特征工程差距

## Day 3 (6/15)

- [x] 移植 Deotte 21 个特征：g/redshift、floor 分箱、delta 分位、TE
- [x] 14 棵树 + 4 MLP 重新训练
- [x] 权重搜索：trees+mlp 最优
- [x] **S3 提交**: trees_only v2 → **LB 0.96760** (+0.0009)
- [x] **S4 提交**: trees+mlp → **LB 0.96815** (+0.0005)
- [x] TabICL 尝试 → OOM 舍弃
- [x] TabM 尝试 → API 不完整 舍弃

## Day 4-5 (6/16-17)

- [x] JaneStreet MLP 架构扫描经验移植（36 组）
- [x] RealMLP (pytabkit) 尝试 → OOF 0.945 远低于树 → 确认通用库不够
- [x] FT-Transformer 训练 → OOF 0.95981 新最强单模
- [x] **S5 提交**: trees+mlp+ft → **LB 0.96844** (+0.0003)
- [x] 手写 RealMLP 消融：4 配置对比
  - 分组 lr/wd +0.0018 唯一有效
  - β₂=0.98、scheduler 无贡献
- [x] FT 消融：5 架构 (A/B/C/D/E)
  - D (regularized) 最优 OOF 0.96117
  - 更深(B)反降 0.0006

## Day 6-7 (6/18-19)

- [x] **服务器灾难**：磁盘满、Python 路径、screen、pip 镜像
  - 整理《云服务器踩坑记录》11 条
- [x] Stacking 尝试 → ~~LB 0.96736~~ 过拟合 舍弃
- [x] 加权平均 → ~~OOF 更低~~ 舍弃
- [x] Binary Chain → ~~LB 0.922~~ 之前已弃

## Day 8 (6/20)

- [x] 精选集成：只保留 BA>0.956 的模型
- [x] **S6 提交**: selective ensemble → **LB 0.96853**

## Day 9 (6/21)

- [x] Deotte notebook 完整移植 → OOF 0.96742 (+0.0076 vs 手写)
- [x] **S7 提交**: Deotte solo → **LB 0.96859**
- [x] **S8 提交**: Deotte + FT/LGB ensemble → **LB 0.96877**
- [x] Deotte 多 seed：s123/s456/s789 全部 ~0.9673-0.9675（极稳）
- [x] 权重网格搜索：D 0.70 + LGB 0.30 最优，FT 权重=0

## Day 10 (6/22)

- [x] **S9 提交**: D 0.70 + LGB 0.30 → **LB 0.96902** (+0.0002)
- [x] LGB 特征选择（去 5 弱特征）→ +0.0002
- [x] **S10 提交**: LGB opt → **LB 0.96911** (+0.00009) 🏆
- [x] Optuna 搜 Deotte 参数 → 15 trials 无增益 → 原参数最优
- [x] 残差 LGB → ~~反降 0.0014~~ 舍弃
- [x] LR Stacker → ~~OOF 0.962 过拟合~~ 舍弃

## Day 11 (6/23)

- [x] smooth_clip → ~~反降 0.00075~~ 舍弃
- [x] flat_ratio 0.1/0.3 → ~~无增益~~ 舍弃
- [x] Deotte clean features（去重）→ OOF 0.96735（≈原值）
- [x] KNN → ~~OOF 0.888 太弱~~ 舍弃
- [x] **S11 提交**: all Deotte seeds + LGB → LB 0.96881（降了）
- [x] 复盘：RETROSPECTIVE + TIMELINE

## 有效提交（11 次）

```
S1   voting_argmax        0.95662   baseline
S2   voting_tuned         0.96670   +0.010  ← 阈值调优
S3   trees_only v2        0.96760   +0.0009 ← 特征工程
S4   trees+mlp            0.96815   +0.0005 ← MLP集成
S5   trees+mlp+ft         0.96844   +0.0003 ← FT集成
S6   selective            0.96853   +0.00009
S7   Deotte solo          0.96859   +0.00006
S8   Deotte ensemble      0.96877   +0.0002
S9   D70+L30              0.96902   +0.0002
S10  LGB opt              0.96911   +0.00009 🏆
S11  all seeds+clean      0.96881   -0.0003  ← 降了
```

## 舍弃清单

```
Binary Chain      OOF 泄漏虚高, LB -0.044
伪标签            98.9%覆盖, 自复制无增益
平衡采样          训练等权扭曲test分布, LB -0.002
Stacking          基模型OOF高度相关, 过拟合
LR Stacker        87维meta特征严重过拟合
加权平均          FT占50%权重, 过拟合
残差LGB           Deotte残差=噪声, 反降
KNN               OOF 0.888 太弱
Optuna树参        搜索未超手动网格
Optuna Deotte参   15 trials原参数最优
smooth_clip       反降 0.00075
flat_ratio        无增益
特征去重          OOF基本不变
```

## 时间分配

```
特征工程+EDA      15%
树模型+tuning      15%
MLP/FT训练+消融    30%
集成+权重搜索      15%
服务器运维         15%
复盘+记录          10%
```
