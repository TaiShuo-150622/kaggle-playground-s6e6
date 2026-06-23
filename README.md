# Kaggle Playground S6E6 — Predicting Stellar Class

> **LB 0.96911** | 25 天 | 30+ 模型 | 树 + MLP + FT-Transformer  
> 完整复盘: [`JOURNEY.md`](JOURNEY.md)

## 项目结构

```
├── README.md                          # 本文件
├── JOURNEY.md                         # 完整复盘：每一步的思考和提升
├── NOTES.md                           # 比赛笔记 (特征/TDA/模型对比)
│
├── code/
│   ├── features/                      # 特征工程
│   │   ├── full_pipeline.py           #   原始数据 → 全部特征 (颜色/比值/TE/分箱)
│   │   ├── add_deotte_features.py     #   Deotte 特征单独测试
│   │   └── quick_test_new_features.py #   快速A/B测试
│   │
│   ├── models/                        # 模型训练
│   │   ├── server_ensemble.py         #   18模型集成 (LGB×5+XGB×5+CB×4+MLP×4)
│   │   ├── train_ft_transformer.py    #   FT-Transformer 训练
│   │   ├── train_handcrafted_realmlp.py # 手写 RealMLP (Deotte R2-103)
│   │   ├── ablation_realmlp.py        #   RealMLP 消融实验
│   │   ├── ablation_ft.py             #   FT-Transformer 消融实验
│   │   ├── server_run.py              #   多GPU并行训练脚本
│   │   ├── run_realmlp.py             #   RealMLP 快速测试
│   │   └── realmlp_test.py            #   RealMLP MPS测试
│   │
│   ├── ensemble/                      # 集成策略
│   │   ├── threshold_tuning.py        #   阈值调优 (scipy.optimize)
│   │   ├── full_pipeline_parallel.py  #   CPU+GPU并行集成
│   │   ├── binary_chain.py           #   Binary Chain 尝试 (失败)
│   │   └── run_all.py                #   全流程: 平衡采样+伪标签+阈值
│   │
│   └── analysis/                      # 分析工具
│       ├── tda_mapper.py              #   GUDHI TDA 持久同调
│       ├── tabicl_test.py             #   TabICL 零训练测试
│       ├── tabicl_mps.py              #   TabICL MPS GPU测试
│       ├── tabicl_quick.py            #   TabICL 快速验证
│       └── tabicl_per_class.py        #   TabICL 单类分析
│
├── eda/                               # EDA 可视化
│   ├── eda_float64_dist.png           #   浮点特征分布
│   ├── eda_plots.png                  #   颜色-颜色图
│   ├── eda_qq_plots.png              #   Q-Q正态检验
│   ├── eda_log_transform.png          #   Log变换效果
│   ├── eda_spatial_color.png          #   空间分布
│   └── eda_tda_all.png               #   TDA持久图汇总
│
└── data/                              # 数据文件 (.gitignored)
    ├── train.csv / test.csv           #   原始比赛数据
    ├── train_fe.csv / test_fe.csv     #   加基础特征后
    └── sample_submission.csv          #   提交模板
```

## 关键结果

| 阶段 | 方法 | LB | OOF |
|------|------|:---:|:---:|
| Baseline | LGB+XGB+CB voting | 0.95662 | 0.9565 |
| + 阈值调优 | scipy.optimize 3阈值 | 0.96670 | 0.9656 |
| + Deotte特征 | 21个新特征 | 0.96760 | 0.9642 |
| + MLP集成 | pytabkit RealMLP | 0.96815 | 0.9642 |
| + FT-Transformer | attention系 | 0.96844 | 0.9650 |
| + Deotte RealMLP | notebook完整移植 | 0.96877 | 0.9674 |
| + LGB多seed | 3 seed平均 | 0.96902 | 0.9679 |
| + LGB特征选择 | 去5个噪声特征 | **0.96911** | 0.9681 |

## 运行

```bash
# 特征工程 + 18模型训练
python3 code/features/full_pipeline.py

# FT-Transformer 训练
python3 code/models/train_ft_transformer.py

# 手写 RealMLP
python3 code/models/train_handcrafted_realmlp.py
```
