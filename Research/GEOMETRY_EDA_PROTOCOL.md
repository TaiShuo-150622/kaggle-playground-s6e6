# 几何感知 EDA 协议（Geometry-Aware EDA Protocol）

> 数据分析 Agent 的强制性 EDA 步骤。在建模之前，必须先完成以下几何诊断。

---

## 协议总览

```
原始数据
  │
  ├─→ Phase 0: 特征分类（群归属判定）
  │     │
  │     └─→ 每个特征 → 所属群 G → 对偶 Ĝ → 嵌入策略
  │
  ├─→ Phase 1: 几何诊断（拓扑 + 度量 + 曲率）
  │     │
  │     ├─→ 持久同调 → 拓扑复杂度
  │     ├─→ 扩散映射 → 内在维数
  │     └─→ Ollivier-Ricci → 曲率分布
  │
  ├─→ Phase 2: PBLD 嵌入（按群类型分别处理）
  │     │
  │     ├─→ ℝ 加法群 → 标准 PBLD
  │     ├─→ ℝ^+ 乘法群 → Log-PBLD
  │     ├─→ S^1 旋转群 → 圆群 PBLD
  │     ├─→ S² 球面 → 球谐 PBLD
  │     └─→ ℤ_n 循环群 → 正则表示
  │
  ├─→ Phase 3: 几何形变（可选）
  │     │
  │     └─→ Ricci 流 → 均匀化度量
  │
  └─→ Phase 4: 输出 → 几何增强特征矩阵 X_geo
```

---

## Phase 0: 特征分类——群归属判定

### 判定规则

对每个数值特征 f，根据其值域和物理含义分配到一个最可能的群：

```python
def classify_feature_group(feature_values: np.ndarray, feature_name: str) -> str:
    """自动判定特征所属的群类型"""
    vmin, vmax = feature_values.min(), feature_values.max()
    is_cyclic = False
    is_positive = vmin >= 0
    
    # 规则 1: 角度/方位 → S^1（循环群）
    if any(kw in feature_name.lower() for kw in 
           ['alpha', 'angle', 'azimuth', 'ra', 'dec', 'lat', 'lon',
            'longitude', 'latitude', 'theta', 'phi', 'psi', 'deg']):
        return 'S1'  # 圆群
    
    # 规则 2: 球面坐标对 → S^2
    if any(kw in feature_name.lower() for kw in ['delta', 'declination']):
        return 'S2_part'  # 赤纬，需要和赤经配对
    
    # 规则 3: 正实数比值 → ℝ^+（乘法群）
    if is_positive and (
        any(kw in feature_name.lower() for kw in ['ratio', 'div', 'per', 'relative']) or
        '/' in feature_name or '_over_' in feature_name):
        return 'R+'  # 乘法群
    
    # 规则 4: 量级/亮度 → ℝ^+（乘法群）
    if is_positive and any(kw in feature_name.lower() for kw in 
                           ['mag', 'flux', 'magnitude', 'brightness']):
        return 'R+'
    
    # 规则 5: 类别/枚举 → ℤ_n（离散循环群）
    if feature_values.dtype in (np.int32, np.int64) and vmin >= 0:
        return 'Zn'
    
    # 规则 6: 周期时间/序列 → S^1
    if is_cyclic or (vmin == 0 and vmax > 1000):
        # 启发式：检查是否可能是编码的周期
        return 'R'  # 默认 ℝ 加法群
    
    # 默认: ℝ 加法群
    return 'R'
```

### 输出

一张特征群归属表：

| 特征 | 群 G | 对偶 Ĝ | 嵌入策略 | 维度 |
|------|------|--------|---------|------|
| u, g, r, i, z | ℝ | ℝ | 标准 PBLD | 5D/特征 |
| redshift | ℝ^+ | ℝ | Log-PBLD | 5D/特征 |
| alpha | S^1 | ℤ | 圆群 PBLD | 5D/特征 |
| delta | S^1 | ℤ | 圆群 PBLD | 5D/特征 |
| (alpha, delta) | S² | 球谐 | 球谐 PBLD | 16D/对 |
| spectral_type | ℤ_4 | ℤ_4 | 正则表示 | 4D |
| galaxy_population | ℤ_2 | ℤ_2 | 正则表示 | 2D |

---

## Phase 1: 几何诊断

### 1.1 持久同调（拓扑）

```python
from gudhi import RipsComplex
import gudhi.representations as gr

def persistent_homology_diagnostic(X: np.ndarray, max_dim: int = 2):
    """计算持久同调并输出诊断"""
    rips = RipsComplex(points=X)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    diag = st.persistence()
    
    # 诊断
    h0 = [(birth, death) for dim, (birth, death) in diag if dim == 0]
    h1 = [(birth, death) for dim, (birth, death) in diag if dim == 1]
    
    report = {
        'h0_persistence': [d-b for b,d in h0],  # 连通分量持久性
        'h1_persistence': [d-b for b,d in h1],  # 环/洞持久性
        'topological_complexity': 'high' if any(d-b > 1.0 for b,d in h1) else 'low'
    }
    
    # 决策：如果 H1 有长持久条 → 存在环结构 → 需要拓扑特征
    #      如果 H1 全短 → 拓扑平凡 → 线性/傅里叶特征足够
    return report
```

### 1.2 扩散映射（内在维数）

```python
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh

def intrinsic_dimension(X: np.ndarray, k: int = 15):
    """通过扩散映射的谱衰减推断内在维数"""
    # 构造图拉普拉斯
    A = kneighbors_graph(X, k, mode='distance')
    W = (A + A.T) / 2
    W.data = np.exp(-W.data**2 / (2 * W.data.std()**2))
    
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    # 谱分解
    eigvals, _ = eigsh(L, k=min(50, X.shape[0]-2), which='SM')
    eigvals = eigvals[eigvals > 1e-10]
    
    # 拟合 λ_i ∝ i^(-2/d) → log(λ) = -(2/d)·log(i) + const
    i = np.arange(1, len(eigvals)+1)
    slope, _ = np.polyfit(np.log(i), np.log(eigvals), 1)
    d_int = -2 / slope  # 内在维数
    
    return {
        'intrinsic_dimension': d_int,
        'spectral_decay_rate': -slope,
        'recommended_embedding_dim': max(2, int(np.ceil(d_int)))
    }
```

### 1.3 Ollivier-Ricci 曲率

```python
import ot  # POT: Python Optimal Transport

def ollivier_ricci_curvature(X: np.ndarray, k: int = 10):
    """计算每个数据点邻域的 Ollivier-Ricci 曲率"""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, indices = nn.kneighbors(X)
    
    curvatures = []
    for i in range(len(X)):
        # 点 i 的邻域分布: uniform over k neighbors
        mu_i = np.ones(k) / k
        # 最近邻 j 的邻域分布
        j = indices[i, 1]  # nearest neighbor
        mu_j = np.ones(k) / k
        
        # Wasserstein 距离
        M = ot.dist(X[indices[i, 1:]], X[indices[j, 1:]])
        w1 = ot.emd2(mu_i, mu_j, M)
        
        d_ij = dists[i, 1]  # ||x_i - x_j||
        curvatures.append(1 - w1 / d_ij if d_ij > 0 else 0)
    
    kappa = np.array(curvatures)
    return {
        'mean_curvature': kappa.mean(),
        'std_curvature': kappa.std(),
        'negative_fraction': (kappa < -0.1).mean(),  # 需要拉伸的区域
        'positive_fraction': (kappa > 0.1).mean(),   # 可压缩的区域
    }
```

### 诊断报告模板

```
=== 几何诊断报告 ===

拓扑:
  H0 持久性: [0.8, 1.2, 0.9, ...] → 3 个长寿命连通分量
  H1 持久性: [0.1, 0.05, 0.08, ...] → 全短 → 拓扑平凡
  → 无需拓扑特征工程

度量:
  内在维数 d_int ≈ 5.3
  → 数据实际生活在 ~5 维流形上
  → PBLD 嵌入维度建议: 5-8

曲率:
  平均 Ollivier-Ricci: +0.03
  负曲率区域: 12%
  正曲率区域: 8%
  → 曲率接近 0（几乎平坦），Ricci 流收益有限
```

---

## Phase 2: PBLD 嵌入（按群分类）

### 2.1 标准 PBLD（ℝ 加法群）

```python
def pbld_embedding_R(x: torch.Tensor, out_dim: int = 5):
    """标准 PBLD: ℝ → T^out_dim"""
    k = out_dim
    w = nn.Parameter(torch.randn(x.shape[1], k) * 2.33)
    b = nn.Parameter(torch.randn(x.shape[1], k))
    nn.init.uniform_(b, -math.pi, math.pi)
    
    periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * w + b))
    return torch.cat([x.unsqueeze(-1), periodic], dim=-1)  # (B, F, k+1)
```

适用：温度、高度、速度、坐标差、任何无符号限制的实数值。

### 2.2 Log-PBLD（ℝ^+ 乘法群）

```python
def pbld_embedding_Rplus(x: torch.Tensor, out_dim: int = 5, eps: float = 1e-6):
    """Log-PBLD: ℝ^+ → T^out_dim, 等价于 log(x) → 标准 PBLD"""
    log_x = torch.log(x.clamp(min=eps))
    return pbld_embedding_R(log_x, out_dim)
```

适用：比值、流量、浓度、任何正实数。

### 2.3 圆群 PBLD（S^1）

```python
def pbld_embedding_S1(theta: torch.Tensor, n_harmonics: int = 4):
    """S^1 圆群: ℤ 对偶 → 傅里叶级数嵌入"""
    # theta: (B,) 弧度值
    harmonics = []
    for n in range(1, n_harmonics + 1):
        harmonics.append(torch.cos(n * theta.unsqueeze(-1)))
        harmonics.append(torch.sin(n * theta.unsqueeze(-1)))
    return torch.cat(harmonics, dim=-1)  # (B, 2*n_harmonics)
```

适用：角度、方位、周期变量。

### 2.4 球面 PBLD（S²）

```python
def pbld_embedding_S2(theta: torch.Tensor, phi: torch.Tensor, L: int = 3):
    """S² 球面: 球谐函数 Y_l^m 嵌入, 共 (L+1)² 维"""
    from scipy.special import sph_harm
    features = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi.numpy(), theta.numpy())
            features.append(torch.tensor(Y_lm.real, dtype=torch.float32))
    return torch.stack(features, dim=-1)  # (B, (L+1)²)
```

适用：球面坐标对（如天文学中的赤经赤纬）。

### 2.5 离散循环群 PBLD（ℤ_n）

```python
def pbld_embedding_Zn(x: torch.Tensor, n: int):
    """ℤ_n 循环群: 正则表示 → 2D 旋转矩阵特征"""
    # x: (B,) 整数值 ∈ [0, n-1]
    theta = 2 * math.pi * x.float() / n
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (B, 2)
```

适用：离散类别、枚举值、周期整数。

---

## Phase 3: Ricci 流形变（可选）

仅在 Ollivier-Ricci 曲率诊断显示大范围负曲率（>15%）时启用。

```python
def ricci_flow_deformation(X: np.ndarray, k: int = 15, iterations: int = 50, 
                           epsilon: float = 0.01):
    """用离散 Ricci 流拉伸特征空间"""
    from sklearn.neighbors import kneighbors_graph
    
    # 初始化: 距离图
    A = kneighbors_graph(X, k, mode='distance').toarray()
    A = (A + A.T) / 2
    
    for t in range(iterations):
        # 计算每条边的 Ollivier-Ricci 曲率
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if A[i,j] == 0: continue
                # 计算曲率 (简化版)
                N_i = np.argsort(A[i])[:k]
                N_j = np.argsort(A[j])[:k]
                w1 = wasserstein_uniform(N_i, N_j, A)
                kappa_ij = 1 - w1 / A[i,j]
                # Ricci 流更新
                A[i,j] *= (1 - epsilon * kappa_ij)
                A[j,i] = A[i,j]
    
    # 用演化后的度量做谱嵌入
    D = np.diag(A.sum(axis=1))
    L = D - A
    _, V = np.linalg.eigh(L)
    X_ricci = V[:, :X.shape[1]]  # 变换后的特征
    
    return X_ricci
```

---

## Phase 4: Agent 的强制执行规则

### 规则 1: EDA 必须包含几何诊断

```
任何数据分析 Agent 在 fit 第一个模型之前，必须输出:
  - 特征群归属表 (Phase 0)
  - 拓扑复杂度报告 (Phase 1.1)
  - 内在维数估计 (Phase 1.2)
  - 曲率分布报告 (Phase 1.3)
```

### 规则 2: 特征工程首选 PBLD

```
在以下情况下，优先使用 PBLD 而非手工特征:
  1. 特征属于明确的群类型 (ℝ, ℝ^+, S^1, S², ℤ_n)
  2. 内在维数 d_int < 10 (PBLD 嵌入维度不超过 10)
  3. 持久同调显示拓扑平凡 (H1 无长持久条)
```

### 规则 3: 每层决策基于几何诊断

```
如果 内在维数 > 2 × 原始维数 → 数据在高维流形上 → 用深层网络
如果 负曲率区域 > 15% → 数据分布不均匀 → 考虑 Ricci 流或类别权重
如果 H1 有长持久条 → 存在拓扑障碍 → 需要拓扑特征（如 mapper 聚类特征）
如果 内在维数 d_int ≤ 3 → 可以直接用低维 PBLD → 树模型也可能有效
```

### 规则 4: 特征工程必须包含几何证明

```
每个新增特征应标注其几何来源:
  - [R]  来自 ℝ 加法群 PBLD
  - [R+] 来自 ℝ^+ 乘法群 Log-PBLD
  - [S1] 来自 S^1 圆群 PBLD
  - [S2] 来自 S² 球面球谐 PBLD
  - [Zn] 来自 ℤ_n 循环群正则表示
  - [Rf] 来自 Ricci 流形变
  - [E8] 来自 E8 格嵌入
```

---

## 参考实现结构

```
src/
├── geometry/
│   ├── __init__.py
│   ├── classify.py          # Phase 0: 特征群归属判定
│   ├── topology.py          # Phase 1.1: 持久同调
│   ├── diffusion.py         # Phase 1.2: 扩散映射 + 内在维数
│   ├── curvature.py         # Phase 1.3: Ollivier-Ricci 曲率
│   ├── pbld.py              # Phase 2: 五种群的 PBLD 嵌入
│   ├── ricci.py             # Phase 3: Ricci 流形变
│   └── report.py            # Phase 4: 几何诊断报告生成
├── features/
│   └── shared.py            # 现有特征工程（逐步迁移到 PBLD）
└── models/
    ├── pbld_transformer.py  # PBLD + Transformer
    └── realmlp_v3.py        # 现有 PBLD 实现
```
