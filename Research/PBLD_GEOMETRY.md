# PBLD 的代数基础与几何推广

> 从 S6E6 特征工程 → Bochner 定理 → 群表示论 → Ricci 流 → E8 格

---

## 一、PBLD 的本质：Bochner 定理的随机逼近

### 1.1 Bochner 定理（1932）

一个连续函数 k: ℝ → ℂ 是正定核（即存在特征映射 φ 使得 k(x,y) = ⟨φ(x), φ(y)⟩）**当且仅当**它是某个正有限 Borel 测度 μ 的傅里叶变换：

```
k(x - y) = ∫_ℝ e^{2πi·ω·(x-y)} dμ(ω)
```

### 1.2 PBLD 作为谱核的蒙特卡洛近似

PBLD 嵌入：`e(x) = [cos(2π·w_j·x + b_j)]_{j=1}^k`

```
⟨e(x), e(y)⟩ = Σ_j cos(2π·w_j·x + b_j)·cos(2π·w_j·y + b_j)
             = ½ Σ_j [cos(2π·w_j·(x-y)) + cos(2π·w_j·(x+y) + 2b_j)]
```

当 b_j ~ Uniform(0, 2π) 时，第二项期望为零。第一项是 Bochner 积分在频率 {w_j} 处的离散采样。

**结论**：PBLD 不是特设的——它是任意平移不变核的谱表示在 k 个随机频率上的蒙特卡洛近似。训练过程中 w_j 被学习 = 从数据中学习最优谱测度 μ。

### 1.3 实证证明

在 Playground S6E6 上：
```
CatBoost + 252 手工特征 → OOF 0.96642
RealMLP(PBLD) + 60 原始特征 → OOF 0.96742
```

PBLD 用更少的特征超越了手工构造的 150+ bins/hash/combos。

---

## 二、群理论视角：Pontryagin 对偶

### 2.1 一般框架

对于任何局部紧 Abel 群 G，Pontryagin 对偶定理给出：

```
Ĝ = Hom(G, U(1))  (连续群同态)
```

G 的所有一维酉表示 = {χ_ξ: G → U(1) | ξ ∈ Ĝ}。

PBLD 的泛化：**从 Ĝ 上的测度 μ 中采样 k 个特征标 {χ_ξⱼ}，构造嵌入**：

```
e_G(g) = [Re(χ_ξⱼ(g))]_{j=1}^k = [cos(ξ_j(g) + b_j)]_{j=1}^k
```

### 2.2 不同群的嵌入

| 群 G | 对偶 Ĝ | 嵌入空间 | 特征标 | 适用数据类型 |
|------|--------|---------|--------|------------|
| ℝ（加法）| ℝ | T^k（k 维环面）| e^{2πi·ω·x} | 温度、星等、红移 |
| ℝ^+（正实数乘法）| ℝ | T^k | ω^{iρ} = e^{iρ·log ω} | 比值、流量比 |
| S¹（旋转）| ℤ | S¹ | e^{inθ} | 角度、方位角 |
| S²（球面）| 不可约表示指标 l∈ℕ₀ | 球谐 | Y_l^m(θ,φ) | 天球坐标 |
| ℤ_n（离散循环）| ℤ_n | n 次单位根 | e^{2πi·k·x/n} | 周期类别 |
| G₁×G₂（直积）| Ĝ₁×Ĝ₂ | T^{k₁+k₂} | 直积 | 多类型组合 |

### 2.3 对 S6E6 数据的直接应用

```
1. 星等 u,g,r,i,z        → ℝ 加法群      → 标准 PBLD ✅
2. 红移 redshift         → ℝ^+ 乘法群    → log + PBLD
   （我们手工做的 log1p_redshift 是这个的退化版）
3. 赤经α + 赤纬δ         → S² 球面       → 球谐 PBLD
   （手工的 alpha_sin/cos 只是 l=1 的特例，球谐有 (L+1)² 个基函数）
4. 光谱型 + 星系族       → 离散群        → 正则表示
```

---

## 三、注意力机制的代数解释

### 3.1 为什么线性嵌入导致注意力退化

论文 2509.20942 的结论在代数上可以严格表述：

线性嵌入 `φ(x) = W·x` 定义了 ℝ 到自身的线性映射。两个嵌入向量的点积：

```
⟨Wx, Wy⟩ = xᵀWᵀWy
```

WᵀW 是一个半正定矩阵，其特征值决定了度量结构。当 W 随机初始化时，WᵀW ≈ I（等距），点积退化为原始点积。**线性嵌入无法改变数据的度量结构**。

### 3.2 PBLD 如何修复

PBLD 的点积在代数上等价于：

```
⟨e_PBLD(x), e_PBLD(y)⟩ = ½ Σ_j cos(2π·w_j·(x-y))
                        = ½ Σ_j Re(χ_w_j(x-y))
```

这等价于在 G 的**对偶群 Ĝ 上取了 k 个采样点**，用特征标 χ_w_j 来测量 x 和 y 的"结构距离"。

```
线性嵌入: ⟨Wx, Wy⟩ = xᵀWᵀWy           (只有 O(d²) 种交互)
PBLD嵌入: Σ_j cos(w_j(x-y))           (相当于 O(k) 个不同尺度的距离核)
```

---

## 四、观测：从数据推断内在几何

### 4.1 三级推断框架

给定无标签数据 X ∈ ℝ^{n×d}：

**拓扑级（持久同调）**：
```
数据点 → Vietoris-Rips 复形 → 持久同调 → 条形码
S6E6 结果：H₁ = 0（无环/洞），H₀ 持久 → 三类孤立团块
→ 数据拓扑平凡，无必要用拓扑特征工程
```

**度量级（扩散映射）**：
```
k-NN 图 → L = D^(-1/2)·W·D^(-1/2) → 解 Lψ = λψ
λ_i ∝ i^(-α)，α ≈ 2/d → 从衰减速率推断内在维数 d_int
ψ_i 是数据的内在坐标（替代 PCA 的非线性版本）
```

**曲率级（Ollivier-Ricci 曲率）**：
```
对边 (x,y)：κ(x,y) = 1 - W₁(μ_x, μ_y)/d(x,y)
κ > 0 → 正弯曲（同类密集区，可压缩）
κ < 0 → 负弯曲（类别边界，需更多特征）
```

### 4.2 曲率指导特征工程

如果某个区域的 Ollivier-Ricci 曲率 κ < -0.1，该区域的特征空间是"拉伸不足"的——这意味着两个类在这个子空间中纠缠。**应该在该区域加更多非线性特征**（如针对性地做多项式交互或额外的 PBLD 频率）。

---

## 五、形变：用 Ricci 流拉伸特征空间

### 5.1 思想

离散 Ricci 流（Hamilton, 1982 / Chow-Luo, 2003）：

```
dg_ij/dt = -2·Ric_ij
```

正曲率区域膨胀（数据点被推开），负曲率区域收缩（数据点被拉近）→ 最终达到 Einstein 度量（曲率均匀）。

### 5.2 算法

```python
# 1. 构造加权 k-NN 图
A = kneighbors_graph(X, k=15, mode='distance')

# 2. 对每条边计算 Ollivier-Ricci 曲率
for edge (i,j):
    κ_ij = 1 - wasserstein_distance(N_i, N_j) / d(x_i, x_j)

# 3. 离散 Ricci 流迭代
for t in range(T):
    for edge (i,j):
        w_ij *= (1 - ε * κ_ij)    # κ > 0 → 增权（拉开）
                                   # κ < 0 → 减权（拉近）
    
# 4. 用演化后的图拉普拉斯做谱嵌入
L_evolved = D_evolved - A_evolved
_, V = eigh(L_evolved)
X_transformed = V[:, :k]  # 前 k 个特征向量
```

### 5.3 对分类的意义

Ricci 流后的空间具有均匀曲率 → 同一类的点自然聚集，不同类的点自然分离。**等价于在几何层面做类别平衡 + 特征解耦**，不需要重采样或类权重。

---

## 六、E8 作为普遍先验

### 6.1 为什么 E8

E8 是最大的例外单 Lie 群（维数 248，根系统 240 个根向量），具有最优的泛性质：

| 性质 | 含义 |
|------|------|
| **最优球堆积** | E8 格在 8 维空间中达到最大堆积密度 π⁴/384 ≈ 0.254 |
| **Killing 形式** | 非退化对称双线性形式 → 自然正定核 |
| **240 个根向量** | 均匀分布在 7 维球面 S⁷ 上 → 最优各向同性离散化 |
| **自对偶** | E8 = (E8)^*，格点同时是采样点和表示指标 |

### 6.2 保守方案：E8 格嵌入

```python
# 1. 将 d 维特征投影到 8 维
X_8d = PCA(n_components=8).fit_transform(X)

# 2. 量化到 E8 格
X_e8 = quantize_to_e8(X_8d)

# 3. 用 240 个根方向作为自然基
features = X_e8 @ E8.roots.T  # (n, 240)
```

数学含义：用已知宇宙中最优的 8 维球堆积来离散化特征空间。240 个根方向 = 240 个"自然特征"。

### 6.3 激进方案：E8 根作为超完备 PBLD 基

```python
class E8PBLD(nn.Module):
    def __init__(self, d_features):
        # 每个特征 → 8 维 E8 空间
        self.embed = nn.Parameter(torch.randn(d_features, 8))
        # 240 个 E8 根向量（固定）
        self.roots = e8_roots()  # (240, 8)
    
    def forward(self, x):
        # 各特征投影到 E8 空间
        e = x @ self.embed          # (B, 8)
        # 在 240 个根方向上的投影
        proj = e @ self.roots.T     # (B, 240)
        # 傅里叶化
        return torch.cat([e, torch.cos(proj)], dim=-1)  # (B, 248)
```

训练过程中 `self.embed` 被学习 → 模型在 E8 的 240 个根方向中选择最优投影方向。**等价于在 E8 的根系统上做 PBLD**。

### 6.4 为什么 8 维

E8 只在 8 维是"最优"的。如果数据内在维数不是 8：

- **d_int < 8** → PCA 升维到 8（无信息损失但有冗余）
- **d_int > 8** → 先降维到 8（有信息损失），或使用 E8 的张量积分解

8 维恰好是神经网络中间层的常用维数——可以在任何 8 维 hidden layer 上施加 E8 先验。

---

## 参考文献

1. Bochner, S. (1932). *Vorlesungen über Fouriersche Integrale.* — Bochner 定理
2. Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines.* — 随机傅里叶特征
3. Hamilton, R. (1982). *Three-manifolds with positive Ricci curvature.* — Ricci 流
4. Chow, B. & Luo, F. (2003). *Combinatorial Ricci flows on surfaces.* — 离散 Ricci 流
5. Ollivier, Y. (2009). *Ricci curvature of Markov chains on metric spaces.* — Ollivier-Ricci 曲率
6. Coifman, R. & Lafon, S. (2006). *Diffusion maps.* — 扩散映射
7. Conway, J. & Sloane, N. (1999). *Sphere Packings, Lattices and Groups.* — E8 格
8. Liang, Z. et al. (2025). *Why Attention Fails.* arXiv:2509.20942 — 线性嵌入与注意力退化
9. Deotte, C. *RealMLP v5 for S6E6.* Kaggle Notebook — PBLD 实现
