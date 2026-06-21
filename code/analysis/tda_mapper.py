"""
TDA 分析 — 三类天体在特征空间中的拓扑结构
GUDHI CoverComplex (Mapper) + RipsComplex (Persistence)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import gudhi
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 数据 + 参数
# ============================================================
train = pd.read_csv("data/train_fe.csv")
color_cols   = ['u_g', 'g_r', 'r_i', 'i_z']
ext_cols     = ['u_r', 'g_i', 'r_z', 'color_curv']
redshift_col = ['redshift']
mag_cols     = ['u', 'g', 'r', 'i', 'z']
pos_cols     = ['alpha_sin', 'alpha_cos', 'delta']

FEATURE_SETS = {
    'A_colors':           color_cols,
    'B_colors_z':         color_cols + redshift_col,
    'C_colors_ext':       color_cols + ext_cols,
    'D_colors_ext_z':     color_cols + ext_cols + redshift_col,
    'E_physical':         color_cols + ext_cols + redshift_col + mag_cols,
    'F_full':             color_cols + ext_cols + redshift_col + mag_cols + pos_cols,
}

SAMPLE_PER_CLASS = 500
RANDOM_SEED = 42
MAX_EDGE_LENGTH = 5.0

# ============================================================
# 1. Mapper (CoverComplex)
# ============================================================
def mapper_graph(data, filter_vals, n_intervals=10, overlap=0.4, min_pts=10):
    """
    用 GUDHI CoverComplex 实现 Mapper
    data: (N,D) 标准化后的点
    filter_vals: (N,) 切片方向的值
    """
    scaler = StandardScaler()
    data_s = scaler.fit_transform(data)

    # 构建 cover：把 filter 值区间分成 n_intervals 段，overlap 比例重叠
    f_min, f_max = filter_vals.min(), filter_vals.max()
    interval_len = (f_max - f_min) / (n_intervals - (n_intervals - 1) * overlap)
    stride = interval_len * (1 - overlap)

    # 每段内的点索引
    intervals = []
    for i in range(n_intervals):
        lo = f_min + i * stride
        hi = lo + interval_len
        mask = (filter_vals >= lo) & (filter_vals <= hi)
        idx = np.where(mask)[0]
        if len(idx) >= min_pts:
            intervals.append(idx)

    if len(intervals) < 2:
        return {'nodes': len(intervals), 'edges': 0}

    # 每段内 DBSCAN 聚类 → 节点
    nodes = []  # list of (interval_idx, set of point indices)
    node_pts = []
    for i, idx in enumerate(intervals):
        if len(idx) <= min_pts:
            nodes.append((i, set(idx)))
            node_pts.append(idx)
            continue
        try:
            clf = DBSCAN(eps=0.5, min_samples=min_pts).fit(data_s[idx])
        except:
            nodes.append((i, set(idx)))
            node_pts.append(idx)
            continue
        labels = clf.labels_
        for lab in set(labels):
            if lab == -1:
                continue
            mask = labels == lab
            nodes.append((i, set(idx[mask])))
            node_pts.append(idx[mask])

    # 边：两个节点共享至少 1 个点
    edges = 0
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if nodes[i][0] != nodes[j][0]:  # 不同 interval
                if nodes[i][1] & nodes[j][1]:
                    edges += 1
            else:  # 同一 interval 不同 cluster 但相邻
                pass

    return {'nodes': len(nodes), 'edges': edges, 'node_pts': node_pts}

# ============================================================
# 2. Persistence (RipsComplex)
# ============================================================
def compute_persistence(data, max_dim=1, max_edge=np.inf, sparse=0.3):
    """VR 复形 → 持久同调"""
    rips = gudhi.RipsComplex(points=data, max_edge_length=max_edge, sparse=sparse)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()
    return st.persistence(), st

def bottleneck_dist(diag1, diag2, dim):
    d1 = np.array([[b, d] for dim_, (b, d) in diag1 if dim_ == dim and d < np.inf])
    d2 = np.array([[b, d] for dim_, (b, d) in diag2 if dim_ == dim and d < np.inf])
    if len(d1) == 0 or len(d2) == 0:
        return np.nan
    return gudhi.bottleneck_distance(d1, d2)

# ============================================================
# 3. 主流程
# ============================================================
def analyze(name, cols):
    print(f"\n{'='*60}")
    print(f"  {name}  ({len(cols)}D)")
    print(f"{'='*60}")

    # 采样 + 标准化
    class_pts = {}
    for cls in ['GALAXY','STAR','QSO']:
        subset = train[train['class']==cls]
        s = subset.sample(min(SAMPLE_PER_CLASS, len(subset)), random_state=RANDOM_SEED)
        class_pts[cls] = s[cols].values

    all_arr = np.vstack([class_pts[c] for c in ['GALAXY','STAR','QSO']])
    sc = StandardScaler().fit(all_arr)
    for cls in ['GALAXY','STAR','QSO']:
        class_pts[cls] = sc.transform(class_pts[cls])

    # ---- Mapper (沿 PC1) ----
    pca_all = PCA(n_components=1).fit(all_arr)
    print("\n  Mapper (沿 PC1):")
    for cls in ['GALAXY','STAR','QSO']:
        pts = class_pts[cls]
        f = PCA(n_components=1).fit_transform(pts).flatten()
        m = mapper_graph(pts, f, n_intervals=8, overlap=0.4, min_pts=10)
        n, e = m['nodes'], m['edges']
        print(f"    {cls:>6}: {n:>3} nodes, {e:>4} edges  {'≈ 线状' if e <= n else '≈ 网状' if e > 2*n else '≈ 树状'}")

    # ---- Mapper (有redshift的话沿redshift) ----
    if 'redshift' in cols:
        z_idx = cols.index('redshift')
        print("\n  Mapper (沿 redshift):")
        for cls in ['GALAXY','STAR','QSO']:
            pts = class_pts[cls]
            f = pts[:, z_idx]
            m = mapper_graph(pts, f, n_intervals=8, overlap=0.4, min_pts=10)
            n, e = m['nodes'], m['edges']
            print(f"    {cls:>6}: {n:>3} nodes, {e:>4} edges  {'≈ 线状' if e <= n else '≈ 网状' if e > 2*n else '≈ 树状'}")

    # ---- Persistence ----
    pers = {}
    print("\n  Persistence (H₀ / H₁):")
    for cls in ['GALAXY','STAR','QSO']:
        p, _ = compute_persistence(class_pts[cls], max_dim=1, max_edge=MAX_EDGE_LENGTH, sparse=0.3)
        pers[cls] = p
        h0 = [d-b for dim,(b,d) in p if dim==0 and d < np.inf]
        h1 = [d-b for dim,(b,d) in p if dim==1 and d < np.inf]
        print(f"    {cls:>6}:  H₀={len(h0)-1 if h0 else 0} (max_life={max(h0):.3f})  "
              f"H₁={len(h1)} (max_life={max(h1):.3f}  {'★ cycle!' if h1 and max(h1)>1.0 else ''})")

    # ---- Bottleneck ----
    for dim_label, dim in [('H₀',0), ('H₁',1)]:
        print(f"\n  Bottleneck {dim_label}:")
        print(f"            {'GALAXY':>8}  {'STAR':>8}  {'QSO':>8}")
        for c1 in ['GALAXY','STAR','QSO']:
            vals = []
            for c2 in ['GALAXY','STAR','QSO']:
                d = bottleneck_dist(pers[c1], pers[c2], dim)
                vals.append(f"{d:.4f}" if not np.isnan(d) else "   N/A")
            print(f"    {c1:>6}:  {'  '.join(vals)}")

    return pers, class_pts

# ============================================================
# 4. 画图
# ============================================================
def plot_results(all_results):
    fig, axes = plt.subplots(len(FEATURE_SETS), 3, figsize=(15, 3.5 * len(FEATURE_SETS)))
    colors = {'GALAXY':'red','STAR':'blue','QSO':'green'}

    for row, (name, (pers, _)) in enumerate(all_results.items()):
        # H₀
        for cls, c in colors.items():
            pts = [(b,d) for dim,(b,d) in pers[cls] if dim==0 and d<np.inf]
            if pts:
                b,d = zip(*pts)
                axes[row,0].scatter(b,d,c=c,alpha=0.5,s=8,label=cls if row==0 else '')
        axes[row,0].plot([0,MAX_EDGE_LENGTH],[0,MAX_EDGE_LENGTH],'k--',alpha=0.3)
        axes[row,0].set_title(f'{name} — H₀')
        if row == 0: axes[row,0].legend(fontsize=7)

        # H₁
        for cls, c in colors.items():
            pts = [(b,d) for dim,(b,d) in pers[cls] if dim==1 and d<np.inf]
            if pts:
                b,d = zip(*pts)
                axes[row,1].scatter(b,d,c=c,alpha=0.6,s=15,label=cls if row==0 else '')
        axes[row,1].plot([0,MAX_EDGE_LENGTH],[0,MAX_EDGE_LENGTH],'k--',alpha=0.3)
        axes[row,1].set_title(f'{name} — H₁')

        # lifetime histogram
        for cls, c in colors.items():
            lifetimes = [d-b for dim,(b,d) in pers[cls] if d<np.inf]
            axes[row,2].hist(lifetimes,bins=25,alpha=0.4,color=c,label=cls if row==0 else '',density=True)
        axes[row,2].set_title(f'{name} — Lifetime')
        if row == 0: axes[row,2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('eda_tda_all.png', dpi=150, bbox_inches='tight')
    print("\nSaved: eda_tda_all.png")

# ============================================================
# 5. 执行
# ============================================================
if __name__ == '__main__':
    all_results = {}
    for name, cols in FEATURE_SETS.items():
        all_results[name] = analyze(name, cols)

    plot_results(all_results)
    print("\nDone!")
