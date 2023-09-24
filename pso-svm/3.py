import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pyswarm import pso
from sklearn.svm import SVC

# 读取用户数据
data = pd.read_excel('./userinfo.xlsx')

# 提取用户数据和指标权重
X = data.iloc[:, 1:].values
weights = [4, 3, 2, 1, 1]  # 指标权重，从左至右依次降低

# 归一化用户数据
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 使用PCA将指标数据降维为2维
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)


# 使用PSO优化SVM的超参数
def fitness_function(params):
    C, gamma = params
    clf = SVC(C=C, gamma=gamma)
    kmeans = KMeans(n_clusters=3)  # 分成3个宿舍
    kmeans.fit(X_2d)
    labels = kmeans.labels_

    # 确保每个宿舍都不超过4个人
    label_counts = np.bincount(labels)
    if all(count <= 4 for count in label_counts):
        return -len(set(labels))
    else:
        return float('inf')  # 惩罚函数，如果有宿舍超过4个人，则返回无穷大


# 定义PSO的参数范围
param_ranges = [(0.1, 10), (0.001, 1)]

# 使用pso进行参数优化
lb = [param[0] for param in param_ranges]
ub = [param[1] for param in param_ranges]

best_params, _ = pso(fitness_function, lb, ub, swarmsize=10, maxiter=20)

# 使用最佳超参数训练SVM模型
C, gamma = best_params
svm = SVC(C=C, gamma=gamma)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_2d)
labels = kmeans.labels_

# 将标签分组到宿舍，确保每间宿舍有四个人
dormitories = [[] for _ in range(3)]  # 分成3个宿舍
for i, label in enumerate(labels):
    if len(dormitories[label]) < 4:
        dormitories[label].append(i + 1)  # 用户号从1开始
    else:
        # 如果已经有四个人，分到下一个宿舍
        for j in range(3):
            if len(dormitories[(label + j) % 3]) < 4:
                dormitories[(label + j) % 3].append(i + 1)
                break

# 打印每个宿舍的舍友列表
for i, dormitory in enumerate(dormitories):
    print(f"宿舍 {i + 1} 的舍友：{dormitory}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 使用PCA降维到三维空间进行可视化
pca = PCA(n_components=3)  # 降维到3维
X_pca = pca.fit_transform(X)

# 使用最佳超参数训练SVM模型
C, gamma = best_params
svm = SVC(C=C, gamma=gamma)
svm.fit(X_pca, labels)  # 使用降维后的数据

# 创建一个网格以绘制决策边界
h = .02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
z_min, z_max = X_pca[:, 2].min() - 1, X_pca[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))

# 预测网格上的每个点的标签
XYZ = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
Z = svm.predict(XYZ)
Z = Z.reshape(xx.shape)

# 绘制决策边界
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contourf(xx, yy, zz, Z, cmap=plt.cm.Paired, alpha=0.8)

# 绘制数据点
for i in range(3):
    ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], X_pca[labels == i, 2], label=f'宿舍 {i + 1}', edgecolors='k')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei or your preferred Chinese font
plt.rcParams['axes.unicode_minus'] = False  # Ensure that minus signs (-) are displayed correctly
ax.set_xlabel('PCA Dimension 1')
ax.set_ylabel('PCA Dimension 2')
ax.set_zlabel('PCA Dimension 3')
ax.set_title('SVM Decision Boundaries')
ax.legend()
plt.show()
