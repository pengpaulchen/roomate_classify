import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
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

# 使用PSO优化SVM的超参数
def fitness_function(params):
    C, gamma = params
    clf = SVC(C=C, gamma=gamma)
    kmeans = KMeans(n_clusters=1)  # 初始化KMeans
    kmeans.fit(X)  # 使用用户数据进行聚类
    labels = kmeans.labels_
    return -len(set(labels))  # 最小化不同标签的数量

# 定义PSO的参数范围
param_ranges = [(0.1, 10), (0.001, 1)]

# 使用pso进行参数优化
lb = [param[0] for param in param_ranges]
ub = [param[1] for param in param_ranges]

best_params, _ = pso(fitness_function, lb, ub, swarmsize=10, maxiter=20)

# 使用最佳超参数训练SVM模型
num_dorm=3
C, gamma = best_params
svm = SVC(C=C, gamma=gamma)
kmeans = KMeans(n_clusters=num_dorm)  # 四个宿舍
kmeans.fit(X)
labels = kmeans.labels_
unique_labels = np.unique(labels)
print("Unique labels:", unique_labels)
# 将标签分组到宿舍，确保每间宿舍有四个人
dormitories = [[] for _ in range(num_dorm)]
for i, label in enumerate(labels):
    if len(dormitories[label]) < 4:
        dormitories[label].append(i + 1)  # 用户号从1开始
    else:
        # 如果已经有四个人，分到下一个宿舍
        for j in range(4):
            if len(dormitories[(label + j) % 4]) < 4:
                dormitories[(label + j) % 4].append(i + 1)
                break

# 打印每个宿舍的舍友列表
for i, dormitory in enumerate(dormitories):
    print(f"宿舍 {i + 1} 的舍友：{dormitory}")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 使用PCA降维到二维空间进行可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用最佳超参数训练SVM模型
C, gamma = best_params
svm = SVC(C=C, gamma=gamma)
svm.fit(X_pca, labels)  # 使用降维后的数据

# 创建一个网格以绘制决策边界
h = .02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测网格上的每个点的标签
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# 绘制数据点
for i in range(num_dorm):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'宿舍 {i + 1}', edgecolors='k')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei or your preferred Chinese font
plt.rcParams['axes.unicode_minus'] = False  # Ensure that minus signs (-) are displayed correctly
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('SVM Decision Boundaries')
plt.legend()
plt.show()
