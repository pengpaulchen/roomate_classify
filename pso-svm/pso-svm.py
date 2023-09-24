import pandas as pd
import numpy as np
from sklearn.svm import SVC
from pyswarm import pso

# 读取用户数据
data = pd.read_excel('./userinfo.xlsx')

# 提取特征和标签
X = data.iloc[:, 1:].values  # 特征（睡觉、声音、人际关系、居住环境）
y = data.iloc[:, 0].values   # 用户号

# 定义SVM分类器
def svm_objective(params):
    C, gamma = params
    classifier = SVC(kernel='rbf', C=C, gamma=gamma)
    classifier.fit(X, y)
    # 计算分类错误率
    misclassified = np.sum(classifier.predict(X) != y)
    return misclassified



# 定义参数范围
param_ranges = [(0.1, 24.0), (0.1, 24.0)]

# 将参数范围转换为上界和下界
lb = [param[0] for param in param_ranges]
ub = [param[1] for param in param_ranges]

# 使用PSO算法优化SVM参数
def pso_objective(params):
    C, gamma = params
    classifier = SVC(kernel='rbf', C=C, gamma=gamma)
    classifier.fit(X, y)
    # 计算分类错误率
    misclassified = np.sum(classifier.predict(X) != y)
    return misclassified

best_params, _ = pso(pso_objective, lb=lb, ub=ub, swarmsize=10, maxiter=100)
# 使用最优参数训练SVM分类器
best_C, best_gamma = best_params
best_classifier = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
best_classifier.fit(X, y)

# 打印出哪些用户可以在一间宿舍
dormitory_assignments = {}
for user_id in np.unique(y):
    user_features = X[y == user_id]
    dorm_assignment = best_classifier.predict(user_features)
    dormitory_assignments[user_id] = dorm_assignment[0]

print("宿舍分配结果:")
for user_id, dorm_assignment in dormitory_assignments.items():
    print(f"用户{user_id}分配到宿舍{dorm_assignment}")

