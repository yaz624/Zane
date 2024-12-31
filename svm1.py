from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from matplotlib import cm

# 加载数据
data = pd.read_csv('cleaned_data.txt', sep=r"\s+", header=None)
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签

# 检查标签类别
print("数据标签唯一值:", np.unique(y))
# 检查 y 中是否有不属于 {0, 1} 的值
invalid_rows = data[data.iloc[:, -1] == 2]
print("包含 2 的行数据：")
print(invalid_rows)
# 确保训练集和测试集类别分布一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 检查训练集是否有足够类别
if len(np.unique(y_train)) < 2:
    raise ValueError("训练集需要至少包含两个类别！")

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用 PCA 将数据降维到 3 维
pca_3d = PCA(n_components=3)
X_train_3d = pca_3d.fit_transform(X_train)
X_test_3d = pca_3d.transform(X_test)

# 打印降维后数据形状
print("降维后训练集的形状（3维）：", X_train_3d.shape)

# 训练 SVM 模型
model_3d = SVC(kernel='rbf', C=1.0)
model_3d.fit(X_train_3d, y_train)

# 模型测试
accuracy_3d = model_3d.score(X_test_3d, y_test)
print("降维到 3 维的测试集准确率：", accuracy_3d)

# 可视化 3 维数据和决策边界
def plot_3d_decision_boundary(X, y, model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='winter', s=30)

    # 创建网格
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    xx, yy, zz = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 20),
        np.linspace(ylim[0], ylim[1], 20),
        np.linspace(zlim[0], zlim[1], 20)
    )

    # 将网格点展开为二维点集
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    decision_values = model.decision_function(grid)

    # 绘制 3D 决策边界
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=decision_values, cmap='coolwarm', alpha=0.2, s=1)

    # 设置标题和轴标签
    ax.set_title("3D SVM Decision Boundary")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")

    # 添加颜色条
    norm = plt.Normalize(vmin=decision_values.min(), vmax=decision_values.max())
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='coolwarm'), ax=ax, shrink=0.5, aspect=5)

    plt.show()

plot_3d_decision_boundary(X_train_3d, y_train, model_3d)
