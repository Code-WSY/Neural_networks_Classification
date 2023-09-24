import matplotlib.pyplot as plt
import numpy as np


class Neural_networks_ClassificationModel:
    def __init__(self):
        self.hidden_structure = [30]
        self.layers = len(self.hidden_structure) + 2
        self.learning_rate_init = 0.2
        self.power_t = 0.5  # 学习率的衰减指数
        self.increase_t = 1.05  # 学习率的增加指数
        self.grad_explosion_t = 0.5 # 梯度爆炸时，学习率的衰减指数
        self.Loss = 0
        self.max_epochs = 200
        self.mini_batch_size = 0.1 # 采样比例
        self.activation_input = "sigmoid"
        self.activation_hidden = "relu"
        self.activation_output = "equation"
        self.n_iter_nochange = 10 # 连续n_iter_nochange次损失函数增加，停止迭代
        self.min_loss = 1e-3  # 当损失函数的值小于这个值时，就停止迭代
        self.tol = 1e-5  # 当损失函数的值变化阈值
        self.scf = 5  # 当损失函数的值变化tol次数大于这个值时，就停止迭代
        self.n_iter = 0  # 迭代次数

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
    ):
        # 对于分类问题，Y的维度需要根据类别数进行调整
        # 格式化Y,首先判断Y的维度，根据其最大值和最小值作差，判断是几分类问题
        # 先判断输入值的格式是否正确
        # 如果不是整数，就报错
        if np.sum(Y - np.round(Y)) != 0:
            raise ValueError("输入错误：Y的值必须为整数。")
        # 如果Y的最小值不为0，就报错
        if np.min(Y) != 0:
            raise ValueError("输入错误：Y必须从0开始分类，如：0,1;0,1,2。")

        num_class = int(np.max(Y) - np.min(Y) + 1)
        Y_class = np.zeros((Y.shape[0], num_class))
        for i in range(Y.shape[0]):
            Y_class[i, int(Y[i])] = 1
        Y = Y_class

        from init_W import init_weight
        from backward import backward
        hidden_structure = self.hidden_structure
        learning_rate = self.learning_rate_init / X.shape[0]  # 这里是讲算是函数中的样本分母移到了学习率这里，X.shape[0]:样本个数
        activation_input = self.activation_input
        activation_hidden = self.activation_hidden
        activation_output = self.activation_output
        """
        X:输入层的输入
        Y:输出层的输出
        """
        # 初始化权重
        W, network_structure = init_weight(X, Y, hidden_structure)
        Loss_epoch = []
        count_increase = 0  # 记录连续增加的次数
        count_scf = 0  # 记录自洽的次数
        for i in range(self.max_epochs):
            self.n_iter += 1
            Loss, W = backward(
                X,
                Y,
                W,
                learning_rate,
                self.mini_batch_size,
                network_structure,
                activation_input,
                activation_hidden,
                activation_output,
            )
            # 记录连续增加的次数，调整学习率
            if Loss > self.Loss:
                count_increase += 1
                # 当连续增加的次数大于等于n_iter_nochange时，停止迭代
                if count_increase >= self.n_iter_nochange:
                    print("连续{}次损失函数增加，停止迭代。".format(self.n_iter_nochange))
                    break
                learning_rate = learning_rate * self.power_t
            else:
                count_increase = 0
                learning_rate = learning_rate * self.increase_t

            # 当损失函数的值小于tol时，就不会继续迭代了
            if Loss < self.min_loss:
                print('此次迭代损失函数的值为:{}，上次迭代损失函数的值为:{}'.format(Loss, self.Loss))
                print("损失函数误差小于{},停止迭代".format(self.min_loss))
                Loss_epoch.append(Loss)
                break

            if abs(self.Loss - Loss) < self.tol and Loss > self.min_loss:
                count_scf += 1
                Loss_epoch.append(Loss)
                if count_scf == self.scf:
                    print('损失函数的值变化小于{}已连续达{}次，停止迭代。'.format(self.tol, self.scf))
                    break
            else:
                count_scf = 0
                Loss_epoch.append(Loss)

            # 当损失函数的值大于1e10时，说明梯度爆炸，需要调小学习率
            if abs(self.Loss - Loss) > 1e10 and count_increase >= 2:
                print("梯度爆炸，程序将自动调小学习率并重新初始化权重。")
                learning_rate = learning_rate * self.grad_explosion_t
                W, network_structure = init_weight(X, Y, hidden_structure)
                # 重新初始化权重后，将连续增加的次数清零,将损失函数的值清零
                self.Loss = 0
                count_increase = 0
                count_scf = 0
                Loss_epoch = []
            else:
                self.Loss = Loss
                Loss_epoch.append(Loss)
            #  每个几次输出损失函数值
            if i % 10 == 0:
                print("epoch:", i, "Loss:", Loss)
        self.W = W
        # 找出Loss_epoch中第一个小于1的值，然后将其之前的值全部删除，目的是让图像更加清晰
        for i in range(len(Loss_epoch)):
            if Loss_epoch[i] < 1:
                Loss_epoch = Loss_epoch[i:]
                break
        return Loss_epoch

    # 预测，这是分类模型和回归模型的区别
    def predict(self, X_pred):
        from forward import forward
        Y_pred = []
        Probability= []
        for sample in range(X_pred.shape[0]):
            X = X_pred[sample, :]
            neural_elements = forward(
                X,
                self.W,
                activation_input=self.activation_input,
                activation_hidden=self.activation_hidden,
                activation_output=self.activation_output,
            )
            probability = neural_elements[-1]
            # 归一化到[0,1]
            probability = (probability - np.min(probability)) / (np.max(probability) - np.min(probability))
            # 继续将其之和为1
            probability = probability / np.sum(probability)
            # 最大概率对应的类别为1，其余为0
            # 找到最大概率对应的索引
            index = np.argmax(probability)
            # 将Y值解压缩
            Y_pred.append(index)
            Probability.append(probability)

        Y_pred = np.array(Y_pred)
        return Y_pred, Probability


if __name__ == "__main__":
    nnc = Neural_networks_ClassificationModel()
    np.random.seed(0)
    sample_num = 500
    # 导入分类数据样本
    from sklearn.datasets import make_moons
    X, Y = make_moons(n_samples=sample_num, noise=0.1, random_state=0)

    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=0,
    )
    # 训练模型
    Loss = nnc.fit(X_train, Y_train)
    print("一共迭代了{}次".format(nnc.n_iter))
    print("power_t:", nnc.power_t)
    y_hat, Probability = nnc.predict(X_test)
    # 输出参数
    print("hidden_structure:", nnc.hidden_structure)
    print("learning_rate_init:", nnc.learning_rate_init)
    print("sample_num:", sample_num)
    # 画图
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    # 作图：混淆矩阵
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    sns.set()
    C2 = confusion_matrix(Y_test, y_hat)
    sns.heatmap(C2, annot=True, cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion matrix")
    plt.title("Neural Networks Classification")
    # 作图：Loss曲线
    plt.subplot(1, 2, 2)
    plt.plot(Loss, c="black", label="Loss", linewidth=2)
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("Neural Networks Classification.png")
    plt.show()
