# coding=utf-8
"""
实现隐马尔科夫模型的基本方法
Input: 初始状态概率向量，状态转移矩阵，发射矩阵
1. 前向算法
2. 后向算法
3. 维特比算法
"""

from numpy import *


class HMM:

    def __init__(self):
        self.A = array([(0.5, 0.2, 0.3), (0.3, 0.5, 0.2), (0.2, 0.3, 0.5)])  # 初始化状态转移矩阵
        self.B = array([(0.5, 0.5), (0.4, 0.6), (0.7, 0.3)])  # 初始化发射矩阵
        self.pi = array([0.2, 0.4, 0.4])  # 初始化概率向量
        self.o = [0, 1, 0]  # 观测序列
        self.t = len(self.o)  # 观测序列的长度
        self.m = len(self.A)  # 状态集合的个数
        self.n = len(self.B[0])  # 观测集合的个数

    # 前向算法
    def forward(self):
        """
        定义alpha: 到时刻t部分观测序列o1,o2,...,ot且状态为qi的前向概率矩阵
        alpha矩阵的行数为观测序列数；
        alpha矩阵的列数为隐状态个数；
        :return: alpha
        """
        alpha = array(zeros((self.t, self.m)))

        # 1. 初值 计算时刻=1, 观测=o1，状态=qi的概率
        for i in range(self.m):
            alpha[0][i] = self.pi[i] * self.B[i][self.o[0]]

        # 2. 递推 对时刻t=1,2,...,T-1,计算下一时刻的alpha
        for t in range(1, self.t):
            for i in range(self.m):
                temp = 0
                for j in range(self.m):
                    # 计算t的前一时刻所有状态的alpha值 * 状态转移概率=t+1时刻i的状态概率
                    temp += alpha[t-1][j] * self.A[j][i]
                # t时刻i的状态概率 * 发射概率得到最终t+1时刻的i的状态概率
                alpha[t][i] = temp * self.B[i][self.o[t]]

        # 3. 终止
        result_forward = 0
        for k in range(self.m):
            result_forward += alpha[self.t-1][k]

        print("前向概率矩阵及当前观测序列概率如下：")
        print(alpha)
        print(result_forward)

    # 后向算法
    def backward(self):
        """
        定义beta: 时刻t，状态为qi的条件下，从t+1到T的部分观测序列ot+1,...,oT的概率
        beta矩阵的行数为观测序列数
        beta矩阵的列数为隐状态个数
        :return: beta
        """
        beta = array(zeros((self.t, self.m)))

        # 1. 初始化最终时刻所有的状态
        for i in range(self.m):
            beta[self.t-1][i] = 1
        # 2. 递推，对t=T-1,T-2,...,1
        t = self.t-2
        while t >= 0:
            for i in range(self.m):
                for j in range(self.m):
                    beta[t][i] = self.A[i][j] * self.B[j][self.o[t+1]] * beta[t+1][j]
            t = t - 1

        # 3. 终止
        result_backward = 0
        for i in range(self.m):
            result_backward += self.pi[i] * self.B[i][self.o[0]] * beta[0][i]

        print("后向概率矩阵及当前观测序列概率如下：")
        print(beta)
        print(result_backward)

    # 维特比算法
    def viterbi(self):
        """
        利用观测序列和模型测参数找出最有可能出现该观测序列的状态序列
        在时刻t，有很多路径可以到达状态i，并且观测序列为self.o
        每个路径都有自己的概率，记最大概率为矩阵delta，前一个状态节点为矩阵phi
        :return:
        """
        delta = array(zeros((self.t, self.m)))
        phi = array(zeros((self.t, self.m)))

        # 1. 初始化
        for i in range(self.m):
            delta[0][i] = self.pi[i] * self.B[i][self.o[0]]
            phi[0][i] = 0

        # 2. 递推 对t=2,3,...,T
        # 对于每一个时刻t
        for t in range(1, self.t):
            # 对于时刻t的每一个状态i的概率
            for i in range(self.m):
                # 假设第1个状态取得路径概率最大值
                maxnum = delta[t-1][0] * self.A[0][i]
                node = 1
                # 遍历t-1时刻的状态j的概率
                for j in range(1, self.m):
                    temp = delta[t-1][j] * self.A[j][i]
                    # 对比
                    if maxnum < temp:
                        maxnum = temp
                        node = j + 1

                # 记录下到当前t时刻的路径状态概率最大值以及相应的前一个节点
                delta[t][i] = maxnum * self.B[i][self.o[t]]
                phi[t][i] = node

        # T时刻最优路径的概率
        max_pro_T = max(delta[self.t-1])
        print("%d时刻最优路径的概率: %.4f" % (self.t, max_pro_T))
        max_pro_index_T = delta[self.t-1].tolist().index(max_pro_T) + 1
        print("%d时刻最优路径的节点: %d" % (self.t, max_pro_index_T))

        # 3. 终止 找到T时刻概率最大的路径,在矩阵phi进行回溯
        t = self.t-1
        while t > 0:
            print("%d时刻最优路径的节点: %d" % (t, phi[t][max_pro_index_T-1]))
            t = t - 1


if __name__ == '__main__':
    hmm = HMM()
    hmm.forward()
    hmm.backward()
    hmm.viterbi()

