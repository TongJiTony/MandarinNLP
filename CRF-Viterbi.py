import numpy as np
from collections import defaultdict

class CRFSegmenter:
    def __init__(self, labels):
        self.labels = labels
        self.trans_probs = defaultdict(lambda: defaultdict(float))  # 转移概率矩阵
        self.emit_probs = defaultdict(lambda: defaultdict(float))   # 发射概率矩阵

    def load_bmes_data(self, file_path):
        """ 读取 BMES 格式的训练数据 """
        train_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split("\n\n")  # 句子之间用双换行分隔
            for sentence in lines:
                sent_data = []
                for line in sentence.split("\n"):
                    if "\t" in line:  # 处理 "字\t标签" 的格式
                        char, tag = line.split("\t")
                        sent_data.append((char, tag))
                train_data.append(sent_data)
        return train_data

    def train(self, data):
        """ 训练模型，计算转移概率和发射概率 """
        trans_counts = defaultdict(lambda: defaultdict(int))
        emit_counts = defaultdict(lambda: defaultdict(int))
        label_counts = defaultdict(int)

        # 统计转移和发射概率
        for sent in data:
            prev_label = None
            for word, label in sent:
                emit_counts[label][word] += 1
                label_counts[label] += 1

                if prev_label is not None:
                    trans_counts[prev_label][label] += 1
                prev_label = label

        # 归一化计算转移概率
        for prev_label, next_labels in trans_counts.items():
            total_trans = sum(next_labels.values())
            for next_label, count in next_labels.items():
                self.trans_probs[prev_label][next_label] = count / total_trans

        # 归一化计算发射概率
        for label, words in emit_counts.items():
            total_emit = sum(words.values())
            for word, count in words.items():
                self.emit_probs[label][word] = count / total_emit

    def viterbi(self, sentence):
        """ 使用 Viterbi 进行序列标注解码 """
        n = len(sentence)
        dp = [{} for _ in range(n)]  # DP 表
        backtrack = [{} for _ in range(n)]  # 回溯表

        # 初始化
        for label in self.labels:
            dp[0][label] = self.emit_probs[label].get(sentence[0], 0.0001)  # 发射概率
            backtrack[0][label] = None

        # 动态规划计算最优路径
        for i in range(1, n):
            for curr_label in self.labels:
                max_prob, best_prev_label = -1, None
                for prev_label in self.labels:
                    prob = dp[i - 1][prev_label] * self.trans_probs[prev_label][curr_label] * \
                           self.emit_probs[curr_label].get(sentence[i], 0.0001)
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_label = prev_label
                dp[i][curr_label] = max_prob
                backtrack[i][curr_label] = best_prev_label

        # 回溯获取最优路径
        best_path = []
        max_final_prob = -1
        last_label = None
        for label in self.labels:
            if dp[-1][label] > max_final_prob:
                max_final_prob = dp[-1][label]
                last_label = label
        best_path.append(last_label)

        for i in range(n - 1, 0, -1):
            best_path.append(backtrack[i][last_label])
            last_label = backtrack[i][last_label]

        best_path.reverse()
        return best_path

# 主程序
if __name__ == "__main__":
    # 标签集
    labels = ["B", "M", "E", "S"]
    crf = CRFSegmenter(labels)

    # 加载 BMES 格式训练数据
    train_file = "bmes_train_pku_mini.txt"
    train_data = crf.load_bmes_data(train_file)

    # 训练 CRF
    crf.train(train_data)
    print("CRF 训练完成！")

    # 测试句子
    test_sentence = ["北", "京", "大", "学", "很", "棒"]
    predicted_tags = crf.viterbi(test_sentence)
    print("BMES 预测结果:", predicted_tags)