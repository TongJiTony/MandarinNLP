import numpy as np

from collections import defaultdict

class CRFSegmenter:
    def __init__(self, labels):
        self.labels = labels
        self.trans_probs = defaultdict(lambda: defaultdict(float))  # 转移概率矩阵
        self.emit_probs = defaultdict(lambda: defaultdict(float))   # 发射概率矩阵

    def train(self, data):
        # 初始化计数器
        trans_counts = defaultdict(lambda: defaultdict(int))
        emit_counts = defaultdict(lambda: defaultdict(int))
        label_counts = defaultdict(int)

        # 遍历数据统计频率
        for sent in data:
            prev_label = None
            for word, label in sent:
                # 统计发射概率（标签 -> 观测值）
                emit_counts[label][word] += 1
                label_counts[label] += 1

                # 统计转移概率（前一个标签 -> 当前标签）
                if prev_label is not None:
                    trans_counts[prev_label][label] += 1
                prev_label = label

        # 计算转移概率（通过频率归一化）
        for prev_label, next_labels in trans_counts.items():
            total_trans = sum(next_labels.values())
            for next_label, count in next_labels.items():
                self.trans_probs[prev_label][next_label] = count / total_trans

        # 计算发射概率（通过频率归一化）
        for label, words in emit_counts.items():
            total_emit = sum(words.values())
            for word, count in words.items():
                self.emit_probs[label][word] = count / total_emit

    def viterbi(self, sentence):
        n = len(sentence)
        dp = [{} for _ in range(n)]  # dp 表
        backtrack = [{} for _ in range(n)]  # 回溯表

        # 初始化
        for label in self.labels:
            dp[0][label] = self.emit_probs[label].get(sentence[0], 0.0001)  # 发射概率
            backtrack[0][label] = None

        # 动态规划
        for i in range(1, n):
            for curr_label in self.labels:
                max_prob, best_prev_label = -1, None
                for prev_label in self.labels:
                    prob = dp[i - 1][prev_label] * self.trans_probs[prev_label][curr_label] * self.emit_probs[curr_label].get(sentence[i], 0.0001)
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_label = prev_label
                dp[i][curr_label] = max_prob
                backtrack[i][curr_label] = best_prev_label

        # 回溯路径
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

# 示例训练
labels = ["B", "I", "O"]
crf = CRFSegmenter(labels)
test_sentence = ["北京", "大学", "很", "棒"]
print(crf.viterbi(test_sentence))