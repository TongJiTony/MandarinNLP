import numpy as np
from collections import defaultdict

class CRFSegmenter:
    def __init__(self, labels):
        self.labels = labels
        self.trans_probs = defaultdict(lambda: defaultdict(float))  # Tk转移概率矩阵，前一个状态到当前状态
        self.emit_probs = defaultdict(lambda: defaultdict(float))   # Sj状态概率矩阵，观测值到当前状态

    def load_bmes_data(self, file_path):
        """ 读取 BMES 格式的训练数据 """
        train_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split("\n\n")  # 句子之间用双换行分隔
            for sentence in lines:
                sent_data = []
                for line in sentence.split("\n"):
                    char, tag = line.split("\t")
                    sent_data.append((char, tag))
                train_data.append(sent_data)
        return train_data

    def bmes_to_words(self, chars, tags):
        words = []
        word = ""
        for char, tag in zip(chars, tags):
            if tag == 'B':
                if word:  # 如果当前有词，先加入到结果中
                    words.append(word)
                word = char
            elif tag == 'M':
                word += char
            elif tag == 'E':
                word += char
                words.append(word)  # 结束一个词
                word = ""
            elif tag == 'S':
                if word:  # 如果当前有词，先加入到结果中
                    words.append(word)
                words.append(char)  # S 直接是一个单独的词
                word = ""
        if word:  # 如果最后还有未结束的词，加入到结果中
            words.append(word)
        return words

    def segment(self, text):
        chars = list(text)  # 将输入文本转为字符列表
        y_pred = self.viterbi(chars)
        return self.bmes_to_words(chars, y_pred) 

    def train(self, data):
        """ 训练模型，计算转移概率和发射概率 """
        trans_counts = defaultdict(lambda: defaultdict(int))
        emit_counts = defaultdict(lambda: defaultdict(int))
        label_counts = defaultdict(int)

        # 统计转移和状态概率
        for sent in data:
            prev_label = None
            for word, label in sent:
                emit_counts[label][word] += 1
                label_counts[label] += 1

                if prev_label is not None:
                    trans_counts[prev_label][label] += 1
                prev_label = label

        # 归一化计算Tk概率
        for prev_label, next_labels in trans_counts.items():
            total_trans = sum(next_labels.values())
            for next_label, count in next_labels.items():
                self.trans_probs[prev_label][next_label] = count / total_trans

        # 归一化计算Sj概率
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
            # 限制首字符的标签种类
            if label not in {'S', 'B'}:
                dp[0][label] = 0.0001
            else:
                dp[0][label] = self.emit_probs[label].get(sentence[0], 0.0001)  # 特征概率
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
            # 末字标签约束: 最后一个字只允许 'E' 和 'S'
            if i == n - 1:
                for curr_label in self.labels:
                    if curr_label not in {'E', 'S'}:
                        dp[i][curr_label] = float("-inf")

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
    train_file = "bmes_train_pku.txt"
    train_data = crf.load_bmes_data(train_file)

    # 训练 CRF
    crf.train(train_data)
    print("CRF 训练完成！")

    # 测试句子
    test_sentence = "共同创造美好的新世纪"
    predicted_tags = crf.viterbi(list(test_sentence))
    print(test_sentence)
    print("BMES 预测结果:", predicted_tags)
    print(crf.segment(test_sentence))