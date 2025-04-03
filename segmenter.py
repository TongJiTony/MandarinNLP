import re

class FMMSegmenter:
    def __init__(self, dict_path):
        self.word_dict = self.load_dict(dict_path)
        self.max_len = max(len(word) for word in self.word_dict) if self.word_dict else 0
        self.special_symbols = "，。？！：《》、（）【】“”‘’~+-*/\=<>()[]\{\}" #自定义特殊符号集合
        self.two_symbols = "——" #破折号特判
    
    def load_dict(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    
    def segment(self, text):
        result = []
        index = 0
        text_len = len(text)
        
        while index < text_len:
            word = None

            # 先进行分词预处理，如特殊符号，年份数字，字母等，提高效率并减少歧义
            # 特殊符号处理
            if text[index] in self.special_symbols:
                word = text[index]
                size = 1
            
            # 匹配中文年份（如“二oo一年”或“二ooo年”）
            elif re.match(r'^[一二三四五六七八九十○O零]+年', text[index:]):
                match = re.match(r'^[一二三四五六七八九十○O零]+年', text[index:])
                word = match.group()
                size = len(word)

            # 匹配数字（连续数字如21、2021等）
            elif re.match(r'^\d+', text[index:]):
                match = re.match(r'^\d+', text[index:])
                word = match.group()  # 匹配到的完整数字字符串
                size = len(word)

            # 匹配字母（连续字母如oo、ooo等）
            elif re.match(r'^[a-zA-Z○]+', text[index:]):
                match = re.match(r'^[a-zA-Z○]+', text[index:])
                word = match.group()  # 匹配到的完整字母字符串
                size = len(word)

            # 从最大长度开始尝试匹配
            else:
                for size in range(min(self.max_len, text_len - index), 0, -1):
                    candidate = text[index: index + size]
                    if candidate in self.word_dict or candidate in self.two_symbols:
                        word = candidate
                        break
                
                # 如果词典中没有匹配，则按单字切分
                if not word:
                    word = text[index]
                    size = 1
            
            result.append(word)
            index += size
        return result

class RMinMSegmenter:
    def __init__(self, dict_path):
        self.word_dict = self.load_dict(dict_path)
        self.max_len = max(len(word) for word in self.word_dict) if self.word_dict else 0
        self.special_symbols = "，。？！：《》、（）【】“”‘’~+-*/\=<>()[]\{\}" #自定义特殊符号集合
        self.two_symbols = "——" #破折号特判
    
    def load_dict(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)

    def segment(self, text):
        result = []
        text_len = len(text)
        index = text_len # 从后往前进行匹配

        while index > 0:
            word = None
 
            # 特殊符号处理
            if text[index - 1 : index] in self.special_symbols:
                word = text[index - 1 : index]
                size = 1
            
            # 匹配中文年份（如“二oo一年”或“二ooo年”）
            elif re.search(r'[一二三四五六七八九十○O零]+年$', text[:index]):
                match = re.search(r'[一二三四五六七八九十○O零]+年$', text[:index])
                word = match.group()
                size = len(word)

            # 匹配数字（连续数字如21、2021等）
            elif re.search(r'\d+$', text[:index]):
                match = re.search(r'\d+$', text[:index])
                word = match.group()  # 匹配到的完整数字字符串
                size = len(word)

            # 匹配字母（连续字母如oo、ooo等）
            elif re.search(r'[a-zA-Z○]+$', text[:index]):
                match = re.search(r'[a-zA-Z○]+$', text[:index])
                word = match.group()  # 匹配到的完整字母字符串
                size = len(word)

            else:
                # 从最小长度开始尝试匹配
                for size in range(2, min(self.max_len, index + 1), 1):
                    candidate = text[index - size : index]
                    if candidate in self.word_dict or candidate in self.two_symbols:
                        word = candidate
                        break
                # 如果词典中没有匹配，则按单字切分
                if not word:
                    word = text[index - 1 : index]
                    size = 1
            result.append(word)
            index -= size

        result = list(reversed(result))
        return result

import sklearn_crfsuite
# 使用crfsuite中的crf进行分词，并添加各种功能函数，如数据加载，特征提取及将预判标签转换为分词结果
class CRFSegmenter:
    def __init__(self):
        # 由于训练数据量不大，所以使用平均感知机，相比之下lbfgs拟牛顿法更适合大规模数据训练
        self.crf = sklearn_crfsuite.CRF(
            algorithm='ap',
            max_iterations=1000,
            all_possible_transitions=True
        )
    
    # 加载 BMES 格式的训练数据
    @staticmethod
    def load_bmes_data(file_path):
        train_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split("\n\n")  # 用双换行分隔句子
            for sentence in lines:
                sent_data = []
                for line in sentence.split("\n"):
                    char, tag = line.split("\t")
                    sent_data.append((char, tag))
                train_data.append(sent_data)
        return train_data

    # 特征提取函数
    @staticmethod
    def word2features(sent, i):
        word = sent[i][0]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word.isdigit()': word.isdigit(),
            'word.isupper()': word.isupper(),
            'word.isalpha()': word.isalpha(),
        }
        if i > 0:
            prev_word = sent[i - 1][0]
            features.update({'prev_word.lower()': prev_word.lower()})
        else:
            features['BOS'] = True  # 句子开头
        if i < len(sent) - 1:
            next_word = sent[i + 1][0]
            features.update({'next_word.lower()': next_word.lower()})
        else:
            features['EOS'] = True  # 句子结尾
        return features

    # 提取句子的特征和标签
    @staticmethod
    def extract_features(sent):
        return [CRFSegmenter.word2features(sent, i) for i in range(len(sent))]

    @staticmethod
    def extract_labels(sent):
        return [label for _, label in sent]

    # BMES 标签转分词结果
    @staticmethod
    def bmes_to_words(chars, tags):
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

    # 训练模型
    def train(self, train_file):
        train_data = self.load_bmes_data(train_file)
        X_train = [self.extract_features(sent) for sent in train_data]
        y_train = [self.extract_labels(sent) for sent in train_data]
        self.crf.fit(X_train, y_train)
        print("CRF 模型训练完成！")

    # 分词接口函数
    def segment(self, text):
        chars = list(text)  # 将输入文本转为字符列表
        dummy_tags = ['S'] * len(chars)  # 初始化一个假标签列表，仅用于生成特征
        sent = [(char, tag) for char, tag in zip(chars, dummy_tags)]
        X_test = self.extract_features(sent)  # 提取测试特征
        y_pred = self.crf.predict([X_test])[0]  # 预测 BMES 标签, 每一句返回的是[[]]，因此需要提取[0]
        return self.bmes_to_words(chars, y_pred) 

import numpy as np
from collections import defaultdict
class DIYCRFSegmenter:
    def __init__(self, labels = ["B", "M", "E", "S"]):
        self.labels = labels
        self.trans_probs = defaultdict(lambda: defaultdict(float))  # Tk转移概率矩阵，前一个状态到当前状态
        self.emit_probs = defaultdict(lambda: defaultdict(float))   # Sj状态概率矩阵，观测值到当前状态

    @staticmethod
    def load_bmes_data(file_path):
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

    @staticmethod
    def bmes_to_words(chars, tags):
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

    def train(self, train_file):
        """ 训练模型，计算转移概率和发射概率 """
        data = self.load_bmes_data(train_file)
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
        
        print("CRF 训练完成！")


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

    def segment(self, text):
        chars = list(text)  # 将输入文本转为字符列表
        y_pred = self.viterbi(chars)
        return self.bmes_to_words(chars, y_pred) 