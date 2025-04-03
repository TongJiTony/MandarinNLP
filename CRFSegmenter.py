import sklearn_crfsuite
# 使用crfsuite中的crf进行分词，并添加各种功能函数，如数据加载，特征提取及将预判标签转换为分词结果
class CRFSegmenter:
    def __init__(self):
        # 由于训练数据量不大，所以使用平均感知机，相比之下lbfgs拟牛顿法更适合大规模数据训练
        self.crf = sklearn_crfsuite.CRF(
            algorithm='ap',
            max_iterations=100,
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

if __name__ == "__main__":
    # 标签集
    crf = CRFSegmenter()

    train_file = "bmes_train_pku.txt"

    # 训练 CRF
    crf.train(train_file)

    # 测试句子
    test_sentence = "人要是行，干一行行一行，一行行行行行， 行行行干哪行都行。"
    test_sentence = "5. 校长说：校服上除了校徽别别别的，让你们别别别的别别别的你非别别的！"
    print(test_sentence)
    print(crf.segment(test_sentence))