import sklearn_crfsuite
from sklearn_crfsuite import metrics

# 加载 BMES 格式的训练数据
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

# 从数据集中提取特征和标签
def extract_features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def extract_labels(sent):
    return [label for _, label in sent]

# 主程序
if __name__ == "__main__":
    # 加载训练数据
    train_file = "bmes_train_pku_mini.txt"
    train_data = load_bmes_data(train_file)

    # 提取训练数据的特征和标签
    X_train = [extract_features(sent) for sent in train_data]
    y_train = [extract_labels(sent) for sent in train_data]

    # 初始化并训练 CRF 模型
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    print("CRF 模型训练完成！")

    # 测试数据（自定义测试句子）
    test_sent = [('清', 'B'), ('华', 'M'), ('大', 'M'), ('学', 'E'), ('是', 'S')]
    X_test = extract_features(test_sent)  # 提取测试特征
    y_pred = crf.predict([X_test])[0]    # 预测 BMES 标签
    print("BMES 标签:", y_pred)