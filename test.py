from sklearn.metrics import precision_score, recall_score, f1_score

# 加载测试文件和黄金标准文件
def load_data(test_file, gold_file):
    with open(test_file, 'r', encoding='utf-8') as tf, open(gold_file, 'r', encoding='utf-8') as gf:
        test_lines = tf.readlines()
        gold_lines = gf.readlines()
    assert len(test_lines) == len(gold_lines), "测试文件与黄金标准文件的行数不一致"
    return test_lines, gold_lines

# 测试分词器
def test_segmenter(segmenter, test_lines, gold_lines):
    y_true = []
    y_pred = []
    for test_line, gold_line in zip(test_lines, gold_lines):
        test_text = test_line.strip()
        gold_tokens = gold_line.strip().split()
        y_true.extend(gold_tokens)  # 黄金标准分词结果
        y_pred.extend(segmenter.segment(test_text))  # 分词器输出
    return y_true, y_pred

# 计算分词准确率，召回率和F1 Score
def evaluate(y_true, y_pred):    
    true_set = set(y_true)
    pred_set = set(y_pred)
    common = true_set & pred_set
    precision = len(common) / len(pred_set) if pred_set else 0
    recall = len(common) / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


from segmenter import FMMSegmenter, RMinMSegmenter, CRFSegmenter, HMMSegmenter

if __name__ == "__main__":
    # 文件路径
    test_file = "pku_test_mini.txt"  # 测试文件路径
    gold_file = "pku_test_mini_gold.txt"  # 黄金标准文件路径

    # 加载数据
    test_lines, gold_lines = load_data(test_file, gold_file)

    # 初始化分词器(通过注释不同段落进行选择，CRF需要进行训练， 机械分词可使用FMM或RMinM，读取dict字典)
    # segmenter = FMMSegmenter('dict.txt')
    # segmenter = HMMSegmenter()
    segmenter = CRFSegmenter()
    # 训练模型（CRF或者HMM都是同样的步骤）
    train_file = "bmes_train_pku.txt"
    segmenter.train(train_file)

    # 测试分词器
    y_true, y_pred = test_segmenter(segmenter, test_lines, gold_lines)

    # 评估分词器性能
    evaluate(y_true, y_pred)