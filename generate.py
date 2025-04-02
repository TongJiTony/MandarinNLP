# To generate train data for crf, convert test_gold.txt.
def load_data(gold_file):
    with open(gold_file, 'r', encoding='utf-8') as gf:
        gold_lines = gf.readlines()
    return gold_lines

def generate_bmes_data(sentence):
    """
    将分词后的句子转换为 BMES 格式。
    :param sentence: 输入句子，分词格式如 "清华大学 是 著名 学府"
    :return: BMES 格式数据
    """
    words = sentence.split()
    bmes_tags = []
    for word in words:
        if len(word) == 1:
            bmes_tags.append((word, "S"))
        else:
            bmes_tags.append((word[0], "B"))
            for char in word[1:-1]:
                bmes_tags.append((char, "M"))
            bmes_tags.append((word[-1], "E"))
    return bmes_tags


if __name__ == "__main__":
    gold_file = "pku_test_gold.txt"
    sentences = load_data(gold_file)
    
    output_file = "bmes_train_pku.txt"
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for sentence in sentences:
            # 转换为 BMES 格式
            bmes_tags = generate_bmes_data(sentence.strip())
            # 写入 BMES 格式结果到文件
            for char, tag in bmes_tags:
                f_out.write(f"{char}\t{tag}\n")
            # 每个句子之间加入一个换行符
            f_out.write("\n")
    
    print(f"BMES 数据已保存到 {output_file}")
