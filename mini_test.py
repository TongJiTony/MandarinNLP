import segmenter

if __name__== '__main__':
    # 测试分词
    text = "清华大学自然语言处理研究。"
    # 输出: ['清华大学', '自然语言处理', '研究']
    hard_text = "共同创造美好的新世纪——二○○一年新年贺词（二○○○年十二月三十一日）（附图片1张）2000"
    # 输出：[共同  创造  美好  的  新  世纪  ——  二○○一年  新年  贺词 （  二○○○年  十二月  三十一日  ）  （  附  图片  1  张  ） ]

    FMMSegmenter = segmenter.FMMSegmenter('dict.txt')
    RMinMSegmenter = segmenter.RMinMSegmenter('dict.txt')

    print("testline:")
    print(text)
    print(hard_text)

    print("FMMSegmenter:")
    print(FMMSegmenter.segment(text))
    print(FMMSegmenter.segment(hard_text))

    print("RMinMSegmenter:")
    print(RMinMSegmenter.segment(text))
    print(RMinMSegmenter.segment(hard_text))

    segmenter = segmenter.CRFSegmenter()
    # 训练模型
    train_file = "bmes_train_pku.txt"
    segmenter.train(train_file)
    
    # 测试分词
    segmented_words = segmenter.segment(hard_text)
    print("分词结果:", segmented_words)