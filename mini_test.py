import segmenter

if __name__== '__main__':
    # 测试分词
    text = "清华大学自然语言处理研究。"
    # 输出: ['清华大学', '自然语言处理', '研究']
    hard_text = "共同创造美好的新世纪——二○○一年新年贺词（二○○○年十二月三十一日）（附图片1张）2000"
    # 输出：[共同  创造  美好  的  新  世纪  ——  二○○一年  新年  贺词 （  二○○○年  十二月  三十一日  ）  （  附  图片  1  张  ） ]

    hard_text = "5. 校长说：校服上除了校徽别别别的，让你们别别别的别别别的你非别别的！"
    # hard_text = "中文信息处理课程由同济大学卫老师讲授"
    # hard_text = "“家事国事天下事，让人民过上幸福生活是头等大事。”"
    hard_text = '''
                1. 中文信息处理课程由同济大学卫老师讲授。
                2. 中国的历史源远流长，有着上下五千年的历史。
                3. 当代大学生在追求梦想的同时，也面临着现实生活中不断变化的竞争和压力。
                4. 两千里的河堤，已经完全支离破碎了，许多地方被敌伪挖成了封锁沟，许多地方被农民改成了耕地。再加上风吹雨打，使许多段河堤连痕迹都没有了。
                5. 最高人民法院认定该区块链金融合约因违反反洗钱法相关条款而无效。
                6. 计算机科学与技术学院的研究生们在国际知名期刊上发表了大量高水平论文，获得广泛认可。'''
    hard_text = "人要是行，干一行行一行，一行行行行行， 行行行干哪行都行。"
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