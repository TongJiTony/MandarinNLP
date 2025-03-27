from segmenter import FMMSegmenter
# 初始化分词器
segmenter = FMMSegmenter('dict.txt')

# 测试分词
text = "清华大学自然语言处理研究"
print(segmenter.segment(text))
# 输出: ['清华大学', '自然语言处理', '研究']