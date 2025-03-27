class FMMSegmenter:
    def __init__(self, dict_path):
        self.word_dict = self.load_dict(dict_path)
        self.max_len = max(len(word) for word in self.word_dict) if self.word_dict else 0
    
    def load_dict(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    
    def segment(self, text):
        result = []
        index = 0
        text_len = len(text)
        
        while index < text_len:
            word = None
            # 从最大长度开始尝试匹配
            for size in range(min(self.max_len, text_len - index), 0, -1):
                candidate = text[index:index+size]
                if candidate in self.word_dict:
                    word = candidate
                    break
            # 如果词典中没有匹配，则按单字切分
            if not word:
                word = text[index]
                size = 1
            result.append(word)
            index += size
        return result