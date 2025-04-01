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