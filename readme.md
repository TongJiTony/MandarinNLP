# MandarinNLP
**Introduction**
This a project focus on Mandarin text segmentation, using different methods.

My first step is to implement mechanical segmenter. I implement FMM and RMinM and compare these two methods.

Next, I introduce machine learning method and create basic HMMsegmenter, using pku_test_gold.txt as source to train the
model. I also create a tool file called generate.py to produce bmes training materials, such as bmes_train_pku.txt

Finally I develop a CRF(condition random fields) segmenter, with the same training materials.

p.s. The mandarin source comes from http://sighan.cs.uchicago.edu/bakeoff2005/

**The directory structure:**
-- segmenter.py (four segmenters code, all support segment() api)

-- test.py and mini_test.py (file test and sentence test)

-- generate.py (tools)

-- *test.txt (mandarin sentences sources)

-- *gold.txt (standard results)

**简介**
这是一个专注于中文文本分词的项目，使用多种方法。

我的第一步是实现机械分词器。我实现了前向最大匹配（FMM）和反向最小匹配（RMinM）并比较这两种方法。

接下来，我引入机器学习方法并创建了一个基于隐马尔可夫模型（HMM）的分词器，训练模型。我还创建了一个工具文件，用来生成BMES（Begin, Middle, End, Single）训练材料。

最后，我开发了条件随机场（CRF）分词器，并使用相同的训练材料。

p.s. 中文文本来源 2005 SIGHAN http://sighan.cs.uchicago.edu/bakeoff2005/

**目录结构：**
-- segmenter.py (四种分词器代码，全部支持segment() api)

-- test.py and mini_test.py (文件测试和句子测试)

-- generate.py (工具文件)

-- *test.txt (中文句子来源)

-- *gold.txt (标准结果)