import nltk
import requests
from bs4 import BeautifulSoup
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')

# 从TXT文件中读取数据
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

# 提取文本中的链接
def extract_links(text):
    # 使用正则表达式提取链接
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return urls

# 爬取评论数据、提取关键词
def scrape_comments(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 在这里编写代码以提取评论数据
        # 例如，查找评论元素并提取文本
        comments = soup.find_all('div', class_='comment')  # 假设评论以<div class="comment">标签包装

        # 初始化停用词列表
        stop_words = set(stopwords.words('chinese'))

        # 存储评论文本
        comment_text = ""

        for comment in comments:
            comment_text += comment.text

        # 分词并去除停用词
        words = word_tokenize(comment_text)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

        # 统计词频
        word_freq = Counter(words)

        # 打印评论文本
        print(f"评论文本：\n{comment_text}\n")

        # 打印关键词和它们的频率
        print(f"关键词和频率：{word_freq}\n")

    except Exception as e:
        print(f"无法爬取评论数据：{e}")

# 主函数
def main():
    file_path = r'.\source\舍友吧.txt'  # 替换为包含链接的TXT文件路径
    data = read_txt_file(file_path)
    links = extract_links(data)

    for link in links:
        print(f"爬取链接：{link}")
        scrape_comments(link)

if __name__ == "__main__":
    main()
