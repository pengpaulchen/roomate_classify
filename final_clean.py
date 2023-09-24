# -*- coding: utf-8 -*-
import re
import jieba
from collections import Counter
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt

# 下载NLTK的停用词数据
nltk.download('stopwords')
# Get Chinese stopwords
chinese_stopwords = set(stopwords.words('chinese'))


# Function to process an HTML file and return a list of cleaned words
def process_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Remove HTML tags and non-Chinese characters
    text = re.sub('<[^<]+?>', '', html)
    text = re.sub('[^\u4e00-\u9fa5]', '', text)
    text = re.sub('|\s|//|', '', text)
    text = re.sub(
        '舍友吧百度贴吧页面的基本信息用户的基本信息吧的基本信息方法参数错误临时打点统计永远为判断对或者的支持吗统计发送普通版本不同的项目标示参数名称如果为空则当前页面当前页面页面数据全局变量页面标示判断吧信息是否存在吧吧名用户信息是否存在用户用户名是否新用户主题信息是否存在一般在页才会有此变量贴子添加扩展属性发送统计参数发送统计网页资讯视频图片知道文库贴吧地图采购进入贴吧全吧搜索吧内搜索搜贴搜人进吧搜标签仅记录失败即可打点未添加图片防盗链的图片请求图片防盗链例子模版名称模块名称舍友贴吧协议隐私政策吧主制度意见反馈网络谣言警示冻结逻辑挪到设置完开关之后中使用舍友吧目录个人贴吧多次加载白名单需要优化舍友舍友',
        '', text)

    # Tokenize the text using jieba
    words = jieba.lcut(text)

    # Filter out stopwords and short words
    filtered_words = [word for word in words if word not in chinese_stopwords and len(word) >= 2]

    # Remove custom stopwords
    stopwords = ["百度", "页面", "信息", "版本", "用户", "参数", "用户名", "变量",
                 "扩展", "添加", '判断', '支持', '统计', '发送', '项目', '标示', '名称',
                 '空则', '数据', '全局变量', '加载', '白名单', '优化', '临时', '打点',
                 '永远', '名', '新', '主题', '页', '才', '会', '贴子', '属性', '广告',
                 '发表', '新贴', '发起', '投票', '发贴', '请', '遵守', '贴', '协议', '七条', '底线',
                 '贴', '投诉', '停止', '浮动', '标题', '内容', '发表', '后', '自动', '分享', '本贴',
                 '签名档', '查看', '发表', '退出', '全新', '创作', '体验', '视频', '号', '专属', '发布', '器',
                 '视频', '上传', '尊享', '创作', '权益', '开通', '即享', '创作', '中心', '服务', '专属', '流量',
                 '扶持', '优质', '视频', '内容', '更', '曝光', '机会', '现金', '收益', '机会', '参与', '视频',
                 '激励', '活动', '优质', '原创', '更', '机会', '分润', '计划', '传递', '后台', '抓取', '话题',
                 '温馨', '提示', '反馈', '帐号', '异常', '删贴', '时请', '提供', '文字', '形式', '帐号', '非',
                 '截图', '发生', '时间', '尽可能', '上传', '截图', '有助于', '贴', '更好', '解决', '贴', '加载',
                 '白名单', '优化', '看贴', '图片吧', '主', '推荐', '舍友', '日', '一二三四五', '六', '签到', '排名',
                 '今', '日本', '第个', '签到', '精彩', '明天', '努力', '排名', '签到', '人数', '一键', '签到', '签级',
                 '一键', '签到', '本月', '漏', '签次', '超级', '会员', '赠送', '张补', '签卡', '点击', '日历', '上漏',
                 '签', '日期', '即可', '补签', '连续', '签到', '天', '累计', '签到', '天', '超级', '会员', '单次', '月',
                 '赠送', '连续', '签到', '卡张', '连续', '签到', '卡月', '日漏', '签天', '已本', '已', '封禁', '今日',
                 '签到',
                 '签到', '经验', '奖励', '连续', '签到', '双倍', '经验', '加粗', '字体', '特权', '红色', '字体', '特权',
                 '一举',
                 '橙名', '签到', '奖励', '经验值', '手机', '签到', '额外', '奖励', '经验值', '连续', '签到', '奖励',
                 '经验值',
                 '双倍', '时', '加粗', '字体', '时', '红色', '字体', '橙名', '高亮', '显示', '条件', '点击', '签到',
                 '即可',
                 '条件', '连续', '签到', '天及', '条件', '连续', '签到', '天及', '中断', '条件', '连续', '签到', '天及',
                 '中断',
                 '条件', '连续', '签到', '天及', '中断', '一键', '签到', '倍', '经验', '客户端', '免费', '会员', '一键',
                 '签到',
                 '关注', '扫', '二维码', '下载', '客户端', '下载', '看', '高清', '直播', '皇冠', '身份', '红色', '显示',
                 '红名',
                 '签到', '六倍', '经验', '兑换', '会员', '赠送', '补', '签卡张', '经验', '书', '购买权', '详情', '会员',
                 '舍友',
                 '目录', '火热', '招募', '中', '点击', '首页', '上', '一页', '共有', '数个', '数篇', '数', '帖子',
                 '置顶', '帖子',
                 '内部', '类名', '上', '帖子', '帖子', '楼', '置顶', '帖子', '插件', '设置', '类', '用于', '检测',
                 '插件', '类', '真的', '见', '宿舍', '一个', '别人', '室友', '那种', '两个', '大学', '喜欢', '奇葩',
                 '天天',
                 '极品', '感觉', '不好', '讨厌', '求助', '女生', '有没有', '有时候', '不想', '几个', '每次', '学校',
                 '平时', '几天','寝室','寝室','早上','参加','特别','实在','受不了','房子','一句','发现']
    # Add your custom stopwords

    # Remove custom stopwords
    clean_words = [word for word in filtered_words if word not in stopwords]

    return clean_words


# Process and count word frequency for multiple HTML files
total_word_counts = Counter()

for page_num in range(1, 51):
    file_path = f'./spider/舍友第{page_num}页.html'
    clean_words = process_html(file_path)
    total_word_counts.update(clean_words)

# Print the most common words
print(total_word_counts.most_common(100))
word_freq_df = pd.DataFrame(total_word_counts.most_common(100), columns=['Word', 'Frequency'])
# Sort the DataFrame by frequency in descending order
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

# Save the DataFrame to an Excel file
word_freq_df.to_excel('result.xlsx', index=False, engine='openpyxl')

# Visualize the top 20 words
top_words = word_freq_df.head(20)
# Configure Matplotlib to use a Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei or your preferred Chinese font
plt.rcParams['axes.unicode_minus'] = False  # Ensure that minus signs (-) are displayed correctly

plt.figure(figsize=(12, 6))
plt.bar(top_words['Word'], top_words['Frequency'])
plt.xticks(rotation=45)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 20 Words Frequency')
plt.tight_layout()
plt.show()
