#encoding:utf-8
import requests
class BaiduTiebaSpider:
    def __init__(self,tiebaname):
        self.tibaname=tiebaname
        self.headers={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36"}
        self.url_temp = "https://tieba.baidu.com/f?kw="+tiebaname+"&ie=utf-8&pn={}"
    def get_url_list(self):
        url_list=[self.url_temp.format(i*50) for i in range(50)]#仅爬取前50页
        return url_list
    def parse_url(self,url):
        print(url)
        response = requests.get(url, headers=self.headers)
        return response.content.decode()
    def save_html(self,response,page_num):
        file_path="./spider/{}第{}页.html".format(self.tibaname,page_num)
        with open(file_path,"w",encoding="utf-8") as f:
            f.write(response)
    def run(self):
        #1、构造url列表
        url_list=self.get_url_list()
        for url in url_list:
        #2、遍历，发送请求，获取响应
            response=self.parse_url(url)
        #3、保存
            page_num=url_list.index(url)+1
            self.save_html(response,page_num)

if __name__=="__main__":
    tiebaspider=BaiduTiebaSpider("舍友")
    tiebaspider.run()