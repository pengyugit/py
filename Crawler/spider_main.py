# coding:utf8
import html_outputer, url_manager, html_downloader,html_parser


class SpiderMain(object):
    def __init__(self):
        self.urls = url_manager.UrlManager()  #URL管理器
        self.downloder = html_downloader.HtmlDownloader()  #网页下载器
        self.parser = html_parser.HtmlParser()   #网页解析器
        self.output = html_outputer.HtmlOutputer() #输出器
        
    def craw(self, root_url):   #爬虫调度程序
        count = 1
        self.urls.add_new_url(root_url)  #添加URL管理器
        while self.urls.has_new_url():  #开始循环爬虫  如果有待爬取url
            try:
                print('第{0}个url'.format(count))
                new_url = self.urls.get_new_url()   #获取页面
                html_cont = self.downloder.download(new_url)  #下载页面
                new_urls, new_data = self.parser.parse(new_url, html_cont)
                print(len(new_urls))
                #下载后解析器解析数据得新的url列表和data
                self.urls.add_new_urls(new_urls) #添加url管理器
                self.output.collect_data(new_data) #收集数据
              
                if count == 10:
                    break
                count = count + 1
            except:
                print('craw failed')
        self.output.output_html()
            
    


if __name__=="__main__":
    root_url = "https://baike.baidu.com/item/Python"
    obj_spider = SpiderMain()
    obj_spider.craw(root_url)
    