import requests
from bs4 import BeautifulSoup
import traceback
import re
import sqlite3


conn = sqlite3.connect('test.db')
cursor = conn.cursor()

def getHTMLText(url, code="utf-8"):
    try:
        r = requests.get(url)
        r.raise_for_status()
        r.encoding = code  # 手工查看code为utf-8 避免访问全网页，提高速度
        return r.text
    except:
        return ""

def getStockList(lst, stockURL):
    html = getHTMLText(stockURL, "GB2312")
    soup = BeautifulSoup(html, 'html.parser')
    a = soup.find_all('td')
    for i in a:
        href = i.find('a')
        try:
            href = href.attrs['href']
            lst.append(re.findall(r"[s][hz]\d{6}", href)[0])  # [0]!!!!!!!!
        except:
            continue
    # a = soup.find_all('a')
    # for i in a:
    #     try:
    #         href = i.attrs['href']
    #         lst.append(re.findall(r"[s][hz]\d{6}", href)[0])  # [0]!!!!!!!!
    #         print('----------')
    #         print(lst)    
    #     except:
    #         continue


def getStockInfo(lst, stockURL):
    count = 0
    print('数据采集中...')
    for stock in lst:
        url = stockURL + stock + ".html"
        html = getHTMLText(url)
        try:
            if html == "":
                continue
            infoDict = {}
            soup = BeautifulSoup(html, 'html.parser')
            stockInfo = soup.find('div', attrs={'class': 'stock-bets'})  # 具体看页面元素
            name = stockInfo.find_all(attrs={'class': 'bets-name'})[0]  #
            infoDict.update({'股票名称': name.text.split()[0]})
            keyList = stockInfo.find_all('dt')
            valueList = stockInfo.find_all('dd')
            data=[]
            data.append( name.text.split()[0].strip())
            for i in range(len(keyList)):
                key = keyList[i].text
                val = valueList[i].text.strip()
                data.append(val)
                infoDict[key] = val
            #print(data)
        except:
            continue
        cursor.execute("INSERT INTO SHARE VALUES ('%s', '%s', '%s','%s', '%s', '%s','%s',\
            '%s','%s', '%s', '%s','%s', '%s','%s', '%s', '%s','%s','%s', '%s', '%s', '%s','%s','%s')"%tuple(data))
        conn.commit()
    print('保存成功')
    

def main():
    #stock_list_url = 'http://quote.eastmoney.com/stocklist.html'
    stock_list_url = 'http://app.finance.ifeng.com/list/stock.php?t=ha&f=chg_pct&o=desc&p=1'
    stock_info_url = 'http://gupiao.baidu.com/stock/'
    slist = []
    page=5
    for p in range(page):
        stock_list_url = stock_list_url[:-1]+str(p+1)
        print(stock_list_url)
        getStockList(slist, stock_list_url)
    print(len(set(slist)))
    getStockInfo(set(slist), stock_info_url)
    print("over")
    conn.close()

main()
