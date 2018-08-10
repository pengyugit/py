# coding:utf8
from bs4 import BeautifulSoup
import urllib.parse 

url = "http://search.cnki.com.cn/search.aspx?q=cnn"
response = urllib.request.urlopen(url)
if response.getcode() == 200: 
    html_cont=response.read()
soup = BeautifulSoup(html_cont, 'html.parser')   


datas2 = {}
page  = soup.find( class_='page-sum')
datas2['sum'] =page.get_text()
nums  = soup.find_all('a', class_='pc')
now=soup.find('span', class_='pc')
print(now.get_text())
for num in nums:
    print(num.get_text()+'  '+num.get('href'))

datas=[]
all  = soup.find_all('div', class_='wz_content')
for string in all:
    data = {}
    item = string.find('a', target='_blank')#文章标题与链接
    data['paper_url']  = item.get('href')# 获取文章url
    data['title'] = item.get_text() # 获取文章标题
    year_count = string.find('span', class_='year-count')#获取文章出处与引用次数
    data['year_count'] = year_count.get_text()
    datas.append(data)


fout = open('output.html', 'w', encoding='utf-8')
fout.write('<meta charset="utf-8">')  #解决显示乱码
fout.write("<html>")
fout.write("<body>")
fout.write("<table>")

fout.write("<tr>")
fout.write("<td>{0}</td>".format(datas2['sum'] ))
fout.write("</tr>")

for data in datas:
    fout.write("<tr>")
    fout.write("<td>{0}</td>".format(data['paper_url']))
    fout.write("<td>{0}</td>".format(data['title']))
    fout.write("<td>{0}</td>".format(data['year_count']))
    fout.write("</tr>")
fout.write("</table>")
fout.write("</body>")
fout.write("</html>")