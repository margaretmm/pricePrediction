import requests
import re
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import csv
import time

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}

def get_one_page(url):
    try:
        response = requests.get(url,headers = headers)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        return None

def parse_one_page(content):
    try:
        soup = BeautifulSoup(content,'html.parser')
        items = soup.find('div',class_=re.compile('js-tips-list'))
        for div in items.find_all('div',class_=re.compile('ershoufang-list')):
            yield {
                #'Name':div.find('a',class_=re.compile('js-title')).text,
                'Address':div.find('span',class_=re.compile('area')).text.strip().replace(' ','').replace('\n',''),
                'Rooms': div.find('dd', class_=re.compile('size')).contents[1].text.strip(),#tag的 .contents 属性可以将tag的子节点以列表的方式输出
                'Area':div.find('dd',class_=re.compile('size')).contents[5].text.strip(),
                'Towards':div.find('dd',class_=re.compile('size')).contents[9].text.strip(),
                'Floor':div.find('dd',class_=re.compile('size')).contents[13].text.replace('\n','').strip(),
                'Decorate':div.find('dd',class_=re.compile('size')).contents[17].text.strip(),
                'Feature':div.find('dd',class_=re.compile('feature')).text.replace('\n','_').strip(),
                'TotalPrice':div.find('span',class_=re.compile('js-price')).text.strip()+div.find('span',class_=re.compile('yue')).text.strip(),
                'Price':div.find('div',class_=re.compile('time')).text.strip()
            }
            #有一些二手房信息缺少部分信息，如：缺少装修信息，或者缺少楼层信息，这时候需要加个判断，不然爬取就会中断。
        if div['Address','Rooms', 'Area', 'Towards', 'Floor', 'Decorate','Feature', 'TotalPrice', 'Price'] == None:
            return None
    except Exception:
        return None

def main():
    area=['binjiang']
    #area=['xihu', 'shangcheng', 'gongshu', 'xiacheng', 'jianggan', 'binjiang', 'xiaoshan', 'yuhang', 'linan']
    #area = ['fuyang', 'jiande']
    fieldnames = ['Address', 'Rooms','Area', 'Towards', 'Floor', 'Decorate', 'Feature', 'TotalPrice', 'Price']
    age=['g2']# ,'g3','g4' <3,<5,6-10,>10
    for a in area:
        for g in age:
            with open('Data_' + a +'_'+g+'.csv', 'a',encoding="utf-8", newline='') as f: # Data.csv 文件存储的路径,如果默认路径就直接写文件名即可。
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(1,12):
                    url = 'http://hz.ganji.com/fang5/'+a+'/'+g+'l1o{}/'.format(i)
                    content = get_one_page(url)
                    print('第{}页抓取完毕'.format(i))
                    lstDiv=parse_one_page(content)
                    for div in lstDiv:
                        print(div)
                        writer.writerow(div)
                        time.sleep(3)#设置爬取频率，一开始我就是爬取的太猛，导致网页需要验证。

if __name__=='__main__':
    main()