#!/usr/bin/python
# -*- coding: utf-8 -*-
# from log import Logger
import datetime
from queue import Queue
import codecs
from urllib import parse
from urllib import request
import requests
import time
import socket
import string
import random
from bs4 import BeautifulSoup
import threading
import os
import json
import argparse
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.proxy import ProxyType

# logger = Logger().get_logger()

headers = {
    'Referer': 'https://xueshu.baidu.com/usercenter/data/authorchannel',
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36',
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,la;q=0.7,pl;q=0.6",
    "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
    'Sec-Fetch-Mode': 'cors',
    'X-Requested-With': 'XMLHttpRequest',
    'Cookie':'Hm_lvt_9bd71603ef6eba860b148df49491c5e2=1567402948; Hm_lpvt_9bd71603ef6eba860b148df49491c5e2=1567402948; BIDUPSID=96DFC362A07013B8C428C68939A2068D; PSTM=1522039303; MCITY=-289%3A; __cfduid=dabd47f0a3598285ff05cbdddcd9765071552456853; BAIDUID=ADD98BD25B1F8EA20A230917C5C6956F:FG=1; Hm_lvt_43172395c04763d0c12e2fae5ce63540=1563339091; BDUSS=FwMVVIaDE0NWNvb0tRczZPdHRSYklhUm54fm0tUDgzNW10bDJib0ppYWR2MWRkSVFBQUFBJCQAAAAAAAAAAAEAAAAHtJ1ZsKXfz7PC0KHV5gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJ0yMF2dMjBdd0; pgv_pvi=1292965888; cflag=13%3A3; BDRCVFR[A88o6x7IGkt]=mk3SLVN4HKm; delPer=0; BD_CK_SAM=1; H_PS_PSSID=; pgv_si=s9203152896; BD_HOME=0; Hm_lvt_f28578486a5410f35e6fbd0da5361e5f=1566961489,1566969211; Hm_lvt_d0e1c62633eae5b65daca0b6f018ef4c=1566969705; PSINO=6; Hm_lpvt_f28578486a5410f35e6fbd0da5361e5f=1567401489; BDRCVFR[w2jhEs_Zudc]=mbxnW11j9Dfmh7GuZR8mvqV; BDSVRTM=20; Hm_lpvt_d0e1c62633eae5b65daca0b6f018ef4c=1567404945'
}

user_agent = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36')

base_url = "https://xueshu.baidu.com"

dcap = dict(DesiredCapabilities.PHANTOMJS)
dcap["phantomjs.page.settings.userAgent"] = user_agent
dcap["phantomjs.page.settings.loadImages"] = False

phantomjs_driver = '/mnt/sdb/ccq/phantomjs-2.1.1-linux-x86_64/bin/phantomjs'


def run_time(func):
    def wrapper(*args, **kw):
        start = datetime.datetime.now()
        func(*args, **kw)
        end = datetime.datetime.now()
        # logger.info('finished in {}s'.format(end - start))
        print('finished in {}s'.format(end - start))

    return wrapper


class Crawler():
    def __init__(self, key_num=-1, thread_num=1, fpath="keyword.csv", start_num=0, end_num=5, retry=3, delay=2):
        self.queue = Queue()
        self.key_num = key_num
        self.thread_num = thread_num
        self.start_num = start_num
        self.end_num = end_num
        self.fpath = fpath
        self.retry = retry
        self.delay = delay
        # self.driver = webdriver.PhantomJS(executable_path='F:\文档\phantomjs-2.1.1-windows\bin\phantomjs.exe',
        #                                   desired_capabilities=dcap)

    def load_keywords(self):
        '''
		加载关键字

		:param num:
		:return:
		'''
        with codecs.open(self.fpath, 'r', encoding='utf-8') as f:
            if self.fpath.startswith('zh'):
                self.keywords = [key.strip().split(' ') for key in f.readlines() if key.strip() != '']
            else:
                self.keywords = list(set(key.strip().split(':')[-1] for key in f.readlines()))
            if self.key_num != -1:
                self.keywords = self.keywords[:self.key_num]

    def init_url_queue(self):
        '''
        初始化队列

        :return:
        '''

        for key in self.keywords:
            try:
                # # 分页查询
                # for page in range(self.start_num, self.end_num + 1):
                #     # 构造请求
                #     query = {}
                #     pn = page * 10
                #
                #     query['wd'] = parse.quote(key, safe=string.printable)
                #
                #     query['pn'] = str(pn)
                #     query = parse.urlencode(query)
                #
                #     if self.fpath.startswith('zh'):
                #         query = parse.unquote(query)
                #
                #     # 构造地址
                #     url = base_url + query
                #
                #     self.queue.put((key, url))
                query = {}
                query['author'] = parse.quote(key[0], safe=string.printable)
                query['affiliate'] = parse.quote(key[1], safe=string.printable)

                query = parse.urlencode(query)

                if self.fpath.startswith('zh'):
                    query = parse.unquote(query)

                # 构造地址
                url = base_url + '/usercenter/data/authorchannel?cmd=inject_page&' + query
                self.queue.put((key, url))

            except socket.timeout as e:
                # logger.error('出错信息：{}'.format(e))
                print('出错信息：{}'.format(e))
                continue
            except Exception as e:
                # logger.error('出错信息：{}'.format(e))
                print('出错信息：{}'.format(e))
                continue

    def download_html(self, url):
        '''
		下载网页

		:param url:
		:param retry:
		:return:
		'''
        try:
            # s = requests.session()
            # s.keep_alive = False

            # req = request.Request(url=url, headers=headers)
            # resp = request.urlopen(req, timeout=5)
            proxy = webdriver.Proxy()
            proxy.proxy_type = ProxyType.MANUAL
            proxy_list = [p.strip() for p in open('ProxyGetter/proxies.txt').readlines()]
            proxy.http_proxy = random.choice(proxy_list)
            proxy.add_to_capabilities(dcap)

            driver = webdriver.PhantomJS(executable_path=phantomjs_driver, desired_capabilities=dcap)
            # driver.implicitly_wait(10)
            # driver.start_session(dcap)

            driver.get(url)
            html_doc = driver.page_source


            # if resp.status != 200:
            #     # logger.error('url open error. url = {}'.format(url))
            #     print('url open error. url = {}'.format(url))
            # html_doc = resp.read()
            # if html_doc == None or html_doc.strip() == '':
            #     # logger.error('NULL html_doc')
            #     print('NULL html_doc')
            return html_doc
        except Exception as e:
            # logger.error("failed and retry to download url {} delay = {}".format(url, self.delay))
            print("failed and retry to download url {} delay = {}".format(url, self.delay))
            if self.retry > 0:
                time.sleep(self.delay)
                self.retry -= 1
                return self.download_html(url)

    def extract_author_code(self, query):
        return

    def extract_html_doc(self, html_doc):
        '''
		抽取网页

		:param html_doc:
		:return:
		'''
        records = []

        soup = BeautifulSoup(html_doc)
        result_div = soup.find('div', id="personalSearch_result")
        try:
            result_lists = result_div.find_all("div", class_="searchResultItem  noborderItem")
        except AttributeError:
            print('AttributeError')
            return None

        for i, result in enumerate(result_lists):
            record = []

            # logger.info("extract for item = {}  delay = {}".format(i, self.delay))
            print("extract for item = {}  delay = {}".format(i, self.delay))
            time.sleep(self.delay)

            # 抽取专家信息页面链接
            sub_href = result.find('a').get('href')
            total_href = "http://xueshu.baidu.com" + sub_href
            # logger.info('title = {}, url = {}'.format(paper_title, total_href))

            author_info_html = self.download_html(total_href)
            author_info = self.extract_author_info(author_info_html)

            records.append(author_info)

            # # 抽取作者信息
            # info = result.find('div', class_="sc_info")
            # authors_block = info.find('span').find_all('a')
            # authors_qs = [x.get('href').strip()[3:] for x in authors_block]
            #
            # for query, author_block in zip(authors_qs, authors_block):
            #     time.sleep(self.delay)
            #     query = query.strip()
            #     # logger.info("author_query = {}".format(query))
            #     query_dict = parse.parse_qs(query)
            #     if query.startswith('ueshu.baidu.com/usercenter'):
            #         # {'xueshu.baidu.com/usercenter/data/author?cmd': ['authoruri'],'wd': ['authoruri:(b2adafebea64e9b0) author:(张铃) 安徽大学人工智能研究所']}
            #         author = query_dict['wd'][0].split()[1].split(':')[1][1:-1]
            #
            #         organizations = [query_dict['wd'][0].split()[2]]
            #     elif query.startswith('wd=authoruri'):
            #         # wd=authoruri%3A%287ca9d3874e254001%29%20author%3A%28%E7%8E%8B%E6%96%87%E9%80%9A%29%20%E5%8C%97%E4%BA%AC%E5%B7%A5%E4%B8%9A%E5%A4%A7%E5%AD%A6%E5%9F%8E%E5%B8%82%E4%BA%A4%E9%80%9A%E5%AD%A6%E9%99%A2%E5%A4%9A%E5%AA%92%E4%BD%93%E4%B8%8E%E6%99%BA%E8%83%BD%E8%BD%AF%E4%BB%B6%E6%8A%80%E6%9C%AF%E5%8C%97%E4%BA%AC%E5%B8%82%E9%87%8D%E7%82%B9%E5%AE%9E%E9%AA%8C%E5%AE%A4&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson&sort=sc_cited
            #         author = query_dict['wd'][0].split()[1].split(':')[1][1:-1]
            #
            #         organizations = [query_dict['wd'][0].split()[2]]
            #     elif query.startswith('wd=author%'):
            #         # wd=author%3A%28%E4%BD%99%E5%87%AF%29%20%E7%99%BE%E5%BA%A6&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson
            #         # wd=author%3A%28%E5%BE%90%E5%86%9B%29%20%E5%93%88%E5%B0%94%E6%BB%A8%E5%B7%A5%E4%B8%9A%E5%A4%A7%E5%AD%A6%E6%B7%B1%E5%9C%B3%E7%A0%94%E7%A9%B6%E7%94%9F%E9%99%A2%E6%99%BA%E8%83%BD%E8%AE%A1%E7%AE%97%E7%A0%94%E7%A9%B6%E4%B8%AD%E5%BF%83&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson
            #         author = query_dict['wd'][0].split(')')[0][8:]
            #         if author == "":
            #             author = author_block.text  # 用关键词搜索页面下的作者文本表示,对于中文没差别，对于英文名字可能是缩写
            #
            #         organizations = self.extract_organizations(author)
            #     else:
            #         logger.error('query error : {}'.format(query))
            #         continue
            #
            #     # logger.info('author = {}'.format(author))
            #     # logger.info('organizations = {}'.format(organizations))
            #
            #     # 添加记录
            #     record.append((author, organizations))
            # records.append({'authors': record, 'author_info': author_info})
        return records

    def extract_author_info(self, html_doc):
        info = {}

        soup = BeautifulSoup(html_doc)
        base_info = soup.find('div', class_='person_baseinfo')
        name = base_info.find('div', class_='p_name').get_text().strip()
        affiliate = base_info.find('div', class_='p_affiliate').get_text().strip()

        info['name'] = name
        info['affiliate'] = affiliate

        props = base_info.find_all('li', class_='p_ach_item')
        for i, prop in enumerate(props):
            prop_name = prop.find('p', class_='p_ach_type c_gray').get_text().strip()
            prop_num = prop.find('p', class_='p_ach_num').get_text().strip()
            info[prop_name] = prop_name

        return info

    def begin(self):
        '''
		开始抓取

		:param keywords:
		:param start_page:
		:param end_page:
		:return:
		'''

        while not self.queue.empty():
            key, url = self.queue.get()

            # logger.info("search url = {} delay = {}".format(url, self.delay))
            print("search url = {} delay = {}".format(url, self.delay))
            time.sleep(self.delay)
            # https://xueshu.baidu.com/s?wd=Deep+Learning&tn=SE_baiduxueshu_c1gjeupa&cl=3&ie=utf-8&bs=Deep+Learning&f=8&rsv_bp=1&rsv_sug2=0&sc_f_para=sc_tasktype%3D%7BfirstSimpleSearch%7D

            # 文件打开
            url_dict = parse.parse_qs(url)
            # url_pn = url_dict['pn'][0]

            html_doc = self.download_html(url)

            if html_doc is None or html_doc.strip() == '':
                continue

            records = self.extract_html_doc(html_doc)

            if records is None or records == []:
                continue

            fmode = "a"
            fname = "{}.json".format('_'.join(key.split(' ')))
            fpath = "data/{}".format(fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) != 0:
                continue
            fp = codecs.open(fpath, fmode, encoding='utf-8')
            json.dump({key: records}, fp)
            fp.close()

            # logger.info("done with keyword = {} ".format(key))
            print("done with keyword = {} ".format(key))

    def run(self):
        for i in range(self.thread_num):
            th = MyThread(str(i), self)
            th.start()


class MyThread(threading.Thread):
    def __init__(self, name, target):
        threading.Thread.__init__(self)
        self.name = name
        self.target = target

    @run_time
    def run(self):
        # logger.info("create thread : {}".format(self.name))
        print("create thread : {}".format(self.name))
        self.target.begin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baidu scolar author index Crawler')
    parser.add_argument('--key-num', type=int, default=-1, metavar='N',
                        help='input number of keywords (default: 100)')
    parser.add_argument('--thread-num', type=int, default=1, metavar='N',
                        help='input number of thread (default: 10)')
    parser.add_argument('--page-num', type=int, default=1, metavar='N',
                        help='input number of pages (default: 1)')
    parser.add_argument('--input', type=str, default='zh_keyword.csv', metavar='S',
                        help='input keyword file path')

    # init_parser = parser.add_mutually_exclusive_group(required=False)
    # init_parser.add_argument('--init', dest='init', action='store_true')
    # init_parser.add_argument('--no-init', dest='init', action='store_false')
    # parser.set_defaults(init=False)

    args = parser.parse_args()

    # logger.info("开始抓取:key_num={},thread_num={},page_num={}".format(args.key_num, args.thread_num, args.page_num))
    print("开始抓取:key_num={},thread_num={},page_num={}".format(args.key_num, args.thread_num, args.page_num))

    # queue size = [end-start+1] * key_nums

    crawler = Crawler(key_num=args.key_num, thread_num=args.thread_num, fpath=args.input, start_num=0,
                      end_num=args.page_num)  # 一个页面固定约 80 s

    crawler.load_keywords()
    crawler.init_url_queue()

    time.sleep(5)

    crawler.run()
