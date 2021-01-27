from selenium import webdriver

chrome_driver = '/mnt/sdb/ccq/chromedriver'

# 进入浏览器设置
options = webdriver.ChromeOptions()
#谷歌无头模式
options.add_argument('--headless')
options.add_argument('--disable-gpu')
# options.add_argument('window-size=1200x600')
# 设置中文
options.add_argument('lang=zh_CN.UTF-8')
# 更换头部
options.add_argument('user-agent="Mozilla/5.0 (iPod; U; CPU iPhone OS 2_1 like Mac OS X; ja-jp) AppleWebKit/525.18.1 (KHTML, like Gecko) Version/3.1.1 Mobile/5F137 Safari/525.20"')
#设置代理
# if proxy:
#     options.add_argument('proxy-server=' + proxy)
# if user_agent:
#     options.add_argument('user-agent=' + user_agent)

browser = webdriver.Chrome(chrome_driver, chrome_options=options)
url = "http://www.baidu.com/"
browser.get(url)
html = browser.find_element_by_xpath("//*").get_attribute("outerHTML")
browser.quit()