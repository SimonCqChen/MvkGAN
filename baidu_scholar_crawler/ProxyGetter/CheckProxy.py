# -*- coding: utf-8 -*-

import sys
from getFreeProxy import GetFreeProxy
from Util.utilFunction import verifyProxyFormat

sys.path.append('../')

from Util.LogHandler import LogHandler

log = LogHandler('check_proxy', file=False)


class CheckProxy(object):

    @staticmethod
    def checkAllGetProxyFunc():
        """
        检查getFreeProxy所有代理获取函数运行情况
        Returns:
            None
        """
        import inspect
        member_list = inspect.getmembers(GetFreeProxy, predicate=inspect.isfunction)
        proxy_count_dict = dict()
        for func_name, func in member_list:
            log.info(u"开始运行 {}".format(func_name))
            try:
                proxy_list = [_ for _ in func() if verifyProxyFormat(_)]
                proxy_count_dict[func_name] = len(proxy_list)
            except Exception as e:
                log.info(u"代理获取函数 {} 运行出错!".format(func_name))
                log.error(str(e))
        log.info(u"所有函数运行完毕 " + "***" * 5)
        for func_name, func in member_list:
            log.info(u"函数 {n}, 获取到代理数: {c}".format(n=func_name, c=proxy_count_dict.get(func_name, 0)))

    @staticmethod
    def checkGetProxyFunc(func):
        """
        检查指定的getFreeProxy某个function运行情况
        Args:
            func: getFreeProxy中某个可调用方法

        Returns:
            None
        """
        func_name = getattr(func, '__name__', "None")
        log.info("start running func: {}".format(func_name))
        count = 0

        #save
        file="proxies.txt"
        f=open(file,'a')

        for proxy in func():
            if verifyProxyFormat(proxy):
                log.info("{} fetch proxy: {}".format(func_name, proxy))
                f.write(proxy)
                f.write('\n')
                count += 1
        f.close()
        log.info("{n} completed, fetch proxy number: {c}".format(n=func_name, c=count))


if __name__ == '__main__':
    CheckProxy.checkAllGetProxyFunc()
    CheckProxy.checkGetProxyFunc(GetFreeProxy.freeProxyFirst)
