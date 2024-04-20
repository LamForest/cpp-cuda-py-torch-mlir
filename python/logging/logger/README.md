# 尽量不要定义继承logging.Logger的类

这里实现了2种自定义formatter的logger：
- custom_logger_class：通过继承logging.Logger来实现自定义的logger类，然后实例化该类，获得一个logger，并设置formatter。
- get_logger：通过logging.getLogger()来获得一个logger（类型为内置的logging.Logger），然后设置formatter。

除了自定义Logger类和通过getLogger()获得logger外，设置formater的方式是完全一样的。然而，其中自定义Logger类的方式存在一个bug：setLevel只有第一次生效，后续的setLevel不会生效。getLogger获得的则logger不存在此问题。


## 测试
复现方式为 `python test.py`
测试代码：
```py
    print("First time set loglevel, setting to INFO")
    logger.setLevel(logging.INFO)
    logger.info('info 111')
    logger.debug('debug 111')
    print("set logger.level to DEBUG")
    logger.setLevel(logging.DEBUG)
    logger.info('info 222')
    logger.debug('debug 222')
```

```sh
╰─ python test.py                                                     ─╯
--------- CustomLoggerClass ----------
First time set loglevel, setting to INFO
2024-04-21 02:10:26,292-custom_logger_class-INFO-(test.py:7) info 111 
set logger.level to DEBUG
2024-04-21 02:10:26,292-custom_logger_class-INFO-(test.py:11) info 222 



--------- get_logger ----------
First time set loglevel, setting to INFO
2024-04-21 02:10:26,293-get_logger-INFO-(test.py:7) info 111 
set logger.level to DEBUG
2024-04-21 02:10:26,293-get_logger-INFO-(test.py:11) info 222 
2024-04-21 02:10:26,293-get_logger-DEBUG-(test.py:12) debug 222 
```

观察输出logger，setLevel(logging.DEBUG)，自定义Logger类并没有生效


## 原因分析

#### 1. _cache并没有被清空
当调用logger.debug logger.info等函数时，logger会判断DEBUG或者info是否大于current level；也许觉得这个判断比较耗时，所以使用了一个cache来优化：
```py
    print("First time set loglevel, setting to INFO")
    logger.setLevel(logging.INFO)
    print(f"{logger._cache=}")
    
    logger.info('info 111')
    logger.debug('debug 111')
    print(f"{logger._cache=}")
    
    print("set logger.level to DEBUG")
    logger.setLevel(logging.DEBUG)
    print(f"{logger._cache=}")
    
    logger.info('info 222')
    logger.debug('debug 222')
```

输出：
```sh
╰─ python test.py
--------- CustomLoggerClass ----------
First time set loglevel, setting to INFO
logger._cache={}#这里为空并不代表被清空，_cache在第一次logger操作前都是空的。
2024-04-21 02:19:15,029-custom_logger_class-INFO-(test.py:9) info 111 
logger._cache={20: True, 10: False}
set logger.level to DEBUG
logger._cache={20: True, 10: False}#并没有被清空
2024-04-21 02:19:15,029-custom_logger_class-INFO-(test.py:17) info 222 



--------- get_logger ----------
First time set loglevel, setting to INFO
logger._cache={}
2024-04-21 02:19:15,030-get_logger-INFO-(test.py:9) info 111 
logger._cache={20: True, 10: False}
set logger.level to DEBUG
logger._cache={}
2024-04-21 02:19:15,030-get_logger-INFO-(test.py:17) info 222 
2024-04-21 02:19:15,030-get_logger-DEBUG-(test.py:18) debug 222 
```

对应源码如下，位于`logging/__init__.py`：

```py
    def isEnabledFor(self, level):
        """
        Is this logger enabled for level 'level'?
        """
        if self.disabled:
            return False

        try:
            return self._cache[level]
        except KeyError:
            _acquireLock()
            try:
                if self.manager.disable >= level:
                    is_enabled = self._cache[level] = False
                else:
                    is_enabled = self._cache[level] = (
                        level >= self.getEffectiveLevel()
                    )
            finally:
                _releaseLock()
            return is_enabled

    def setLevel(self, level):
        """
        Set the logging level of this logger.  level must be an int or a str.
        """
        self.level = _checkLevel(level)
        self.manager._clear_cache()
```
`_cache`的内容为 `Level : True/False`的dict。每次setLevel后，都需要清空`_cache`。

观察输出，我们发现，自定义Logger类的logger的`_cache`，并不会在setLevel后被清空，而getLogger的logger的`_cache`，在setLevel后会被清空。

#### 2. 为什么_cache没有被清空
继续分析`self.manager._clear_cache`的源码，发现只有注册在`self.manager.loggerDict`中的logger才会被清空`_cache`。然而，自定义的Logger类并没有注册在`self.manager.loggerDict`中，所以没有被清空。
```py
    def _clear_cache(self):
        """
        Clear the cache for all loggers in loggerDict
        Called when level changes are made
        """

        _acquireLock()
        for logger in self.loggerDict.values():
            if isinstance(logger, Logger):
                logger._cache.clear()
        self.root._cache.clear()
        _releaseLock()
```