# test-lmdb
在尝试将NVAE用于wav-mel特征路上，所实验写的一些代码，留底备用，当作各个模块模板


* 横向比较当前所能见到的 griffin-lim 算法，将 mel 转 mag 转 wav 的路线代码，最好用的就是 [这套](https://github.com/Lukelluke/test-lmdb/blob/master/third_best_mel_griffin/%E6%88%90%E5%8A%9F%E7%89%88%E6%9C%AC.py)

* 由于 Tacotron 几份代码实现，mel 只作为生成 mag 过程中的监督，而非最后生成 mel，所以大多只实现 mag2wav

* mel2mag2wav路线上，直观理解是 给 mel 直接 **左乘** 一个 **逆mel变换**， 这个步骤如果用 numpy 的求伪逆函数 [np.linalg.pinv](https://github.com/Lukelluke/test-lmdb/blob/master/mel_mag/audio2.py#L270),
效果不如手动处理的版本，（在 n_mels==512时，两套相差不大，当n_mels开始减小时，e.g ==256，伪逆的方法就开始出现噪音了，在 n_mels==80 时， 手算法效果，噪音要比 伪逆 方法要低）
所以可以当作 **griffin-lim ：mel2wav 模板使用**

***

* 其他关于 LMDB 特性，慢慢道来，原理是存乘一张 {key:value} 键值对s，其中数据类型为 二进制，需要手动做 encode（）

* 数据的还原，从 LMDB 中取出来，有两条路线： 1. 二进制到numpy，简单； 2. 加上 buffers=True 参数，就变成了string串，网上资料不多，注释看的稀里糊涂，
效果就是：（1，80，80）的数据，不带buffers参数，返回shape为 （6400，），加上 buffers，返回就变成了 （25600，），就很神奇，先不管了。

***

