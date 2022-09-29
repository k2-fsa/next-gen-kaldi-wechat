# Pruned RNN-T 何以又快又好

> **NGK**小组毕竟不是~~香港记者~~，不能每周都搞一个大新闻。
> 近期有个别同学在交流群里问 Pruned RNN-T 的细节，这周就深入一点挖挖这个旧~~坟~~(闻）吧。

之前我们发过一篇文章，介绍了如何用 Pruned RNN-T 进行“多、快、好、省” 的训练，复习资料在这里 [多快好省的 RNN-T 训练](https://mp.weixin.qq.com/s/bgJHwHp0PyFy0pWGVWvv0w), 本文将更深入的讲解 Pruned RNN-T 的实现细节。

> 本文不会包含完整的公式推导，旨在帮助大家更好的理解原理，看懂代码。更多的细节请阅读[论文](https://arxiv.org/pdf/2206.13236.pdf "论文") 和代码[rnnt_loss.py](https://github.com/k2-fsa/k2/blob/master/k2/python/k2/rnnt_loss.py "rnnt_loss.py")

## 训练 RNN-T 模型慢在哪？

要快速地训练一个模型，要么一次迭代多计算几条数据（batch size 大），要么单次迭代的速度快一些。很遗憾，在常规的 RNN-T 中这两点都很难做到，其中的症结就是 RNN-T 模型最后一层的输出是一个 4-D 的向量，即 `(N, T, U, V)` 向量。这样一个大的向量需要占据很大的显存，导致没法使用大的 batch size 来训练，另外，如此大的向量也造成 joiner 网络的计算量非常大，从而增加单次迭代的时间。

## Pruned RNN-T 为什么能快？

Pruned RNN-T 快的秘诀在于解决了这个四维向量的问题，在 [多快好省的 RNN-T 训练](https://mp.weixin.qq.com/s/bgJHwHp0PyFy0pWGVWvv0w) 一文中，我们介绍了，通过 Prune 的策略将四维 `(N,T,U,V)`向量剪裁至 `(N, T, S, V)`, 其中 $U \gg S$。并且给出了 benchmark 的数据，相比于 torchaudio 中的 rnnt_loss, Pruned RNN-T 内存使用量仅为其约 **1/6**，计算速度却为其约**10 倍**，在上述的两个问题上都取得了巨大进步。内存使用的降低是比较直观的，因为矩阵变小了，速度的提升从何而来呢？关键在 joiner 网络上，上述的 benchmark 中是包含 joiner 网络的，Pruned RNN-T 进入 joiner 网络的是更小得多的 `(N，T，S，V)`向量 ，所以 joiner 网络里面的非线性层和 Linear 层的计算量大大减小，从而实现加速。  
当然，Pruned RNN-T 的快还部分来源于它高效的代码实现，下面我们还会提到。

## 如何 Prune？

为何能够对 RNN-T 的计算进行 Prune，在[多快好省的 RNN-T 训练](https://mp.weixin.qq.com/s/bgJHwHp0PyFy0pWGVWvv0w) 一文中已经交代得比较清楚了，简单说就是音频和文本的单调对应特性决定了 lattice 中的大多数节点对最终的 Loss 几乎没有贡献，放一张论文里的图给大家复习一下。
![](https://files.mdnice.com/user/32622/4d213dbb-952a-46dc-a9c8-8cb6c2ff0dae.png)

<center> 图(1)

### 平凡联合网络

为了确定剪裁的边界，我们提出了一个**“平凡联合网络”**（`trivial joiner`）的概念，这个 `trivial joiner` 是 encoder 和 predictor 的简单相加，即 `am + lm`。使用这样一个简单的 joiner 网络是为了在**不生成四维向量**的情况下得到一个 lattice（细节我们在下面的代码实现中介绍），以便在这个 lattice 上求得剪裁边界。下图是 Pruned RNN-T 计算的流程图，我们实际上计算了两次损失函数，一次是在上述的 `trivial joiner` 上，一次是在正常的包含非线性层的 joiner 上（下图中的 s_range 就是上面提到的 S）。

![](https://files.mdnice.com/user/32622/822a6ed9-132c-4d7f-85cb-7b6ff86a30c0.png)

<center> 图(2)

在一个 lattice 中，每一个节点包含了两个概率，即 $ y(t,u)$ 和 $\varnothing(t, u)$ , $ y(t, u) $ 表示在第 $t$ 帧给定 $ y_{0..u} $ 的情况下发射 $y_{u+1}$ 的对数概率，$ \varnothing(t, u) $ 则代表在第 $t$ 帧给定 $ y_{0..u} $ 的情况下发射 $blank$ 的对数概率。由于 `trivial joiner` 是个简单的相加，所以我们不需要在相加之后的向量中来获取这两个概率，只需分别在 $am$ 和 $lm$ 中获得这两个概率，然后将 $am $ 和 $lm $ 中得到的概率分别加起来就行。获取概率的操作就是个简单的查询，在代码中使用 `torch.gather` 来实现，这个过程和乘法分配律非常相似。

> 注：两个 shape 不一样的向量相加得先统一 shape，即 `logit = am.unsqueeze(2) + lm.unsqueeze(1)`,所以如果相加之后再获取概率，我们就不得不生成一个四维向量。

### 剪裁边界的确定

有了 $y(t, u)$ 和 $\varnothing(t, u)$ 之后我们就有了 lattice, 可以计算损失函数了，和其他实现一样，在计算损失函数的同时我们也会一并计算 $y(t,u)$ 和 $ \varnothing(t,u) $ 的梯度，这和其他方法使用的前向后向算法并无二致。我们剪裁的目的是尽可能多的保留梯度。在论文中我们讨论了两种计算方式，一种是全局最优方案，即，遍历所有可能边界，这显然不现实。第二种是局部最优方案，即，保证剪裁后每一帧尽可能多将梯度保留，计算方法如下所示：
$$p_t =  \operatorname{argmax}_{p=0}^{U{-}S{+}1}( -y'(t, p - 1) + \sum_{u=p}^{p{+}S{-}1} \varnothing'(t, u))$$
上式通过最大化保留路径里的梯度来获得边界。如上图 1 所示，如在第 3 帧以 $u=2$ 为剪裁边界，每帧保留 4 个 symbol，那么能够保留的梯度就是图中四条绿线的值的和减去红线的值，之所以要减去红线，是因为绿线值中已经包含了红线值，而红线的值将随着点`(3,1)`被 `prune` 掉。  
当然，这样得到的边界还有一些缺陷，必须符合一些条件才能保证保留下来的路径的完整性，主要有以下三个约束，即，`端点约束`、`单调约束`和`连续约束`。其中连续约束的实现非常巧妙，感兴趣的同学可以在 k2 的代码中搜索 `_adjust_pruning_lower_bound` 函数，有非常详细的注释。

$$
  \begin{align}
  & 0  \leq p_t  \leq U-S+1 \\
  & p_t            \le p_{t+1} \\
  & p_{t+1} - p_t \le S
  \end{align}
$$

剪裁的过程是非常简单的查询操作，把在边界里的点挑出来即可，代码中用 `torch.gather` 实现。

## Pruned RNN-T 为什么能好？

理论上 prune 操作总会带来一些信息的损失，Pruned RNN-T 怎么还能更好呢？这主要得益于 rnnt_loss_simple 的加入，simple loss 的加入相当于使用了额外 joiner，起到了一定的正则化的作用。另外，我们还实现了一个带平滑的 rnnt_loss_simple 版本，即 rnnt_loss_smoothed。平滑的版本进一步将声学部分（`am`）和语言学部分（`lm`）从 `trivial joiner` 里面剥离开来，这样便于根据需要设定不同的 am 和 lm 权重。在 Icefall 的实验中，我们发现给 lm 设置一个单独的权重（0.25）, 即让 predictor 网络更像一个独立的语言模型，可以提升模型的准确性，而给 am 单独设置权重没能取得提升，甚至还有下降，所以目前 am 的权重默认值为 0。

## 道理我都懂,代码怎么写？

看到这里大家应该对原理有个了解了，那么具体怎么来实现呢？这里主要分析比较关键的两点，前向后向计算的加速和归一化的实现，一个工程相关，另一个数学相关。至于边界的剪裁部分，代码里的注释很详细，应该没有阅读障碍，这里不赘述。

### 前向后向计算

这部分是在给定 lattice 上计算损失函数和对应梯度，核心的实现在 `k2/python/csrc/torch/mutual_information_cuda.cu` 中，为啥我们进行了两次损失函数计算还能获得那么大的加速，一部分得益于高效的实现。
在 cuda 的实现中，我们先将 lattice 分成 32 \* 32 的块，让 cuda 中每个线程块（`thread block`)负责其中一个块的计算。如下图所示，由于每个数据块的计算都要依赖其左边和下边的块，所以我们将顺次计算`(0,0) -> (1,0) (0,1) -> (0,3)(1,2)(2,1)(3,0)...`, 在每个块的内部，我们首先将对应的数据，即 $y(t,u)$ 、 $\varnothing(t,u)$ 和 $alpha$ 向量（代码里为 $p$ 向量）载入块共享内存（类似 cache），每个 cuda 线程块内部包含非常多线程，所以这个过程是并行的。然后再在块内实现上述类似的计算，即`(0,0) -> (1,0) (0,1) -> (0,3)(1,2)(2,1)(3,0)...`迭代，由于块内从 cache 读取数据，并且只做简单的 $logadd$ 操作，所以非常快。

![](https://files.mdnice.com/user/32622/514f3743-9e03-4e0a-a755-61f4ac9c5878.png)
<center> 图（3）
  
> 注：上述两次 `(0,0) -> (1,0) (0,1) -> (0,3)(1,2)(2,1)(3,0)...`一次是在块这个粒度，一次是在块内元素的粒度。

这种分块计算的策略，一是实现了更高的并行度，二是将数据读取和计算分开进行，有效利用高速 cache，从而达到效率的提升。
> 读懂上述代码需要一些 cuda 线程模型和内存模型的知识，读一下 [CUDA C Programming Guide](https://docs.nvidia.com/pdf/CUDA_C_Programming_Guide.pdf "CUDA C Programming Guide") 的第二章就够用了。

### 怎么做归一化？
归一化的操作估计是大家看代码过程中比较困惑的，一顿 $exp$、$log$ 和矩阵乘法的操作怎么就实现了归一化呢？理解归一化先要明白 $y(t,u)$ 和 $\varnothing(t,u)$ 是什么，上面我们讲到了他们分别是发射 $symbol$ 和 $blank$ 的对数概率。一般情况下他们是在 joiner 的输出`(N，T，U，V)`向量 `V` 这个维度上做 $logsoftmax$ 操作得到的。
先来复习一下 $logsoftmax$ 操作：

$$
  \begin{aligned}
  \log \left(\frac{e^{x_{j}}}{\sum_{i=1}^{n} e^{x_{i}}}\right) &=\log \left(e^{x_{j}}\right)-\log \left(\sum_{i=1}^{n} e^{x_{i}}\right) \\
  &=x_{j}-\log \left(\sum_{i=1}^{n} e^{x_{i}}\right)
  \end{aligned}
$$

我们在代码里的所有归一化操作就是要实现这个 $logsoftmax$，上式中减号后面那一项就是代码中的 normalizer，是一个 $LogSumExp$。在 `trivial joiner` 中，我们不会真正计算 `am + lm` (避免生成 4-D 向量），所以不能直接在 `V` 这个维度上做 $LogSumExp$，而 `trivial joiner` 只是 `am` 和 `lm` 的线性相加，所以上面的 $logsoftmax$ 操作可以写成：
$$
  L_{trivial}(t, u, v) = L_\mathrm{enc}(t,v) + L_{dec}(u, v) - L_{normalizer}(t, u)
$$
其中：
$$
  L_{normalizer}(t, u) = \log \sum_v \exp \left( L_{enc}(t, v) + L_{dec}(u, v) \right)
$$
论文里说 normalizer 的操作可以看作是一个对数空间的矩阵乘法，即 $ L_{enc} * {L_{dec}}^T $ 在代码里我们先对 $L_{enc}$ 和 $ L_{dec} $ 做了 $exp$ 操作，于是他们两个的矩阵相乘即 $ \sum_v e^{enc} * e^{dec} $ ,也即 $ \sum_v e^{enc + dec} $，等于上式中的normalizer $log$ 后面的部分。当然，代码实现里还做了一些防止溢出的处理，这里不再详述。
在 rnnt_loss_smoothed 上，还要稍微复杂一点，但原理都是一样的，smoothed 版本实现的是：
$$
  L_{smoothed}(t, u, v) = \left(1-\alpha^{lm}-\alpha^{acoustic}\right) L_{trivial}(t, u, v) + \alpha^\mathrm{lm} L_\mathrm{lm}(t, u, v) + \alpha^\mathrm{acoustic} L_\mathrm{acoustic}(t, u, v)
$$ 
其中：
$$
\begin{align}
L_\mathrm{trivial}(t, u, v) &= \operatorname{LogSoftmax}_v \left( L_\mathrm{enc}(t, v) + L_\mathrm{dec}(u, v) \right)  \\
L_\mathrm{acoustic}(t, u, v) &= \operatorname{LogSoftmax}_v \left( L_\mathrm{enc}(t, v) + L_\mathrm{dec}^\mathrm{avg} \right)  \\
L_\mathrm{lm}(t, u, v) &=\operatorname{LogSoftmax}_v  L_\mathrm{dec}(u, v)
\end{align}
$$

$L_{dec}^{avg}$定义为：

$$L_\mathrm{dec}^\mathrm{avg}(u, v)  = \log \frac{1}{U+1} \sum_{u=0}^{U} \operatorname{Softmax}_v L_\mathrm{dec}(u, v)$$
切记，我们的目标是使 $ L_{smoothed}(t,u,v)$ 在 `V` 维度上归一化，$ L_{lm}(t,u,v) $ 比较直观，就是在单矩阵上做 $logsoftmax$ 操作，$ L_{acoustic}(t, u, v) $ 的实现和上面讨论的思路一样，只是把 $ L_{dec} $ 换成了 $ L_{dec}^{avg} $。而 $ L_{trivial} $、$ L_{acoustic} $ 和 $ L_{lm} $ 都归一化后，在三者中间的所有线性组合都是归一化的。

$$
\begin{align}
\sum_v L_{smoothed} &= \sum_v (1 - \alpha^{lm}-\alpha^{acoustic}) L_{trivial} + \alpha^{lm}L_{lm}+\alpha^{acoustic}L_{acoustic}\\
&= \sum_v L_{trivial} + \alpha^{lm}\sum_v(L_{lm} - L_{trivial}) + \alpha^{acoustic}\sum_v(L_{acoustic} - L_{trivial})\\
&=\sum_v L_{trivial} + \alpha^{lm}(\sum_vL_{lm} - \sum_vL_{trivial}) + \alpha^{acoustic}(\sum_vL_{acoustic} - \sum_v L_{trivial})
\end{align}
$$

## 总结
本文介绍了Pruned RNN-T 之所以能够又快又好的实现细节，首先我们复习了 Pruned RNN-T 能更快的原因（解决了 4-D 向量的问题），然后讲解了使用 `trivial joiner` 来进行剪裁的机制，最后就如何能在不生成四维向量的情况下计算 rnnt_loss_simple 的代码实现做了些说明，希望能帮助大家更好的读懂代码。

