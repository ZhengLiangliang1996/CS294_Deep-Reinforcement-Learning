### Lecture 3 Introduction to RL
In the last chapter, we know that some methodology like $o_t$, state $s_t$, policy $\pi_\theta(a_t|o_t)$

we can get $o_t$ from $s_t$, then use the policy $\pi_\theta(a_t|o_t)$ to get $a_t$. finally we can use the transition function $p(s_{t+1}|s_t,a_t)$ 

- reward functions : generally it is the question about which action is better or worse $r(s, a)$

s : state <br/>
a : action <br/>
$r(s, a)$ : reward functions <br/>
$p(s'|s, a)$ : probability of transition function
<br/>
<br/>
##### Markov Decision process
- $M = \{S, A, \mathcal{T}, r \}$
Compare to the Markov Chain, Bellman added 2 different components in the function. FIrst one is the Action Space $A$. It is noticeable that the transition $T$becomes a tensor, ${\mathcal{T}}_{i,j,k}=p(s_{t+1}=i|s_t=j,a_t=k)$
, if $\mu_{t,j}=p(s_t=j)，\xi_{t,k}=p(a_t=k)$, exists an linear relationship which is ： $\mu_{t+1,i}=\sum_{j,k}\mathcal{T}_{i,j,k}\mu_{t,j}\xi_{t,k}$. Then we can see from this that state $s_{t+1}$ is only related to the current state $s_t$

##### Partially Observed Markov Decision Process
- A more general process is Partially Observed Markov Decision Process, POMDP). The fomular has been added 2 other components $\mathcal{M}=\{\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{T},\mathcal{E},r\}$, Observation space and emission probability. The emission probability decides the pro of $o_t$ given state $s_t$


- what is tensor
<div style="text-align:center">
<img src="https://pic3.zhimg.com/80/v2-6dd8366ecb7fdb05c65c284d2468321e_hd.jpg"
     alt="Markdown Monster icon" class="center"/>
</div>

1.单个的数值叫Scalar。也就是一个数字。比如说250这样的自然数。也可以是实数。下同。
2.一维的数组叫Vector，也就是向量。比如 {a1，a2，a3......an}。这里a1，a2...的值都是Scalar。
3.二维的数组叫Matrix，也就是矩阵。
4.三维以上的都叫Tensor，也就是张量。
________________________________________________________
作者：Jacky Yang
链接：https://www.zhihu.com/question/20695804/answer/159192844
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
________________________________________________________

##### The goal of RL
- Assume the policy is clear and parametrized by $\theta$, policy $\pi_\theta(a|s)$ can be gained by Neural network. Then we get $s'$ by applying the transition prob function $p(s'|s,a)$, and it becomes a cycle.
 <div style="text-align:center">
<img src="https://pic2.zhimg.com/80/v2-13724835585ecf7ce89a42a5918d0edd_hd.jpg"
     alt="Markdown Monster icon" class="center"/>
</div>
- Consider a finite state trajectory，$\tau=\{\mathbf{s}_1,\mathbf{a}_1,\ldots,\mathbf{s}_T,\mathbf{a}_T\}$, the probability of the trajectory is $p_\theta(\tau)=p(\mathbf{s}_1)\prod_{t=1}^T\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)p(\mathbf{s}_{t+1}|\mathbf{s}_t,\mathbf{a}_t)$ 连乘，理解看上图循环。初始状态$s1$不能控制，之后就由策略函数和当前状态决定。 Then our goal is to maximize the $\theta^*$ which is the parameter in reward function. $\theta^*=\arg\max_\theta\mathbf{E}_{\tau\sim p_\theta(\tau)}\left[\sum_tr(\mathbf{s}_t,\mathbf{a}_t)\right]$
- If the trajectory is infinite, according to the property that $\left[\begin{array}{l}\mathbf{s}_{t+1}\\\mathbf{a}_{t+1}\end{array}\right]=\mathcal{T}\left[\begin{array}{l}\mathbf{s}_t\\\mathbf{a}_t\end{array}\right]$, then after k, the transition will be $\left[\begin{array}{l}\mathbf{s}_{t+k}\\\mathbf{a}_{t+k}\end{array}\right]=\mathcal{T}^k\left[\begin{array}{l}\mathbf{s}_t\\\mathbf{a}_t\end{array}\right]$, since this is an inifite trajectory, we consider a stationary distribution: after certain amount of training, and the state will not change, $\mu=\mathcal{T}\mu$, 其中$\mu$表示的是特定的状态的policy，then becomes $(\mathcal{T}-\mathbf{I})\mu=0$, then $\mu$ is the eigenvector of $\mathcal{T}$ when eigenvalue is 1. 由于$\mathcal{T}$是一个随机矩阵，给定一些正则条件，这样的向量总是存在的，此时$\mu=p_\theta(\mathbf{s},\mathbf{a})$是其平稳分布。对于无限长度的问题，我们也可以对目标函数进行平均，$\theta^*=\arg\max_\theta\frac{1}{T}\sum_{t=1}^T\mathbf{E}_{(\mathbf{s}_t,\mathbf{a}_t)\sim p_\theta(\mathbf{s}_t,\mathbf{a}_t)}r(\mathbf{s}_t,\mathbf{a}_t)\rightarrow \mathbf{E}_{(\mathbf{s},\mathbf{a})\sim p_\theta(\mathbf{s},\mathbf{a})}r(\mathbf{s},\mathbf{a})$，它将完全由平稳分布下的情形所控制。

##### General Types of RL algorithm
 <div style="text-align:center">
<img src="https://pic1.zhimg.com/80/v2-882bb913a3508d175342763a0d183104_hd.jpg"
     alt="Markdown Monster icon" class="center"/>
</div>
<br/>
##### Q function and value function 
First we want to describe a reward expectation $\sum_{t=1}^T\mathbf{E}_{(\mathbf{s}_t,\mathbf{a}_t)\sim p_\theta(\mathbf{s}_t,\mathbf{a}_t)}r(\mathbf{s}_t,\mathbf{a}_t)$, which can be expanded as : $\mathbf{E}_{\mathbf{s}_1\sim p(\mathbf{s}_1)}[\mathbf{E}_{\mathbf{a}_1\sim \pi(\mathbf{a}_1|\mathbf{s}_1)}[r(\mathbf{s}_1,\mathbf{a}_1)+\mathbf{E}_{\mathbf{s}_2\sim p(\mathbf{s}_2|\mathbf{s}_1,\mathbf{a}_1)}[\mathbf{E}_{\mathbf{a}_2\sim \pi(\mathbf{a}_2|\mathbf{s}_2)}[r(\mathbf{s}_2,\mathbf{a}_2)+\ldots|\mathbf{s}_2]|\mathbf{s}_1,\mathbf{a}_1]|\mathbf{s}_1]]$, in which the first state is formed by the initial distribution, and the second depends on the first state and action etc, so what we need to do is to find a good $a_1$. Extracting the recursive part and we define 
$Q(\mathbf{s}_1,\mathbf{a}_1)=r(\mathbf{s}_1,\mathbf{a}_1)+\mathbf{E}_{\mathbf{s}_2\sim p(\mathbf{s}_2|\mathbf{s}_1,\mathbf{a}_1)}[\mathbf{E}_{\mathbf{a}_2\sim \pi(\mathbf{a}_2|\mathbf{s}_2)}[r(\mathbf{s}_2,\mathbf{a}_2)+\ldots|\mathbf{s}_2]|\mathbf{s}_1,\mathbf{a}_1]$. 如果我们知道这样一个函数，那么原来的问题就可以被简写为 $\mathbf{E}_{\mathbf{s}_1\sim p(\mathbf{s}_1)}[\mathbf{E}_{\mathbf{a}_1\sim \pi(\mathbf{a}_1|\mathbf{s}_1)}[Q(\mathbf{s}_1,\mathbf{a}_1)|\mathbf{s}_1]]$ ，我们对$\mathbf{a}_1$的选择的事实上就不依赖于其他的东西了。我们把这样的函数称为Q函数 (Q-function)，表现为在状态$\mathbf{s}_1$下，选择行动$\mathbf{a}_1$所能带来的收益函数的条件期望。如果Q函数已知，那么改进策略将非常容易：我们只需要挑选一个$\mathbf{a}_1$，使得Q函数最大化就行了，即$\pi(\mathbf{s}_1,\mathbf{a}_1)=I(\mathbf{a}_1=\arg\max_{\mathbf{a}_1}Q(\mathbf{s}_1,\mathbf{a}_1))$。同样也可以在其他步骤类似这样做法。

- Q函数：$Q^\pi(\mathbf{s}_t,\mathbf{a}_t)=\sum_{t'=t}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_t,\mathbf{a}_t]$，从t时刻状态为$\mathbf{s}_t$起，执行行动$\mathbf{a}_t$，之后根据给定策略决策，未来总收益的条件期望。
- 值函数 (value function)：$V^\pi(\mathbf{s}_t)=\sum_{t'=t}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_t]$，从t时刻状态为$\mathbf{s}_t$起，根据给定策略决策，未来总收益的条件期望。

<br/>
##### Tradeoffs when using RL algorithm
- policy gradient : get the gradient bashed on the parameter
- value-based : Estimate value function or Q function based on optimal policy
- actor-critic : Estimate value function or Q function bashed on current policy. And used this policy to get gradient policy  in this way to optimize。所以也可以看作是策略梯度法和值函数方法的一个混合体。
- Model-based RL : 它需要去估计转移概率来作为模型，描述物理现象或者其他的系统动态。有了模型以后，可以做很多事情。譬如可以做行动的安排（不需要显式的策略），可以去计算梯度改进策略，也可以结合一些模拟或使用动态规划来进行无模型训练。

- 首先是样本效率 (sample efficiency)，就是要有一个不错的策略效果需要橙色方块收集多少数据；其次是稳定性和易用性，主要体现在选择超参数和学习率等调参的难度，在不同的算法中这个难度可以差别很大。
- 离线(off policy)的意义是我们可以在不用现在的策略去生成新样本的情况下，就能改进我们的策略。其实就是能够使用其他策略生成过的历史遗留数据来使得现在的策略更优。
- 在线 (on-policy) 算法指的是每次策略被更改，即便只改动了一点点，我们也需要去生成新的样本。在线算法用于梯度法通常会有一些问题，因为梯度算法经常会走梯度步，些许调整策略就得生成大量样本。

各算法间的样本效率比较图
<div style="text-align:center">
<img src="https://pic3.zhimg.com/80/v2-48fd01f64686e8c5a04a2f968923ce62_hd.jpg"
     alt="Markdown Monster icon" class="center"/>
</div>

