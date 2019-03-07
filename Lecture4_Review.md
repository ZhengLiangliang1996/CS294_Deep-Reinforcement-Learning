## Lecture 4 Actor Critic

#### Combination between policy gradient and value-function
1. The essence of gradient is to find the gradient of objective function 

$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\left[\nabla_\theta\log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})\hat{Q}_{i,t}\right]
$$
2. Reward to go is the $\hat{Q}_{i,t}=\sum_{t'=t}^Tr(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})$, which means take the gradient of policy from time t(causality)
- So now let's look a ${Q}_{i,t}$, which is reward estimation of the action $\mathbf{a}_{i,t}$ given the state $\mathbf{s}_{i,t}$. We're using Q here is because it's closely related to Q function. 需要补充
- 
如果我们要得到其估计量，可以求出我们模拟出来的一条轨迹的收益的后面一段也就是\hat{Q}_{i,t}。但是事实上我们在之前也说过，在同一个分布中抽取的轨迹可能也是千差万别的。用MDP的语言解释，可能是因为我们在之后根据策略函数分布随机选择了不同的动作，也有可能是选择了同一个动作但是由于系统环境的随机性导致下一个状态不同。

- A real expected reward is more complex than the result according to the action given state, so the infinite different situation should be integrated to get a real reward to go. 

$$
\hat{Q}_{i,t}\approx\sum_{t'=t}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_t,\mathbf{a}_t]
$$

- Which can be used to subsitute $\hat{Q}_{i,t}$, then $\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\left[\nabla_\theta\log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})Q(\mathbf{s}_t,\mathbf{a}_t)\right]$ will be a better esitimate of the policy gradient. But actually in this case we only used one sample, which will result in large vairance, if we used infinite sample, then the variance will be samll.
- So we got the same bottleneck like lecture 3, then we can introduce baseline again, we can used value-function $V^\pi(\mathbf{s}_t)=\mathbf{E}_{\mathbf{a}_t\sim\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)}[Q^\pi(\mathbf{s}_t,\mathbf{a}_t)]$, which means that the expectation of Q-function given state, the reason why we used value-function here is that remember in the last lecture, we used $b_t$ which is also a expectation. So after applying this technique, the origianl one becomes:
$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\left[\nabla_\theta\log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})A^\pi(\mathbf{s}_t,\mathbf{a}_t)\right]
$$

- $A^\pi(\mathbf{s}_t,\mathbf{a}_t)=Q^\pi(\mathbf{s}_t,\mathbf{a}_t)-V^\pi(\mathbf{s}_t)$ is the advantage function indicating under the state $\mathbf{s}_t$, how much advantageous is reward of taking action $\mathbf{a}_t$ over that of taking the avarage action.

补充，给优势函数对a取期望

#### Actor-Critic Algorithm:
- review what's the stage of classical reinforcement learning, frist we get samples, so we changed a little bit in the second stage, basically what we're trying to do now is to fit a value-function, one of $Q^\pi,V^\pi,A^\pi$, so we transfer our problem from 'estimate reward' to 'fit our model'.
- A straightforward way is to fit $A^\pi$ since this is what we want in graident expression. There is a trick here that in the Q-function $Q^\pi(\mathbf{s}_t,\mathbf{a}_t)=\sum_{t'=t}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_t,\mathbf{a}_t]$ which only depends on state and action, then the sample input is only  dimensional space of Cartesion Product. if the dimension gets bigger, our vairance will be larger. Based on the relation between Q-function and value-function, $Q^\pi(\mathbf{s}_t,\mathbf{a}_t)=r(\mathbf{s}_t,\mathbf{a}_t)+\sum_{t'=t+1}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_t,\mathbf{a}_t]$, 

to 22mins