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

- Which can be used to subsitute $\hat{Q}_{i,t}$, then $\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\left[\nabla_\theta\log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})Q(\mathbf{s}_t,\mathbf{a}_t)\right]$ will be a better esitimate of the policy gradient. But actually in this case we only used one sample, which will result in large vairance, if we used infinite sample, then the variance will be small.
- So we got the same bottleneck like lecture 3, then we can introduce baseline again, we can used value-function $V^\pi(\mathbf{s}_t)=\mathbf{E}_{\mathbf{a}_t\sim\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)}[Q^\pi(\mathbf{s}_t,\mathbf{a}_t)]$, which means that the expectation of Q-function given state, the reason why we used value-function here is that remember in the last lecture, we used $b_t$ which is also an expectation. So after applying this technique, the original one becomes:
$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\left[\nabla_\theta\log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})A^\pi(\mathbf{s}_t,\mathbf{a}_t)\right]
$$

- $A^\pi(\mathbf{s}_t,\mathbf{a}_t)=Q^\pi(\mathbf{s}_t,\mathbf{a}_t)-V^\pi(\mathbf{s}_t)$ is the advantage function indicating under the state $\mathbf{s}_t$, how much advantageous is reward of taking action $\mathbf{a}_t$ over that of taking the avarage action.

补充，给优势函数对a取期望

#### Actor-Critic Algorithm:
- review what's the stage of classical reinforcement learning, frist we get sample, so we changed a little bit in the second stage, basically what we're trying to do now is to fit a value-function, one of $Q^\pi,V^\pi,A^\pi$, so we transfer our problem from 'estimate reward' to 'fit our model'. 
- A straightforward way is to fit $A^\pi$ since this is what we want in graident expression. There is a trick here that in the Q-function $Q^\pi(\mathbf{s}_t,\mathbf{a}_t)=\sum_{t'=t}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_t,\mathbf{a}_t]$ which only depends on state and action, then the sample input is only  dimensional space of Cartesion Product. if the dimension gets bigger, our vairance will be larger. Based on the relation between Q-function and value-function, $Q^\pi(\mathbf{s}_t,\mathbf{a}_t)=r(\mathbf{s}_t,\mathbf{a}_t)+\sum_{t'=t+1}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_t,\mathbf{a}_t]$, since in the settings, given current state and action, the current reward can be calculated. Then the first term is known, the second term is the expectation of the value function: $\mathbf{E}_{\mathbf{s}_{t+1}\sim p(\mathbf{s_{t+1}}|\mathbf{s}_t,\mathbf{a}_t)}[V^\pi(\mathbf{s}_{t+1})]$. Expectation means  average the reward wrt the state and action in the whole trajectory. Meanwhile, we  only used 'next' state here, then our Q function can be written as $Q^\pi(\mathbf{s}_t,\mathbf{a}_t)\approx r(\mathbf{s}_t,\mathbf{a}_t)+V^\pi(\mathbf{s}_{t+1})$. At the same time, advantageous function can be writte then as $A^\pi(\mathbf{s}_t,\mathbf{a}_t)\approx r(\mathbf{s}_t,\mathbf{a}_t)+V^\pi(\mathbf{s}_{t+1})-V^\pi(\mathbf{s}_t)$. So the problem is siplified as fitting value function V. So the neural network input can only be the state space: what we're trying to do is to train a neural network with parameter $\phi$,using input $s$ and ouput estimate our Value function $\hat{V}^\pi(\mathbf{s})$. For the sake of convinience, most of the actor-critic is to fit value function. The process is called 'Policy Evaluation'.

### Policy Evaluation
- This algorithm is not trying to optimize policy, instead, it tries to evaluate how good is the policy from a certain given starting state. 
- The value function is $V^\pi(\mathbf{s}_t)=\sum_{t'=t}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_{t'}]$, objective function is $J(\theta)=\mathbf{E}_{\mathbf{s}_1\sim p(\mathbf{s}_1)}[V^\pi(\mathbf{s}_1)]$ is just a expectation, so fitting value function can be used to calculate objective function at the same time. 
- Same as policy gradient, we still used Monte Carlo, approximate trajectory to estimate $V^\pi(\mathbf{s}_t)\approx\sum_{t'=t}^Tr(\mathbf{s}_{t'},\mathbf{a}_{t'})$.
- NN sometimes could be biased, think of an example, if we start from state both are close, which will result in very different bias. For NN, same input will result in same output because NN is just a function approximator. For deterministic mdel, its ouput is well-defined, which means 1 input will be correspond with one ouput. NN has many versions, NN will fit many samples and average them, the more sample is, the better the NN fit, but if there is a gap between same state, then the answer will be every defferent, the variance is still small anyways.
- Training NN: 
    - Collecting training data $\left\{\left(\mathbf{s}_{i,t},y_{i,t}:=\sum_{t'=t}^T r(\mathbf{s}_{i,t},\mathbf{a}_{i,t})\right)\right\}$, error function is $\mathcal{L}(\phi)=\frac{1}{2}\sum_i\left\Vert\hat{V}_\phi^\pi(\mathbf{s}_i)-y_i\right\Vert^2$ for finite number of sample. 
    - $y_{i,t}=\sum_{t'=t}^T\mathbf{E}_{\pi_\theta}[r(\mathbf{s}_{t'},\mathbf{a}_{t'})|\mathbf{s}_{i,t}]\approx r(\mathbf{s}_{i,t},\mathbf{a}_{i,t})+V^\pi(\mathbf{s}_{i,t+1})\approx r(\mathbf{s}_{i,t},\mathbf{a}_{i,t})+\hat{V}_\phi^\pi(\mathbf{s}_{i,t+1})$, first term after approxi is the estimate of Q using 1 trajectory, the second is to estimate using Neural Network.

#### Batch actor-critic algorithm
- from https://zhuanlan.zhihu.com/p/32727209
1. 运行机器人，根据策略$\pi_\theta(\mathbf{a}|\mathbf{s})$得到一些样本$\{\mathbf{s}_i,\mathbf{a}_i\}$，包括所处状态、行动和收益。
2. 使用样本收益之和拟合$\hat{V}^\pi_\phi(\mathbf{s})$。这一步样本可以做蒙特卡洛，也可以做自助法；拟合可以用最小二乘的目标函数。
3. 评估优势函数$\hat{A}^\pi(\mathbf{s}_i,\mathbf{a}_i)=r(\mathbf{s}_i,\mathbf{a}_i)+\hat{V}^\pi_\phi(\mathbf{s}_i')-\hat{V}^\pi_\phi(\mathbf{s}_i)$。
4. 放入策略梯度函数$\nabla_\theta J(\theta)\approx\sum_{t=1}^T\left[\nabla_\theta\log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)\hat{A}^\pi(\mathbf{s}_t,\mathbf{a}_t)\right]$。
5.走一个梯度步$\theta\leftarrow \theta+\alpha\nabla_\theta J(\theta)$。

