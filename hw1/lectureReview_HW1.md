# CS294 Homework 1
#### Section 2 Behavioral Cloning
##### Parameters for experts are as followed:
###### <span style="color:green">Simply change the bash file: rollouts: 20 and ./ run it</span>
- Hopper-v2 : mean return 3777.8673 std of return 3.6949

- Ant-v2 : mean return 4802.90699 std of return 86.6941336401

- HalfCheetah-v2 : mean return 4156.0385 std of return 68.9829

- Humanoid-v2 : mean return 10429.0622 std of return 39.3558 need more iter

- Reacher-v2 : mean return -3.80316 std of return 1.57784

- Walker2d-v2 : mean return 5547.4035 std of return 40.12338

##### concrete concept of the behavioral cloning (imitation study)


 We can actually do the imitation game by applying the supervising learning, let's say we try to solve the autonumous driving problem, we can simply get data from aksing large amout of drivers to drive. data $o_t$ for overservation and action $a_t$. Then try to put them into a supervised learning algorithm and find the solution using ADAM or or something. then we get a strategt $\pi_\theta(a_t|o_t)$ . BUT the reality is that this is not a good idea.
<div style="text-align:center">
<img src="https://pic1.zhimg.com/80/v2-61c70dd8a08a4830c3411dba40f55f18_hd.jpg"
     alt="Markdown Monster icon" class="center"/>
</div>

Once we got a small error, then it will deviate further and furthur. This kind of deviation will accumulated when training.

NVIDIA had a research on this, they install left, center and right camera .

<div style="text-align:center">
<img src="https://pic4.zhimg.com/80/v2-2418365cab69f33917105ad124bb1407_hd.jpg"
     alt="Markdown Monster icon" class="center"/>
</div>

For a trajectory distribution $p(r)$, the whole process is $\tau:=(s_1, a_1, s_2, a_2,...,s_t,a_t)$. Then we can regard this as a probabilitic problem. We have data $o_t$ and $a_t$. When training, there is one distribution named $p_data(ot)$, which is the one we that matches the real situation. But in our strategy, $p_{\pi\theta}(o_t)$ is different from $p_data(ot)$. Instead of trying to let expected trajectory $p_{\pi\theta}(o_t)$ approximate $p_data(ot)$, what we can actually do is make $p_data(ot)$ approximate $p_{\pi\theta}(o_t)$. To do that, when drivers are driving, we're trying to get the data according to $\pi_\theta(a_t|o_t)$. Which is the intuition behind DAgger algorithm.

