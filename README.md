# Reinforcement_Learning

## Random Variable
- a variable whose vales depend on outcomes of a random event
- Uppercase letter **X** for random variable.
- Lowercase letter **x** for an observed value.

|Random Variable|Possible Values|Random Events|Probabitites|
|:-:|:-:|:-:|:-:|
|X|0|coin head|P(X = 0) = 0.5|
|X|1|coin tail|P(X = 1) = 0.5|

## Probability Density Function(PDF)
- PDF provides a relative likelihood that  the value of the random variable would equal that sample.
- ex. Gaussian distribution

$$ p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

- Random variable **X** is in the domain **χ**
- For continuous distribution,
$$\int_{\chi}^{}p(x)dx=1$$
- For discrete distribution,
$$\sum {\scriptsize{x\in \chi}}^{} {\displaystyle p(x)}=1$$

## Expectation
- Random variable **X** is in the domain **χ**
- For continuous distribution, the expectation of f(X) is:
$$E[f(x)] = \int_{\chi}^{}p(x)f(x)dx=1$$
- For discrete distribution, the expectation of f(X) is:
$$E[f(x)] = \sum {\scriptsize{x\in \chi}}^{} {\displaystyle p(x)f(x)}=1$$


## Random Sample
- It is a subset of a larger population that is selected in a way that every member of the population has an equal chance of being chosen. 

## Terminology
- State **s**
- Action **a**
- Policy **π**
$π(a | s) = P(A = a | S = s)$
- Reward **r**
- state transition
$p(s' | s , a) = P(S' = s' | S = s , A = a)$


## Return
- Definition: Return(aka cumulative future reward)
$$U_t = R_t + R_{t+1} + R_{t+2} + R_{t+3} + ...$$
- Definition: Discounted return(aka cumulative discounted future reward)
**γ**: discount rate(tuning hyper-parameter)
$$U_t = R_t + \gamma R_{t+1} + \gamma^2R_{t+2} + \gamma ^3R_{t+3} + ...$$

#### At time step t, the return Ut is random
- Two sources of randomness:
1. Action can be random: $P[A = a | S = s] = π(a | s)$
2. New state can be random: $P[S' = s' | S = s, A = a] = p(s' | s, a)$

## Value Function Q(s, a)
### Action-value function
- Definition: Action-value function for policy π.
$$Qπ(s_t, a_t) = E[U_t|S_t = s_t, A_t = a_t]$$

### Optimal action-value function
- Definition: Optimal action-value function
$$Q^*(s_t, a_t) = max\ Qπ(s_t, a_t)$$

### State-value function
- Definition: State-value function
- Action are discrete
$$Vπ(s_t) = E_A[Qπ(s_t, A)] = Σ_aπ(a | s_t)‧Qπ(s_t, a)$$
- Action are continuous
$$Vπ(s
_t) = E_A[Qπ(s_t, A)] = ∫π(a | s_t)‧Qπ(s_t, a)$$

## Play game using reinforcement learning
- Observe state **s(t)**, make action **a(t)**, environment gives **s(t+1)** and reward **r(t)**
![](https://i.imgur.com/7SAPAfU.png)
- The agent can be controlled by either π(a | s) or Q^*^(s, a)

## Value-Based Reinforcement Learning
### Deep Q-Ntewor(DQN)
- Use neural network Q(s, a; w) to approximate Q*(s, a)
### Temporal Difference (TD) Learning
- Make a prediction: q = Q(w)
- Finish the trop and get target y
- Loss L = $$\frac{1}{2}(q-y)^2$$
- Loss L = $$\frac{1}{2}(Q(w)-y)^2$$
- Gradient: $$\frac{\delta L}{\delta w} = \frac{\delta q}{\delta w}\cdot \frac{\delta L}{\delta q} = (q - y) \cdot \frac{\delta Q(w)}{\delta w}$$
- Grandient descent: $$W_{t+1} = W_t - \alpha \cdot \frac{\delta L}{\delta w}\vert_{w=w_t}$$
## TD learning for DQN
- equation: $$T_{A\rightarrow C} \approx T_{A\rightarrow B} + T_{B\rightarrow C}$$
- In deep reinforcement learning: $$Q(s_t, a_t;w)\approx r_t + \gamma \cdot Q(s_{t+1}, a_{t+1};w)$$

$$U_t = R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + ...\\
=R_t + \gamma (R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + ...)\\
= R_t + \gamma \cdot U_{t+1}$$
- DQN's output, $$Q(s_t, a_t;w),\ is\ estimate\  of\ E[U_t]$$
- DQN's output, $$Q(s_{t+1}, a_{t+1};w),\ is\ estimate\  of\ E[U_{t+1}]$$
- Thus, $$Q(s_t, a_t;w) \approx
 E[R_t + \gamma \cdot Q(s_{t+1}, A_{t+1};w)]$$
 


## MAZE
![maze_1.gif](https://github.com/kerong2002/Reinforcement_Learning/blob/main/Book_L2/maze_1.gif)

## Taxi
![taxi.gif](https://github.com/kerong2002/Reinforcement_Learning/blob/main/GYM/taxi_class.gif)

