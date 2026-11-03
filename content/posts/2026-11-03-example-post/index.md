---
title: "A Gentle Look at Gradient Descent"
date: 2026-11-03
draft: false
math: true
tags: [optimization, machine-learning, math]
summary: "Deriving the gradient descent update, its convergence rate on smooth convex functions, and why momentum helps."
ShowToc: true
---

Gradient descent is the workhorse behind almost every modern learning algorithm. This post derives the update rule, states a classic convergence result, and looks at momentum.

## The Update Rule

Given a differentiable loss $\mathcal{L}(\theta)$ with parameters $\theta \in \mathbb{R}^d$, a first-order Taylor expansion around $\theta_t$ gives

$$
\mathcal{L}(\theta_t + \Delta) \approx \mathcal{L}(\theta_t) + \nabla \mathcal{L}(\theta_t)^\top \Delta.
$$

The direction of steepest decrease is $\Delta = -\nabla \mathcal{L}(\theta_t)$, which leads to the update with learning rate $\eta > 0$:

$$
\theta_{t+1} = \theta_t - \eta \, \nabla \mathcal{L}(\theta_t).
$$

{{< figure src="gradient-descent.svg" caption="Fig. 1. Gradient descent iterates zig-zagging across the contours of an ill-conditioned quadratic toward the minimum $\theta^*$." align="center" >}}

## Convergence on Smooth Convex Functions

Suppose $\mathcal{L}$ is convex and $\beta$-smooth, i.e. $\|\nabla \mathcal{L}(x) - \nabla \mathcal{L}(y)\| \le \beta \|x - y\|$. With $\eta = 1/\beta$, gradient descent satisfies ([Nesterov, 2004](#references))

$$
\mathcal{L}(\theta_T) - \mathcal{L}(\theta^*) \le \frac{\beta \, \|\theta_0 - \theta^*\|^2}{2T}.
$$

So the suboptimality gap shrinks as $O(1/T)$.

## Momentum

Plain gradient descent oscillates along steep directions (Fig. 1). Polyak's heavy-ball method ([Polyak, 1964](#references)) keeps a running velocity $v_t$:

$$
\begin{aligned}
v_{t+1} &= \mu \, v_t - \eta \, \nabla \mathcal{L}(\theta_t), \\
\theta_{t+1} &= \theta_t + v_{t+1},
\end{aligned}
$$

where $\mu \in [0, 1)$ is the momentum coefficient. Components of the gradient that keep flipping sign cancel out, while consistent ones accumulate.

## Citation

Cited as:

> Your Name. (Nov 2026). A Gentle Look at Gradient Descent. My Blog. https://username.github.io/posts/2026-11-03-example-post/

## References

[1] Yurii Nesterov. ["Introductory Lectures on Convex Optimization: A Basic Course."](https://doi.org/10.1007/978-1-4419-8853-9) Springer, 2004.

[2] Boris T. Polyak. ["Some methods of speeding up the convergence of iteration methods."](https://doi.org/10.1016/0041-5553(64)90137-5) USSR Computational Mathematics and Mathematical Physics 4(5), 1964.

[3] Sebastian Ruder. ["An overview of gradient descent optimization algorithms."](https://arxiv.org/abs/1609.04747) arXiv:1609.04747, 2016.
