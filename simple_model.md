# Simple model

## Setup

$$s = AK_s^\alpha p^{-\theta}$$
$$p = BK_p^\beta$$
$$r_i = \frac{p_i}{\sum_j p_j}$$

$$\pi_i = \prod_j \left( \frac{s_j}{1+s_j} \right) (r_i - d_i) - d_i - R(K_{s,i} + K_{p_i})$$

## Deriving FOCs

We have
$$\frac{\partial p}{K_p} = B\beta K_p^{\beta-1}$$
$$\frac{\partial s}{K_s} = A\alpha K_s^{\alpha-1} p_i^{-\theta_i}$$
$$\frac{\partial s}{K_p} = -AB^{-\theta} \beta \theta K_p^{-1-\beta \theta}$$

and

$$\frac{\partial}{\partial s_i} \prod_j \left( \frac{s_j}{1+s_j} \right) = \frac{1}{s_i(1+s_i)} \prod_j \left( \frac{s_j}{1+s_j} \right)$$
$$\frac{\partial}{\partial p_i} r_i = \frac{\sum_{j \neq i} p_j}{\left(\sum_{j} p_j\right)^2}$$

so

$$\frac{\partial \pi_i}{\partial K_{s,i}} = \frac{1}{s_i(1+s_i)} \prod_j \left( \frac{s_j}{1+s_j} \right) A_i\alpha_i K_{s,i}^{\alpha_i-1} p_i^{-\theta_i} (r_i - d_i) - R$$
$$\frac{\partial \pi_i}{\partial K_{p,i}} = - \frac{1}{s_i(1+s_i)} \prod_j \left( \frac{s_j}{1+s_j} \right) A_i B_i^{-\theta_i} \beta_i \theta_i K_{p,i}^{-1-\beta_i \theta_i} (r_i - d_i) + \prod_j \left(\frac{s_j}{1+s_j}\right) \frac{\sum_{j \neq i} p_j}{\left(\sum_{j} p_j\right)^2} B_i \beta_i K_{p,i}^{\beta_i-1} - R,$$
which must both be equal to zero in FOC. This is a mess (especially since I've mixed using s, p, and K), but it gets a lot simpler if we let $\theta_i = 0$ (meaning safety does not become more expensive at higher levels of performance):

$$\left. \frac{\partial \pi_i}{\partial K_{s,i}} \right|_{\theta_i = 0} = \frac{1}{s_i(1+s_i)} \prod_j \left( \frac{s_j}{1+s_j} \right) A_i\alpha_i K_{s,i}^{\alpha_i-1} (r_i - d_i) - R$$
$$\left. \frac{\partial \pi_i}{\partial K_{p,i}} \right|_{\theta_i = 0} = \prod_j \left(\frac{s_j}{1+s_j}\right) \frac{\sum_{j \neq i} p_j}{\left(\sum_{j} p_j\right)^2} B_i \beta_i K_{p,i}^{\beta_i-1} - R$$

This gives, for example,
$$\frac{\sum_{j \neq i} p_j}{\left(\sum_{j} p_j\right)^2} B_i \beta_i K_{p,i}^{\beta_i-1} = \frac{1}{s_i(1+s_i)} A_i\alpha_i K_{s,i}^{\alpha_i-1} (r_i - d_i),$$
which we can start to get some insights from.

Above formula can be rearranged to
$$\frac{1}{A_i \alpha_i} K_{i,s}^{1-\alpha_i} s_i (1+s_i) = \frac{1}{B_i \beta_i} K_{i,p}^{1-\beta_i} \frac{p_i (r_i - d_i)}{r_i(1-r_i)}.$$
Left side is nonnegative and increasing, concave in $K_{i,s}$. Right side is a bit harder. Nonnegativity of left side means we need $r_i > d_i$ for an interior solution.

## A different approach

It's likely I've made a mistake above... let's try attacking from a different angle. Think about problem only in terms of $s_i$ and $p_i$, and assume that $\theta_i = 0$. Objective can be written as

$$\pi_i = \prod_j\left(\frac{s_j}{1+s_j}\right)(r_i - d_i) - d_i - R\left( \left(\frac{s_i}{A_i}\right)^{1/\alpha_i} + \left(\frac{p_i}{B_i}\right)^{1/\beta_i}\right)$$
subject to $s_i, p_i \geq 0$. For more concise notation, we'll define
$$Q = \prod_j\left(\frac{s_j}{1+s_j}\right)$$
and note the following derivatives:
$$\frac{\partial Q}{\partial s_i} = \frac{Q}{s_i(1+s_i)}$$
$$\frac{\partial r_i}{\partial p_i} = \frac{r_i(1-r_i)}{p_i}$$

So FOCs with respect to $s_i$ and $p_i$ are:
$$\frac{Q}{s_i(1+s_i)} (r_i - d_i) - \frac{R}{A_i \alpha_i} \left(\frac{s_i}{A_i}\right)^{(1-\alpha_i)/\alpha_i} = 0$$
$$Q \frac{r_i(1-r_i)}{p_i} - \frac{R}{B_i \beta_i} \left(\frac{p_i}{B_i}\right)^{(1-\beta_i)/\beta_i} = 0$$
Equivalently,
$$R = Q A_i^{1/\alpha_i} \cdot \frac{r_i - d_i}{s_i^{1/\alpha_i}(1 + s_i)} = QB_i^{1/\beta_i} \cdot \frac{r_i(1-r_i)}{p_i^{1/\beta_i}}.$$

This is much cleaner than what we had before!

### Case of zero disaster cost

With $d_i = 0$, we can get
$$A_i^{1/\alpha_i} \cdot \frac{1}{s_i^{1/\alpha_i} (1+s_i)} = B_i^{1/\beta_i} \cdot \frac{1-r_i}{p_i^{1/\beta_i}}.$$
LHS is decreasing in $s_i$. RHS is decreasing in $p_i$, though it also depends on other agents' choice of $p$ via $r_i$.

Thus if $s_i$ increases, then $p_i$ increases or $p_j$ decreases for some $j \neq i$. What about the case where everyone is identical? The condition becomes
$$A^{1/\alpha} \cdot \frac{1}{s^{1/\alpha}(1+s)} = B^{1/\beta} \cdot \frac{n-1}{np^{1/\beta}}.$$
Here the relationship is clear -- $s$ and $p$ always move in the same direction.

### Jumping back to general symmetric case

The general condition for the symmetric case is
$$R = Q A^{1/\alpha} \cdot \frac{r - d}{s^{1/\alpha}(1+s)} = QB^{1/\beta} \cdot \frac{r(1-r)}{p^{1/\beta}}$$
or, equivalently,
$$R/Q = A^{1/\alpha} \left( \frac{1}{n} - d \right) \frac{1}{s^{1/\alpha}(1+s)} = B^{1/\beta} \left(\frac{n-1}{n^2}\right) \frac{1}{p^{1/\beta}}.$$
It's worth noting that we need $d < 1/n$. Safety and performance always move together (in opposite direction of $R$) in symmetric case. It's worth noting that the Q in the denominator provides a dampening effect, since $Q$ is actually an increasing function of $s$: if $R$ increases, then $s$ must increase, which then increases $Q$ (at least a little bit; the effect is less pronounced when $s$ is large so $Q \approx 1$), which means that $s$ doesn't have to change as much. The formula for performance is maybe worthwhile to look at:
$$p = B \left[\left(\frac{n-1}{n^2}\right)\frac{Q}{R} \right]^\beta$$

Formula for $s$ is more complicated...

## Two-player heterogeneous game

To-do