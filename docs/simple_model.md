# Simple model

## Setup

$$s = AK_s^\alpha p^{-\theta}$$
$$p = BK_p^\beta$$
$$r_i = \frac{p_i}{\sum_j p_j}$$

$$\pi_i = \prod_j \left( \frac{s_j}{1+s_j} \right) (r_i + d_i) - d_i - R(K_{s,i} + K_{p_i})$$

## Deriving FOCs

Think about problem only in terms of $s_i$ and $p_i$, and assume for now that $\theta_i = 0$. Objective can be written as

$$\pi_i = \prod_j\left(\frac{s_j}{1+s_j}\right)(r_i + d_i) - d_i - R\left( \left(\frac{s_i}{A_i}\right)^{1/\alpha_i} + \left(\frac{p_i}{B_i}\right)^{1/\beta_i}\right)$$
subject to $s_i, p_i \geq 0$. For more concise notation, we'll define
$$Q = \prod_j\left(\frac{s_j}{1+s_j}\right)$$
and note the following derivatives:
$$\frac{\partial Q}{\partial s_i} = \frac{Q}{s_i(1+s_i)}$$
$$\frac{\partial r_i}{\partial p_i} = \frac{r_i(1-r_i)}{p_i}$$

So FOCs with respect to $s_i$ and $p_i$ are:
$$\frac{Q}{s_i(1+s_i)} (r_i + d_i) - \frac{R}{A_i \alpha_i} \left(\frac{s_i}{A_i}\right)^{(1-\alpha_i)/\alpha_i} = 0$$
$$Q \frac{r_i(1-r_i)}{p_i} - \frac{R}{B_i \beta_i} \left(\frac{p_i}{B_i}\right)^{(1-\beta_i)/\beta_i} = 0$$
Equivalently,
$$R = Q A_i^{1/\alpha_i} \cdot \frac{r_i + d_i}{s_i^{1/\alpha_i}(1 + s_i)} = QB_i^{1/\beta_i} \cdot \frac{r_i(1-r_i)}{p_i^{1/\beta_i}}.$$

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
$$R = Q A^{1/\alpha} \cdot \frac{r + d}{s^{1/\alpha}(1+s)} = QB^{1/\beta} \cdot \frac{r(1-r)}{p^{1/\beta}}$$
or, equivalently,
$$R/Q = A^{1/\alpha} \left( \frac{1}{n} + d \right) \frac{1}{s^{1/\alpha}(1+s)} = B^{1/\beta} \left(\frac{n-1}{n^2}\right) \frac{1}{p^{1/\beta}}.$$
Safety and performance always move together (in opposite direction of $R$) in symmetric case. It's worth noting that the Q in the denominator provides a dampening effect, since $Q$ is actually an increasing function of $s$: if $R$ increases, then $s$ must increase, which then increases $Q$ (at least a little bit; the effect is less pronounced when $s$ is large so $Q \approx 1$), which means that $s$ doesn't have to change as much. The formula for performance is maybe worthwhile to look at:
$$p = B \left[\left(\frac{n-1}{n^2}\right)\frac{Q}{R} \right]^\beta$$

Formula for $s$ is more complicated... I actually need to think more about "feedback" effect of Q on l.h.s. ...

## Two-player heterogeneous game

Heterogeneous game is kind of complicated, so let's just start with two players. Helpful property here is that
$$r_i(1-r_i) = \frac{p_1 p_2}{p_1 + p_2}$$
for $i = 1, 2$. Second FOC thus gives
$$\left( \frac{p_1}{B_1} \right)^{1/\beta_1} = \left( \frac{p_2}{B_2} \right)^{1/\beta_2},$$
so we have a simple relationship between the performance of the two players. We'll assume for now that $\beta_1 = \beta_2 =: \beta$, so we get
$$\frac{p_1}{B_1} = \frac{p_2}{B_2}.$$
This also means that
$$r_i = \frac{p_i}{p_i + p_{-i}} = \frac{p_i}{p_i + \frac{B_i}{B_{-i}} p_i} = \frac{B_{-i}}{B_i + B_{-i}}$$
so
$$r_i(1-r_i) = \frac{B_1 B_2}{(B_1 + B_2)^2}$$
and therefore
$$\frac{Q}{R} = \frac{(B_1 + B_2)^2}{B_1 B_2} \left( \frac{p_i}{B_i} \right)^{1/\beta}$$
which then provides a nice formula for $p_i$:
$$p_i = \left(\frac{Q}{R} \frac{B_{-i}}{(B_i + B_{-i})^2} \right)^\beta \cdot B_i^{\beta + 1}$$
Performance moves in the same direction as $Q$ (proba of a safe outcome). I think this should scale up to games with more than 2 players. I don't think this necessarily applies if $\beta$ is not the same for everyone.

To figure out how $p$ reacts to changes in $R$, need to first think about safety side of things. 

Safety condition is
$$\frac{R}{Q} = \frac{r_i + d_i}{1 + s_i} \left(\frac{A_i}{s_i}\right)^{1/\alpha_i}$$
as before. This is still complicated by fact that $s_i$ also appears in $Q$, but we can at least relate safety of the two players:
$$\frac{r_1 + d_1}{1 + s_1} \left(\frac{A_1}{s_1}\right)^{1/\alpha_1} = \frac{r_2 + d_2}{1 + s_2} \left(\frac{A_2}{s_2}\right)^{1/\alpha_2}$$
Rearranging and using formulas to relate the $p$'s:
$$\left(\frac{B_1}{B_1 + B_2} + d_1\right)^{-1} \left(\frac{s_1}{A_1}\right)^{1/\alpha_1} (1+s_1) = \left(\frac{B_2}{B_1 + B_2} + d_2\right)^{-1} \left(\frac{s_2}{A_2}\right)^{1/\alpha_2} (1+s_2)$$
It's a bit messy but it does tell us a few helpful things. $s_1$ will be higher relative to $s_2$ if:

* $A_1$ is high relative to $A_2$ (more efficient at producing safety means higher safety)
* $B_1$ is high relative to $B_2$ (more efficient at producing performance means higher safety)
* $d_1$ is high relative to $d_2$ (more to lose from disaster means higher safety)

It's also worth noting that the two players will always change their levels of safety in the same direction, as long as their $\alpha$ parameters are roughly the same.

### How do s and p respond to change in R?

Equivalence between first and second FOCS can be stated as

$$\frac{Q}{R} = \left(\frac{B_i}{B_i + B_{-i}} + d_i \right)^{-1} (1 + s_i) \left( \frac{s_i}{A_i} \right)^{1/\alpha_i} = \frac{(B_i + B_{-i})^2}{B_i B_{-i}} \left( \frac{p_i}{B_i} \right)^{1/\beta}$$

This means that $p_i$ and $s_i$ must move in the same direction. 

What have we figured out so far?

* The $p$'s always move in the same direction.
* The $p$'s move in the same direction as $Q/R$.
* The $s$'s always move in the same direction.
* The $p$'s move in the same direction as the $s$'s (meaning also that the $p$'s move in the same direction as $Q$).

Suppose that $R$ increases. To "balance this out," we need $Q$ to increase, or $p$ to decrease. Those two things are mutually exclusive, so either both $Q$ and $p$ increase, or both $Q$ and $p$ decrease. It's not really clear which one of these should happen, but I think I should be able to work out conditions to determine when each case occurs.
