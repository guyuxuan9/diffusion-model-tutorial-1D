# diffusion-model-tutorial-1D
- Tutorial link: https://e-dorigatti.github.io/math/deep%20learning/2023/06/25/diffusion.html
- Evidence Lower Bound (ELBO), KL divergence proof: https://jaketae.github.io/study/elbo/
- Denoising Diffusion Probabilistic Models: https://arxiv.org/pdf/2006.11239
  
# Forward Process

$$q(x_{1:T} | x_0) = \prod_{t=0}^T q(x_t | x_{t-1}), $$ 
$$\text{where} \quad q(x_t | x_{t-1}) := \mathcal{N}\left(x_t | \sqrt(1-\beta_t) x_{t-1}; \beta_t I\right)$$

$q(x_{1:T})$ means the joint distribution of $x_1, x_2, x_3, ... x_T$. It is the same as $q(x_1, x_2, x_3, ... x_T)$. $\beta_t$ could be chosen manually or learnt by the network. Assume we choose values such as $0 < \beta_t < 1$, we can prove that the final distribution after many iterations $q(x_T) \sim \mathcal{N}(0, I)$.  Note that we iteratively add noise to the original distribution. From step $t=1$ onwards, the noise comes from two parts:
- A shrinking of the previous step's value $x_{t-1}$ by $\sqrt(1-\beta_t)$, which means the variance from previous steps gets scaled by $\sqrt(1-\beta_t)$.
- Addition of new independent noise with variance $\beta_t I$.



The following is a sketch of proof.
- For means, $\lim_{N\to\infty} \mu_N = \lim_{N\to\infty} \left(\sqrt{1-\beta_t}\right)^N = 0$
- For variance, consider:
    - $\sigma_0^2 = 0$
    - $\sigma_1^2 = \beta_1$
    - $\sigma_2^2 = (1-\beta_2)\beta_1I + \beta_2I = \left[ 1 - (1-\beta_1)(1-\beta_2) \right]I$.
    - $\sigma_3^2 = \left[ 1 - (1-\beta_1)(1-\beta_2) \right] (1-\beta_3)I + \beta_3I = \left[ 1 - (1-\beta_1)(1-\beta_2)(1-\beta_3) \right]I$
    - Therefore, $\sigma_N^2 = (1 - \bar{\alpha_N}) I$, where $\bar{\alpha_N} = \prod_{s=1}^t (1 - \beta_s)$. Now for very large $N$,  $\lim_{N\to\infty}\sigma_N^2 = 0$
# Reverse Process
The idea is to sample a $x_T$ from a zero-mean unit-variance Gassian distribution and use the learnt Neural Network to recover the $x_0$. We want to maximize the likelihood of observing the original input $x_0$, $\textit{i.e.}, \text{max} E\left[ p_\theta(x_0) \right]$. This is equivalent to minimizing the negative log likelihood of $x_0$ under the predicted distribution parameterized by $\theta$, $\textit{i.e.}, \text{min} E\left[- log p_\theta(x_0) \right]$

$$\begin{aligned}
\log\left( p(x) \right) &= \log\left( \int p(x, h) \, dh \right) \\
                               &= \log\left( \int q(h|x) \frac{p(x, h}{q(h|x)} dh \right) \\
                               &= \log\left( E_q \left[ \frac{p(x, h)}{q(h|x)} \right] \right) \\
                               &\geq E_q\left( \log \left[ \frac{p(x, h)}{q(h|x)} \right] \right)
\end{aligned}$$

The above equation shows how to compute the Evidence Lower Bound (ELBO) or Variational Lower Bound for $\log\left( p_\theta(x) \right)$. The last inequality is called **Jensen's inequality** which states that for convex function $f$, $E\left[ f(x) \right] \geq f\left( E(x) \right)$. $\log$ is a concave function, so we reverse the sign. Let's apply the result to the diffusion model.

$$\begin{aligned}
-\log\left( p_\theta(x) \right) &\geq E_q\left( \log \left[ \frac{p_\theta(x_0, x_{1:T})}{q(x_{1:T}|x_0)} \right] \right) \\
                                &= E_q \left[ -\log p_\theta(x_{0:T}) + \log q(x_{1:T} | x_0 ) \right] \\
                                &= E_q \left[ -\log \left(p_\theta(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t) \right) + \log q(x_{1:T}|x_0) \right] \\
                                &= E_q \left[ -\log \left(p_\theta(x_T)\right) - \sum_{t=1}^T  \log p_\theta(x_{t-1}|x_t) + \log q(x_{1:T}|x_0) \right] \\
                                &= E_q \left[ -\log \left(p_\theta(x_T)\right) - \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t | x_{t-1})} \right]
\end{aligned}$$
