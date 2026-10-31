---
title: 'Checking the effect of your Meridian MMM priors without refitting the model multiple times'
date: 2026-10-31
excerpt: 'Fit your Meridian MMM once, then reweight the saved posterior draws to see the effect of a different prior, instead of refitting.'
tags:
  - marketing mix modeling
  - bayesian
  - meridian
  - importance sampling
---

**TL;DR**
- In practice, we usually fit multiple Marketing Mix Models (MMM), possibly under different assumptions or prior choices. This short note is about avoiding such refits, whenever possible.
- Fit the model once, then reweight the saved draws to see the posterior under a different prior.
- Two checks say whether to trust the result: the effective sample size (ESS) and Pareto k̂. **Use it if k̂ < 0.5, refit if k̂ ≥ 0.7.**
- Tested on 300 alternative priors for a national model and 60 for a 40-geo regional model, against full refits.

**Detailed examples with code for national and regional models can be found in the [GitHub repository](https://github.com/hansamos/prior_sensitivity_meridian).**

## The problem
Fitting a good MMM on a real dataset is not easy. It is a painful, iterative process and can be time-consuming, particularly when the model is estimated with sampling-based methods (MCMC).
One aspect of model fitting is robustness to prior choices. Checking it means refitting once per alternative prior, so most people never check.
However, there is a well-known, decades-old trick called importance sampling that can bypass these multiple refits. Mathematical details are given at the end, together with a curiosity.

## The Solution
You fit the model once. Your reference fit already contains draws from the posterior. If the new prior is not too different, as is usually the case in a sensitivity analysis, those draws can be reweighted instead of refitted (math at the end).

## Why it matters for budget decisions
- **Practical:** test many priors (a lower or higher ROI, a faster or slower carryover, earlier or later diminishing returns, a prior from a recent lift test, a more or less confident prior for one channel) before you move budget, not after.
- **Fast and cheaper:** reweighting needs no new sampling, only a calculation on the saved draws. Each refit is a full MCMC run. Avoiding most of them can bring the time to value down from weeks or months to a few days, and also reduce the modeling cost.
- **Robust:** if the budget recommendation holds across plausible priors, you can trust it. If it flips, the data does not settle the question.
- **Safe:** k̂ and ESS tell you which answers to trust and which priors need a real refit.

## How easy is it?
**1. Fit the model as usual**, once, with the reference prior:

```python
import tensorflow_probability.substrates.jax as tfp
from meridian.model import model, prior_distribution, spec

tfd = tfp.distributions

mmm = model.Meridian(
    input_data=input_data,  # your Meridian InputData
    model_spec=spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(roi_m=tfd.LogNormal(0.2, 0.9)),
        media_prior_type="roi",
    ),
)
mmm.sample_posterior(n_chains=4, n_adapt=1000, n_burnin=500, n_keep=1000, seed=1)
```

**2. Change the prior, but do not refit: reweight the posterior.** Here the new prior is a higher and tighter ROI:

```python
import arviz as az
import numpy as np
from scipy import stats

roi = mmm.inference_data.posterior["roi_m"].values   # (chain, draw, channel)
roi = roi.reshape(-1, roi.shape[-1])                  # (draws, channel)

# Weight of each draw: new prior / reference prior
log_w = (stats.lognorm.logpdf(roi, s=0.6, scale=np.exp(0.5))    # new prior
         - stats.lognorm.logpdf(roi, s=0.9, scale=np.exp(0.2))  # reference prior
         ).sum(axis=1)

# Smooth the weights and check them
log_w, k_hat = az.psislw(log_w)
w = np.exp(log_w)
w /= w.sum()
ess_pct = 100 / np.sum(w**2) / len(w)
print(f"k-hat = {k_hat:.2f}, ESS = {ess_pct:.0f}% of draws")   # use if k-hat < 0.5, refit if >= 0.7

# Posterior mean of ROI per channel under the new prior
new_roi_mean = (w[:, None] * roi).sum(axis=0)
```

To test another parameter, change the two log densities (for example `stats.halfnorm`). Step 1 is the only expensive one, and you do it once.

## The test

The reweighting method applies to any Bayesian MMM, but here we demonstrate it with Google Meridian. We use the model in the [Introduction to Meridian Demo](https://developers.google.com/meridian/notebook/meridian-getting-started#configure-model).



We start with a national model and test three parameters: `roi_m`, `alpha_m` and `ec_m`.  For each, 5 tiers of prior change, from very close to very far, with 20 new priors per tier. We refit all 300 to get the ground truth, and we compare it to the reweighted posterior using the KL divergence.

## From very close to very far priors

![Prior tiers](/images/posts/prior-sensitivity-meridian/national_fig1_prior_tiers.png)

*Each column is a tier: the new priors (blue) move further away from the reference prior (black) as the KL grows.*

## Result 1: it works until the prior moves far
The reweighted posterior matches a full refit up to mid-sized changes, then breaks down.

![Error by tier](/images/posts/prior-sensitivity-meridian/national_fig2_error_by_tier.png)

*The error stays small up to mid-sized changes and grows for far and very far priors, most for `ec_m`.*

- **Very close to mid:** KL to the refit is 0.004–0.015, the same level as the difference between two independent refits (about 0.01).
- **Far and very far:** KL 0.06–1.09, worst for `ec_m`.
- **Some far `roi_m` priors are extreme:** they imply that media explains more than 100% of revenue. These were chosen as stress tests to see whether the method breaks, not as realistic alternatives.

## Result 2: ESS and k̂ tell you when it works
![ESS vs error](/images/posts/prior-sensitivity-meridian/national_fig4_ess_vs_error.png)

*Most priors that keep less than 10% of the draws (shaded) are far from the refit, but not all of them.*

![k̂ vs error](/images/posts/prior-sensitivity-meridian/national_fig3_khat_vs_error.png)

*The error rises with k̂, and priors with k̂ ≥ 0.7 (shaded) are the ones to refit.*

|  | k̂ < 0.5 | k̂ 0.5–0.7 | k̂ ≥ 0.7 |
|---|---|---|---|
| **ESS ≥ 10%** | **use it**<br>121 priors, KL 0.005 | **check**<br>4 priors, KL 0.015 | **refit**<br>1 prior, KL 0.042 |
| **ESS < 10%** | **use it**<br>23 priors, KL 0.009 | **check**<br>30 priors, KL 0.028 | **refit**<br>121 priors, KL 0.234 |

Each cell: number of priors and median KL to the refit. k̂ decides. A low ESS with k̂ < 0.5 is still fine (23 priors, KL ≤ 0.013), and ESS ≥ 10% does not rescue k̂ ≥ 0.5.

## Time saved
For our 300 test priors:

| | Compute time |
|---|---|
| 300 full refits | 4.0 hours (median 46 s each) |
| 300 reweightings | 10 seconds (median 0.02 s each) |
| Reweight all, refit only k̂ ≥ 0.5 (156 priors) | 2.1 hours |
| Reweight all, refit only k̂ ≥ 0.7 (122 priors) | 1.6 hours |

**Reweighting is about 1,400 times faster than refitting in total compute time (4.0 hours vs 10 seconds).** Two of our five tiers (far and very far) are not realistic prior choices. We chose them as stress tests to break the method, so a typical sensitivity analysis with closer priors should need fewer refits.

## Does it work on a regional (geo) model?
Yes, with the same pattern. We repeated the test on the full 40-geo version of the dataset (156 weeks, 5 channels), with two parameters: `roi_m` and `eta_m`. `eta_m` is the scale of the media effect across geos and is one of the most important parameters in regional models. We used 3 tiers of prior change (close, mid, far), 10 new priors per tier and parameter, and refitted all 60. Again, we compare the full refit with the reweighted posterior using the KL divergence.

![Regional error by tier](/images/posts/prior-sensitivity-meridian/reg_fig2_error_by_tier.png)

*As in the national test, the error grows with the size of the prior change, and faster for `eta_m` than for `roi_m`.*

|  | k̂ < 0.5 | k̂ 0.5–0.7 | k̂ ≥ 0.7 |
|---|---|---|---|
| **ESS ≥ 10%** | **use it**<br>20 priors, KL 0.006 | **check**<br>4 priors, KL 0.015 | **refit**<br>1 prior, KL 0.040 |
| **ESS < 10%** | **use it**<br>5 priors, KL 0.004 | **check**<br>2 priors, KL 0.027 | **refit**<br>28 priors, KL 0.060 |

- **k̂ < 0.5:** all 25 priors have KL ≤ 0.026, close to the refit-to-refit noise (about 0.01).
- **`eta_m` is more fragile than `roi_m`:** all 10 far `eta_m` priors have k̂ ≥ 0.7, against 7 of 10 for `roi_m`. Check the diagnostics before trusting a reweighted `eta_m`.

## Future work

This was a very limited simulation exercise with only one dataset, but reweighting via importance sampling has been time-tested and is a very reliable tool for sensitivity analysis. Maybe the Google Meridian team should add it as a built-in option.

---

## Math: the core idea

**Importance sampling.** The reference fit gives draws $$\theta_s \sim p(\theta \mid y, p_0)$$ under the reference prior $$p_0$$. For a new prior $$p_1$$:

$$p(\theta \mid y, p_1) \;\propto\; p(\theta \mid y, p_0)\,\frac{p_1(\theta)}{p_0(\theta)}$$

The likelihood cancels. So each draw gets a weight

$$w_s = \frac{p_1(\theta_s)}{p_0(\theta_s)}, \qquad \mathbb{E}_{p_1}[f] \approx \frac{\sum_s w_s f(\theta_s)}{\sum_s w_s}.$$

Only the changed prior matters, evaluated on the sampled parameter.

**When it fails.** If $$p_1$$ puts mass where the reference draws are rare, a few weights dominate. The effective sample size, $$\mathrm{ESS} = (\sum w)^2 / \sum w^2$$, drops. We report it as a share of the total number of draws.

**PSIS** (Vehtari et al., 2024). Pareto-smoothed importance sampling fits a generalized Pareto distribution to the largest weights and replaces them with smoothed values. Its shape parameter $$\hat k$$ measures how heavy the tail of the weights is.

| k̂ | Meaning |
|---|---|
| < 0.5 | finite variance, reliable |
| 0.5 – 0.7 | usable, slower convergence |
| ≥ 0.7 | unreliable |

**Distance between priors.** We measure how far a new prior is from a reference with the divergence $$\mathrm{KL}(p_1 \,\|\, p_0) = \mathbf{E}_{p_1}(\log \frac{p_1}{p_0})$$ and group priors into tiers by its quantiles.

## Curiosity

The PSIS method is available in the [ArviZ](https://www.arviz.org) project and integrates easily with Google Meridian.
ArviZ was supported by Google Summer of Code in [2022](https://summerofcode.withgoogle.com/archive/2022/projects/ap4EPHJ0) and [2024](https://summerofcode.withgoogle.com/archive/2024/projects/X9jNcYK7). Now that work is also useful for Google's own product, Meridian.
So let's keep investing in basic science: you never know when it will be useful.

## Reference
Vehtari, A., Simpson, D., Gelman, A., Yao, Y., and Gabry, J. (2024). Pareto smoothed importance sampling. *Journal of Machine Learning Research*, 25(72):1–58.
