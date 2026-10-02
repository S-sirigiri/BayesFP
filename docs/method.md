# Method and implementation

[Project overview](../README.md) · [API reference](reference.md)

## Posterior target

The [BayesFP paper, Sections 3–4](https://arxiv.org/html/2606.21014v1) defines a cost-tilted policy distribution:

$$
p(x \mid O=1) \propto p_{\mathrm{data}}(x)\exp(\beta\mathcal{J}(x)),\qquad \beta<0.
$$

Here the pretrained policy provides the prior, and the cost combines task objectives and constraint penalties. Lower-cost trajectories receive greater likelihood. The method couples guided dynamics with Feynman–Kac log-weight updates. For flow matching, the paper converts the deterministic flow into a marginal-preserving stochastic process and obtains its score from the learned velocity field. See the paper for assumptions and guarantees; finite-particle implementations require separate evaluation.

## Toy implementation

The following describes the checked-in [triangle sampler](../src/ToyExamples/triangle_obstacles_2d/samplers.py) and [square sampler](../src/ToyExamples/square_obstacle_2d/samplers.py), which use discrete DDPMs.

1. `data.py` optimizes demonstration trajectories around training obstacles, caches them, and resamples their lengths.
2. `train.py` trains an unconditional temporal U-Net to predict Gaussian noise, maintaining an exponential moving average of its parameters.
3. At inference, `cost.py` evaluates penalties for the selected new obstacles. These penalties do not retrain the denoiser.
4. `infer.py` compares ancestral DDPM sampling, linear-combination guidance, and particle-based FKC sampling.

Trajectories have shape `(batch, waypoints, 2)`. The triangle sampler pins the start after every reverse step; the square sampler pins both endpoints.

At zero clearance, the cost sums squared positive penetration depths over regions, averaging over waypoints. Cost gradients are computed on detached predicted clean trajectories, rather than differentiated through the denoiser. The samplers normalize each trajectory's gradient before adding guidance to the noise prediction:

```text
eps_guided = eps + sqrt(1 - alpha_bar[t]) * guidance_strength * normalized_grad_J
```

The configuration uses a **positive** `guidance_weight` for cost reduction; this is a strength convention, not the paper's negative signed beta.

## Weights and resampling

`sample_fkc` maintains `num_particles` candidate trajectories. It accumulates log-weight increments using either `energy` (cost differences with the configured schedule) or `girsanov` (the code's score/drift-based correction plus annealing term).

Weights are normalized with softmax. Effective sample size is `1 / sum(w**2)`. When ESS falls below `resample_ess_frac * num_particles` inside `active_window`, systematic resampling copies selected ancestors and resets log-weights to zero. The active window controls **resampling**, not whether guidance is applied.

With `return_best: true`, the returned population is sorted by final constraint cost. The inference driver runs one population per requested output and takes its first particle. Thus the default figures show lowest-cost selections from populations, rather than independent draws from a weighted posterior. Turning off `return_best` leaves the population unsorted; the driver still takes its first particle.

These practical choices—gradient normalization, detached clean predictions, discrete updates, and final cost selection—matter when relating toy results to the continuous-time formulation. The toy API returns diagnostics but does not expose a ready-made posterior-expectation estimator.
