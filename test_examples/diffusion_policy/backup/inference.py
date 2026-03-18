import torch


class Inference:
    """
    Shared conditional_sample implementation for diffusion policies.

    Expects subclasses to define:
      - self.model
      - self.noise_scheduler
      - self.num_inference_steps
    """

    # sentinel so we can distinguish:
    #  - transformer policies that *pass* cond=None
    #  - unet policies that don't use cond at all
    _COND_UNSET = object()

    # ========= inference ============
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            cond=_COND_UNSET,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            if cond is self._COND_UNSET:
                model_output = model(trajectory, t,
                    local_cond=local_cond, global_cond=global_cond)
            else:
                # TransformerForDiffusion forward has `cond` as the 3rd arg (can be None) :contentReference[oaicite:2]{index=2}
                model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory