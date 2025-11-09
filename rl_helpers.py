import jax.numpy as jnp
import jax
import optax
import rlax
import distrax


# all_values = jnp.concatenate((values.squeeze(), final_value[None, :]), axis=0)
# discounts = (1 - dones.squeeze()) * self.config.gamma
# batch_advantage_estimation = vmap(rlax.truncated_generalized_advantage_estimation,
#                                   in_axes=(1, 1, None, 1))
#
# advantages = batch_advantage_estimation(rewards, discounts, self.config.gae_lambda, all_values)

