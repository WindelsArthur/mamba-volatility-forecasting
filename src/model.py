# model.py 

import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel
from transformers.models.mamba.modeling_mamba import MambaCache


class MambaTS(nn.Module):
    def __init__(
        self,
        d_in,
        d_model,
        n_layers,
        d_state,
        expand,
        conv_kernel,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        residual_in_fp32: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act

        self.in_proj = nn.Linear(d_in, d_model)

        config = MambaConfig(
            hidden_size=d_model,
            state_size=d_state,
            num_hidden_layers=n_layers,
            expand=expand,
            conv_kernel=conv_kernel,
            hidden_act=hidden_act,
            use_conv_bias=use_conv_bias,
            use_bias=use_bias,
            residual_in_fp32=residual_in_fp32,
        )
        self.backbone = MambaModel(config)
        self.out_proj = nn.Linear(d_model, 1)

    @torch.no_grad()
    def _extract_cache_states(self, cache_params: MambaCache) -> dict:
        out = {}
        if hasattr(cache_params, "ssm_states"):
            out["ssm_states"] = cache_params.ssm_states
        elif hasattr(cache_params, "ssm_state"):
            out["ssm_states"] = cache_params.ssm_state
        if hasattr(cache_params, "conv_states"):
            out["conv_states"] = cache_params.conv_states
        elif hasattr(cache_params, "conv_state"):
            out["conv_states"] = cache_params.conv_state
        return out

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_hidden_states: bool = False,
        return_ssm_states: bool = False,
        init_ssm_states: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        """
        x: (B,T,d_in)

        Returns:
          y_hat: (B,T,1)
          aux: dict (may contain 'hidden_states', 'ssm_states', 'conv_states')
        """
        B, T, _ = x.shape
        x_emb = self.in_proj(x)  # (B,T,d_model)
        aux: dict = {}

        if not return_ssm_states:
            out = self.backbone(
                inputs_embeds=x_emb,
                use_cache=False,
                output_hidden_states=return_hidden_states,
                return_dict=True,
            )
            if return_hidden_states:
                aux["hidden_states"] = out.hidden_states
            return self.out_proj(out.last_hidden_state), aux

        # --- sequential path (analysis/inference) ---
        cache = MambaCache(
            config=self.backbone.config,
            max_batch_size=B,
            device=x.device,
            dtype=x_emb.dtype,
        )

        interm = self.backbone.config.intermediate_size
        if init_ssm_states is not None:
            if isinstance(init_ssm_states, torch.Tensor):
                if not (init_ssm_states.dim() == 4 and init_ssm_states.size(0) == self.n_layers):
                    raise ValueError("init_ssm_states tensor must have shape (n_layers, B, intermediate_size, d_state).")
                init_ssm_states = list(init_ssm_states)
            if len(init_ssm_states) != self.n_layers:
                raise ValueError("init_ssm_states must be a list of length n_layers.")
            for l in range(self.n_layers):
                s = init_ssm_states[l]
                if s.shape != (B, interm, self.d_state):
                    raise ValueError(
                        f"init_ssm_states[{l}] must have shape (B, intermediate_size, d_state)=({B},{interm},{self.d_state}), got {tuple(s.shape)}"
                    )
                cache.ssm_states[l].copy_(s.to(device=cache.ssm_states[l].device, dtype=cache.ssm_states[l].dtype))

        ssm_traj, conv_traj = [], []
        hs_last = [] if return_hidden_states else None
        ys = []

        k = self.backbone.config.conv_kernel

        for t in range(T):
            cache_position = (
                torch.arange(0, k, device=x.device, dtype=torch.long) if t == 0
                else torch.tensor([t], device=x.device, dtype=torch.long)
            )

            out_t = self.backbone(
                inputs_embeds=x_emb[:, t:t + 1, :],  # (B,1,d_model)
                cache_params=cache,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=return_hidden_states,
                return_dict=True,
            )
            cache = out_t.cache_params

            ys.append(self.out_proj(out_t.last_hidden_state))  # (B,1,1)

            cdict = self._extract_cache_states(cache)
            if "ssm_states" in cdict:
                ssm_traj.append([s.detach().clone() for s in cdict["ssm_states"]])
            if "conv_states" in cdict:
                conv_traj.append([s.detach().clone() for s in cdict["conv_states"]])
            if return_hidden_states:
                hs_last.append(out_t.hidden_states)

        y_hat = torch.cat(ys, dim=1)  # (B,T,1)

        if ssm_traj:
            aux["ssm_states"] = ssm_traj
        if conv_traj:
            aux["conv_states"] = conv_traj
        if return_hidden_states:
            aux["hidden_states_stepwise"] = hs_last

        return y_hat, aux