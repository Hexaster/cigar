"""CIGAR mechanistic release model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIGARModel(nn.Module):
    """Combustion-informed generation and retention model.

    Per-puff generation at the combustion cone is

        G_j = k_G * (G1 * q_j^2 + G2 * q_j + G3 / A) * A

    with q_j the cone intake divided by the cross-sectional area A. Each puff is
    then attenuated as it passes through the rod, pre-filter and post-filter
    segments; the mainstream yield is the sum over puffs.
    """

    def __init__(self, num_types: int = 15) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, 1)
        with torch.no_grad():
            self.type_embedding.weight.fill_(1.0)

        self.raw_G1 = nn.Parameter(torch.tensor(1.0))
        self.raw_G2 = nn.Parameter(torch.tensor(1.0))
        self.raw_G3 = nn.Parameter(torch.tensor(1.0))
        self.raw_a_init = nn.Parameter(torch.tensor(0.0))
        self.raw_m_rod = nn.Parameter(torch.tensor(-1.0))
        self.raw_n_rod = nn.Parameter(torch.tensor(1.0))
        self.raw_p_rod = nn.Parameter(torch.tensor(1.0))
        self.raw_m_fil = nn.Parameter(torch.tensor(-1.0))
        self.raw_n_fil = nn.Parameter(torch.tensor(1.0))
        self.raw_p_fil = nn.Parameter(torch.tensor(1.0))

    def resolve_k_G(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        device = batch["static"].device
        batch_size = batch["static"].shape[0]
        k_g = self.type_embedding.weight.mean() * torch.ones(batch_size, device=device)

        type_idx = batch.get("type_idx")
        if type_idx is not None:
            type_idx = type_idx.to(device).long()
            known = (type_idx >= 0) & (type_idx < self.type_embedding.num_embeddings)
            if known.any():
                k_g[known] = self.type_embedding(type_idx[known]).squeeze(-1)

        provided = batch.get("provided_factor")
        if provided is not None:
            provided = provided.to(device).float()
            k_g = torch.where(torch.isfinite(provided), provided, k_g)
        return k_g

    def get_physical_params(self) -> dict[str, torch.Tensor]:
        return {
            "G1": F.softplus(self.raw_G1) * -1e-4,
            "G2": F.softplus(self.raw_G2) * 0.1,
            "G3": F.softplus(self.raw_G3),
            "a_init": torch.sigmoid(self.raw_a_init),
            "m_rod": F.softplus(self.raw_m_rod) * -1,
            "n_rod": F.softplus(self.raw_n_rod),
            "p_rod": F.softplus(self.raw_p_rod),
            "m_fil": F.softplus(self.raw_m_fil) * -1,
            "n_fil": F.softplus(self.raw_n_fil),
            "p_fil": F.softplus(self.raw_p_fil),
        }

    @staticmethod
    def calculate_k1(V: torch.Tensor, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return m * torch.log(V) + n

    @staticmethod
    def calculate_k2(V: torch.Tensor, P: torch.Tensor, C: torch.Tensor, position: str) -> torch.Tensor:
        if position == "rod":
            return (-0.07192 + 1.373e-5 * V**2 + 0.9734 * V ** (-2 / 3)) * P * C**2 / 5.76 / 56.4
        if position in {"pre", "post"}:
            return (0.0485 + 2.611e-4 * V + 1.104 * V ** (-2 / 3)) * P * C**2 / 242 / 5.76
        raise ValueError("position must be 'rod', 'pre', or 'post'")

    @staticmethod
    def calculate_k3(p: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return p / V

    @staticmethod
    def calculate_eta(
        k1: torch.Tensor,
        k2: torch.Tensor,
        k3: torch.Tensor,
        a_in: torch.Tensor,
        L: torch.Tensor,
    ) -> torch.Tensor:
        denom = k1 + k3 - k2
        param1 = (k1 - k2) / denom
        param2 = (k3 + a_in * (k1 - k2)) / denom
        eta = param1 * (1 - a_in) * torch.exp((-k1 - k3) * L) + param2 * torch.exp(-k2 * L)
        return torch.clamp(eta, min=1e-6, max=1.0)

    @staticmethod
    def calculate_a_out(
        a_in: torch.Tensor,
        k1: torch.Tensor,
        k3: torch.Tensor,
        L: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        a_out = 1 - ((1 - a_in) * torch.exp(-(k1 + k3) * L)) / eta
        return torch.clamp(a_out, min=1e-6, max=1.0)

    def calculate_retention(
        self,
        V: torch.Tensor,
        P: torch.Tensor,
        C: torch.Tensor,
        L: torch.Tensor,
        a_in: torch.Tensor,
        m: torch.Tensor,
        n: torch.Tensor,
        p: torch.Tensor,
        position: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        V_safe = torch.clamp(V, min=1.0)
        k1 = self.calculate_k1(V_safe, m, n)
        k2 = self.calculate_k2(V_safe, P, C, position)
        k1 = torch.clamp(k1, min=k2 + 1e-6)
        k3 = self.calculate_k3(p, V_safe)
        eta = self.calculate_eta(k1, k2, k3, a_in, L)
        a_out = self.calculate_a_out(a_in, k1, k3, L, eta)
        return eta, a_out, k1, k2, k3

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False):
        static = batch["static"]
        dynamic = batch["dynamic"]
        puff_count = batch["puff_count"]
        area = batch["area"].unsqueeze(1)
        batch_size = static.shape[0]
        params = self.get_physical_params()
        k_g = self.resolve_k_G(batch)

        C = static[:, 0]
        L_pre = static[:, 1]
        L_post = static[:, 2]
        P_rod = static[:, 3]
        P_fil = static[:, 4]
        total_yield = torch.zeros(batch_size, device=static.device)
        aux_rows = []

        for j in range(dynamic.shape[1]):
            L_rod = dynamic[:, j, 0]
            q_area = dynamic[:, j, 1]
            q_area_sq = dynamic[:, j, 2]
            V_rod = dynamic[:, j, 3]
            V_pre = dynamic[:, j, 4]
            V_post = dynamic[:, j, 5]

            generation = k_g * (params["G1"] * q_area_sq + params["G2"] * q_area + params["G3"] / area.squeeze()) * area.squeeze()
            a_current = params["a_init"].expand(batch_size)

            eta_rod, a_rod, k1_rod, k2_rod, k3_rod = self.calculate_retention(
                V_rod, P_rod, C, L_rod, a_current, params["m_rod"], params["n_rod"], params["p_rod"], "rod"
            )
            eta_pre, a_pre, k1_pre, k2_pre, k3_pre = self.calculate_retention(
                V_pre, P_fil, C, L_pre, a_rod, params["m_fil"], params["n_fil"], params["p_fil"], "pre"
            )
            eta_post, a_post, k1_post, k2_post, k3_post = self.calculate_retention(
                V_post, P_fil, C, L_post, a_pre, params["m_fil"], params["n_fil"], params["p_fil"], "post"
            )

            eta_total = eta_rod * eta_pre * eta_post
            mask = (j < puff_count).float()
            yield_j = generation * eta_total * mask
            total_yield = total_yield + yield_j

            if return_aux:
                aux_rows.append(
                    {
                        "puff": j + 1,
                        "generation": generation.detach(),
                        "yield": yield_j.detach(),
                        "cumulative": total_yield.detach().clone(),
                        "rod": (V_rod.detach(), k1_rod.detach(), k2_rod.detach(), k3_rod.detach(), a_rod.detach(), eta_rod.detach()),
                        "pre": (V_pre.detach(), k1_pre.detach(), k2_pre.detach(), k3_pre.detach(), a_pre.detach(), eta_pre.detach()),
                        "post": (V_post.detach(), k1_post.detach(), k2_post.detach(), k3_post.detach(), a_post.detach(), eta_post.detach()),
                    }
                )

        if return_aux:
            return total_yield, aux_rows
        return total_yield


class FixedParamCIGARModel(CIGARModel):
    """CIGAR model with effective physical parameters fixed for reporting."""

    PARAM_NAMES = [
        "G1",
        "G2",
        "G3",
        "a_init",
        "m_rod",
        "n_rod",
        "p_rod",
        "m_fil",
        "n_fil",
        "p_fil",
    ]

    def __init__(self, params: dict[str, float], num_types: int = 15) -> None:
        super().__init__(num_types=num_types)
        missing = [name for name in self.PARAM_NAMES if name not in params]
        if missing:
            raise ValueError(f"Missing fixed parameters: {missing}")
        self._fixed_params = {}
        for name in self.PARAM_NAMES:
            tensor = torch.tensor(float(params[name]), dtype=torch.float32)
            self.register_buffer(f"fixed_{name}", tensor)
            self._fixed_params[name] = tensor
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def get_physical_params(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, f"fixed_{name}") for name in self.PARAM_NAMES}
