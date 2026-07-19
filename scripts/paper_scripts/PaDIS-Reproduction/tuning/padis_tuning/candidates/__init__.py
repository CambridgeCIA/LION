"""Candidate catalogues for PaDIS reconstruction hyperparameter tuning."""

from padis_tuning.candidates.core import (
    Candidate,
    RunRecord,
    broad_candidates,
    current_default_candidates,
    flag_value_args,
    focused_candidates,
    lion_physics_candidate,
    pilot_candidates,
    safe_name,
    sampler_candidate,
    unique_candidates,
    zeta_candidates,
)
from padis_tuning.candidates.extended import (
    candidate_set,
    consensus_24h_candidates,
    consensus_24h_no_defaults_candidates,
    lion_physics_full_candidates,
    lion_physics_pc_public_gap_candidates,
    reproduction_candidates,
    sampler_full_candidates,
)
from padis_tuning.candidates.paper import (
    padis_dps_lion_full_candidates,
    paper_full_candidates,
    public_paper_sampler_candidates,
    public_repo_full_candidates,
)

__all__ = [
    "Candidate",
    "RunRecord",
    "broad_candidates",
    "candidate_set",
    "consensus_24h_candidates",
    "consensus_24h_no_defaults_candidates",
    "current_default_candidates",
    "flag_value_args",
    "focused_candidates",
    "lion_physics_candidate",
    "lion_physics_full_candidates",
    "lion_physics_pc_public_gap_candidates",
    "padis_dps_lion_full_candidates",
    "paper_full_candidates",
    "pilot_candidates",
    "public_paper_sampler_candidates",
    "public_repo_full_candidates",
    "reproduction_candidates",
    "safe_name",
    "sampler_candidate",
    "sampler_full_candidates",
    "unique_candidates",
    "zeta_candidates",
]
