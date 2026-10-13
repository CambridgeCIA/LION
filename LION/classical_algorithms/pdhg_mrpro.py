"""PDHG algorithm."""

# based on Mrpro implementation, which has the following license:
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS

#    APPENDIX: How to apply the Apache License to your work.

#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.

#    Copyright 2023 Physikalisch-Technische Bundesanstalt,
#    Copyright 2023 MRpro Contributors

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# simplified and modified to include checkpointing to reduce memory usage for unrolling

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
from mrpro.operators import (
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctional,
    ProximableFunctionalSeparableSum,
)
from mrpro.operators.functionals import ZeroFunctional
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm


def pdhg(
    f: ProximableFunctionalSeparableSum | ProximableFunctional | None,
    g: ProximableFunctionalSeparableSum | ProximableFunctional | None,
    operator: LinearOperator | LinearOperatorMatrix | None,
    initial_values: Sequence[torch.Tensor],
    max_iterations: int,
    primal_stepsize: float | torch.Tensor | None,
    dual_stepsize: float | torch.Tensor | None,
    relaxation: float = 1.0,
    *,
    use_checkpoint: bool = False,
    checkpoint_chunk_size: int | None = None,
    checkpoint_use_reentrant: bool = False,
    use_tqdm: bool = False,
):
    """Run the primal-dual hybrid gradient algorithm.

    The implementation supports a block operator and separable proximal functionals.
    Gradient checkpointing can be used to reduce memory usage when differentiating through the iterations.

    Args:
        f: Proximal functional applied to the dual variables.
            If ``None``, a zero functional is used for every operator row.
        g: Proximal functional applied to the primal variables.
            If ``None``, a zero functional is used for every operator column.
        operator: Block linear operator used in the primal and dual updates.
        initial_values: Initial primal variables, with one tensor per operator column.
        max_iterations: Number of PDHG iterations.
        primal_stepsize: Step size used in the primal update.
        dual_stepsize: Step size used in the dual update.
        relaxation: Interpolation weight between the previous and updated primal variables.
            A value of ``0.0`` keeps the previous primal as the relaxed variable,
            while ``1.0`` uses the updated primal directly.
        use_checkpoint: Whether to use gradient checkpointing.
        checkpoint_chunk_size: Number of PDHG iterations in each checkpointed chunk.
            If ``None``, all iterations are placed in one chunk.
        checkpoint_use_reentrant: Value passed to ``torch.utils.checkpoint.checkpoint``.
        use_tqdm: Whether to display iteration progress.

    Returns:
        The final primal variables.

    Raises:
        ValueError: If the number of functionals does not match the block
            operator dimensions.
    """
    if f is None and g is None:
        warnings.warn(
            "Both f and g are None. The objective is constant. Returning x0 as a possible solution",
            stacklevel=2,
        )
        return tuple(initial_values)

    if operator is None:
        # Use identity operator if no operator is supplied
        n_rows = len(f) if isinstance(f, ProximableFunctionalSeparableSum) else 1
        n_columns = len(g) if isinstance(g, ProximableFunctionalSeparableSum) else 1
        if n_rows != n_columns:
            raise ValueError(
                "If operator is None, the number of elements in f and g should be the same"
            )
        operator_matrix = LinearOperatorMatrix.from_diagonal(
            *((IdentityOp(),) * n_rows)
        )
    else:
        if isinstance(operator, LinearOperator):
            # We always use a matrix of operators for homogeneous handling
            operator_matrix = LinearOperatorMatrix.from_diagonal(operator)
        else:
            operator_matrix = operator
        n_rows, n_columns = operator_matrix.shape

    # We always use a separable sum for homogeneous handling, even if it is just a ZeroFunctional
    if f is None:
        f_sum: ProximableFunctionalSeparableSum = ProximableFunctionalSeparableSum(
            *(ZeroFunctional(),) * n_rows
        )
    elif isinstance(f, ProximableFunctional):
        f_sum = ProximableFunctionalSeparableSum(f)
    else:
        f_sum = f

    if len(f_sum) != n_rows:
        raise ValueError(
            "Number of rows in operator does not match number of functionals in f"
        )

    if g is None:
        g_sum: ProximableFunctionalSeparableSum = ProximableFunctionalSeparableSum(
            *(ZeroFunctional(),) * n_columns
        )
    elif isinstance(g, ProximableFunctional):
        g_sum = ProximableFunctionalSeparableSum(g)
    else:
        g_sum = g

    if len(g_sum) != n_columns:
        raise ValueError(
            "Number of columns in operator does not match number of functionals in g"
        )

    if primal_stepsize is None or dual_stepsize is None:
        # choose primal and dual step size such that their product is 1/|operator|**2 to ensure convergence
        operator_norm = operator_matrix.operator_norm(
            *[torch.randn_like(v) for v in initial_values]
        ).amax()
        if primal_stepsize is None and dual_stepsize is None:
            primal_stepsize_ = dual_stepsize_ = 1.0 / operator_norm
        elif primal_stepsize is None and dual_stepsize is not None:
            primal_stepsize_ = 1 / (operator_norm * dual_stepsize)
            dual_stepsize_ = torch.as_tensor(dual_stepsize)
        elif dual_stepsize is None and primal_stepsize is not None:
            dual_stepsize_ = 1 / (operator_norm * primal_stepsize)
            primal_stepsize_ = torch.as_tensor(primal_stepsize)
        else:
            primal_stepsize_ = torch.as_tensor(primal_stepsize)
            dual_stepsize_ = torch.as_tensor(dual_stepsize)
    else:
        primal_stepsize_ = torch.as_tensor(primal_stepsize)
        dual_stepsize_ = torch.as_tensor(dual_stepsize)

    primals_relaxed: Sequence[torch.Tensor] = initial_values
    duals: Sequence[torch.Tensor] = (0 * operator_matrix)(*initial_values)
    primals: Sequence[torch.Tensor] = initial_values

    def _one_iter(
        primals: Sequence[torch.Tensor],
        primals_relaxed: Sequence[torch.Tensor],
        duals: Sequence[torch.Tensor],
    ) -> tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        duals = tuple(
            dual + dual_stepsize_ * step
            for dual, step in zip(
                duals, operator_matrix(*primals_relaxed), strict=False
            )
        )
        duals = f_sum.prox_convex_conj(*duals, sigma=dual_stepsize_)

        primals_new = tuple(
            primal - primal_stepsize_ * step
            for primal, step in zip(primals, operator_matrix.H(*duals), strict=False)
        )

        primals_new = g_sum.prox(*primals_new, sigma=primal_stepsize_)

        primals_relaxed = tuple(
            torch.lerp(primal, primal_new, relaxation)
            for primal, primal_new in zip(primals, primals_new, strict=False)
        )

        primals = primals_new

        return primals, primals_relaxed, duals

    if (not use_checkpoint) or (not torch.is_grad_enabled()):
        iterator = (
            tqdm(range(max_iterations), desc="PDHG iterations", dynamic_ncols=True)
            if use_tqdm
            else range(max_iterations)
        )
        for _ in iterator:
            primals, primals_relaxed, duals = _one_iter(primals, primals_relaxed, duals)
        return primals

    def _pack(
        p: Sequence[torch.Tensor],
        p_relaxed: Sequence[torch.Tensor],
        d: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        return (*p, *p_relaxed, *d)

    def _unpack(
        flat: Sequence[torch.Tensor],
    ) -> tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        p = flat[:n_columns]
        p_relaxed = flat[n_columns : 2 * n_columns]
        d = flat[2 * n_columns : 2 * n_columns + n_rows]
        return p, p_relaxed, d

    if checkpoint_chunk_size is None:
        checkpoint_chunk_size = max_iterations

    flat_pack = _pack(primals, primals_relaxed, duals)
    last_partial_chunk = 1 if (max_iterations % checkpoint_chunk_size > 0) else 0
    n_chunks = max_iterations // checkpoint_chunk_size + last_partial_chunk
    remaining = max_iterations
    iterator = (
        tqdm(range(n_chunks), desc="PDHG chunks") if use_tqdm else range(n_chunks)
    )
    for _ in iterator:
        this_chunk = min(checkpoint_chunk_size, remaining)
        remaining -= this_chunk

        def run_chunk(
            *flat_pack: torch.Tensor, n_iterations: int = this_chunk
        ) -> Sequence[torch.Tensor]:
            # Run up to chunk_size iterations inside one checkpointed region
            for _ in range(n_iterations):
                flat_pack = _pack(*_one_iter(*_unpack(flat_pack)))
            return flat_pack

        flat_pack = checkpoint(
            run_chunk, *flat_pack, use_reentrant=checkpoint_use_reentrant
        )

    primals, _, _ = _unpack(flat_pack)
    return primals
