import sys

from pathlib import Path



import numpy as np



from scipy.optimize import minimize





# ============================================================

# PROJECT PATH

# ============================================================



PROJECT_DIR = (

    Path(__file__)

    .resolve()

    .parent

    .parent

)



if str(PROJECT_DIR) not in sys.path:



    sys.path.append(

        str(PROJECT_DIR)

    )





# ============================================================

# IMPORTS

# ============================================================



from cost import (

    hitting_cost,

    memory_cost,

    long_term_cost,

)





from config import (

    M,

    BETA,

    H,

    RHO,

    A_MATRICES,

)





# ============================================================

# NUMERICAL SOLVER SETTINGS

# ============================================================



SLSQP_FTOL = 1e-9



SLSQP_MAXITER = 2000



SIMPLEX_TOL = 1e-7





# ============================================================

# SIMPLEX CONSTRAINTS

# ============================================================



def simplex_constraints():



    return [

        {

            "type": "eq",



            "fun":

                lambda u:

                np.sum(u) - 1.0,

        }

    ]





def simplex_bounds(

    dim,

):



    return [

        (0.0, 1.0)

    ] * dim





# ============================================================

# SIMPLEX HELPERS

# ============================================================



def uniform_simplex_point(

    dim,

):



    return (

        np.ones(

            dim,

            dtype=float,

        )

        /

        float(

            dim

        )

    )





def simplex_feasible(

    u,

    tol=SIMPLEX_TOL,

):



    u = np.asarray(

        u,

        dtype=float,

    )



    if not np.all(

        np.isfinite(

            u

        )

    ):



        return False



    if np.any(

        u < -tol

    ):



        return False



    if np.any(

        u > 1.0 + tol

    ):



        return False



    if abs(

        np.sum(

            u

        )

        - 1.0

    ) > tol:



        return False



    return True





def clean_simplex_solution(

    u,

):



    u = np.asarray(

        u,

        dtype=float,

    ).copy()



    # Remove tiny numerical violations.

    u = np.clip(

        u,

        0.0,

        1.0,

    )



    total = float(

        np.sum(

            u

        )

    )



    if (

        not np.isfinite(

            total

        )

        or

        total <= 0.0

    ):



        raise RuntimeError(

            "Cannot normalize simplex solution: "

            f"sum={total}, u={u}."

        )



    u /= total



    return u





# ============================================================

# GENERIC SLSQP SOLVE

#

# Used by the hitting-cost minimizer and the S-MROO-MAX

# primal update.

# ============================================================



def run_slsqp(

    objective,

    x0,

    dim,

    maxiter=SLSQP_MAXITER,

    ftol=SLSQP_FTOL,

):



    x0 = np.asarray(

        x0,

        dtype=float,

    )



    result = minimize(



        fun=objective,



        x0=x0,



        method="SLSQP",



        bounds=simplex_bounds(

            dim

        ),



        constraints=simplex_constraints(),



        options={

            "ftol":

                ftol,



            "maxiter":

                maxiter,



            "disp":

                False,

        },

    )



    return result





# ============================================================

# CHOOSE BEST SOLVER RESULT

# ============================================================



def result_is_usable(

    result,

):



    if result is None:



        return False



    if not hasattr(

        result,

        "x",

    ):



        return False



    x = np.asarray(

        result.x,

        dtype=float,

    )



    if not simplex_feasible(

        x

    ):



        return False



    try:



        value = float(

            result.fun

        )



    except (

        TypeError,

        ValueError,

    ):



        return False



    if not np.isfinite(

        value

    ):



        return False



    return True





def choose_best_result(

    results,

):



    usable = [

        result

        for result

        in results

        if result_is_usable(

            result

        )

    ]



    successful = [

        result

        for result

        in usable

        if bool(

            result.success

        )

    ]



    if len(

        successful

    ) > 0:



        return min(

            successful,

            key=lambda result:

                float(

                    result.fun

                ),

        )



    return None





# ============================================================

# HITTING-COST MINIMIZER

#

# v_t = argmin_u f_t(u)

# ============================================================



def compute_v_t(

    y_t,

):



    y_t = np.asarray(

        y_t,

        dtype=float,

    )



    D = len(

        y_t

    )



    def objective(

        u,

    ):



        return hitting_cost(

            u,

            y_t,

        )



    # --------------------------------------------------------

    # Build feasible starting points.

    #

    # y_t is used when it is already feasible.

    # Otherwise start from the uniform simplex point.

    # --------------------------------------------------------



    starts = []



    if simplex_feasible(

        y_t

    ):



        starts.append(

            y_t.copy()

        )



    uniform_start = (

        uniform_simplex_point(

            D

        )

    )



    starts.append(

        uniform_start

    )



    results = []



    for x0 in starts:



        result = run_slsqp(

            objective=objective,

            x0=x0,

            dim=D,

            maxiter=SLSQP_MAXITER,

            ftol=SLSQP_FTOL,

        )



        results.append(

            result

        )



        if (

            result.success

            and

            result_is_usable(

                result

            )

        ):



            break



    best_result = (

        choose_best_result(

            results

        )

    )



    if best_result is None:



        diagnostic_parts = []



        for index, result in enumerate(

            results

        ):



            diagnostic_parts.append(

                (

                    f"attempt={index + 1}, "

                    f"success={result.success}, "

                    f"message={result.message}, "

                    f"nit={getattr(result, 'nit', None)}, "

                    f"fun={getattr(result, 'fun', None)}, "

                    f"x={getattr(result, 'x', None)}"

                )

            )



        raise RuntimeError(

            "Failed to compute v_t after retries. "

            +

            " | ".join(

                diagnostic_parts

            )

        )



    v_t = clean_simplex_solution(

        best_result.x

    )



    return v_t





# ============================================================

# THEORETICAL S-MROO-MAX PARAMETERS

#

# q(z) = rho ||z||_inf

#

# L_q = rho

#

# gamma_{R,inf} = min(R,D)

#

#

# Pair 1:

#

# lambda_1 =

#

#     m /

#     (

#         m gamma

#         + beta L_q H^2

#     )

#

# lambda_2 = 0

#

#

# Pair 2:

#

# lambda_1 = 1

#

# lambda_2 =

#

#     m(gamma - 1)

#     + beta L_q H^2

# ============================================================



def get_smroo_max_parameters(

    D,

    R=None,

    parameter_pair=1,

):



    # --------------------------------------------------------

    # MAX-norm Lipschitz constant

    # --------------------------------------------------------



    L_q = float(

        RHO

    )



    # --------------------------------------------------------

    # Decomposition factor

    #

    # gamma_{R,inf} = min(R,D)

    # --------------------------------------------------------



    if R is None:



        gamma_R_q = float(

            D

        )



    else:



        gamma_R_q = float(

            min(

                R,

                D,

            )

        )



    # --------------------------------------------------------

    # beta * L_q * H^2

    # --------------------------------------------------------



    memory_constant = (

        BETA

        *

        L_q

        *

        H**2

    )



    # --------------------------------------------------------

    # THEORETICAL PAIR 1

    # --------------------------------------------------------



    if parameter_pair == 1:



        lambda_1 = (

            M

            /

            (

                M

                *

                gamma_R_q



                +



                memory_constant

            )

        )



        lambda_2 = 0.0



    # --------------------------------------------------------

    # THEORETICAL PAIR 2

    # --------------------------------------------------------



    elif parameter_pair == 2:



        lambda_1 = 1.0



        lambda_2 = (

            M

            *

            (

                gamma_R_q

                - 1.0

            )



            +



            memory_constant

        )



    else:



        raise ValueError(

            "parameter_pair must be 1 or 2."

        )



    return (

        lambda_1,

        lambda_2,

        gamma_R_q,

        L_q,

    )





# ============================================================

# ONE S-MROO-MAX STEP

#

# Algorithm 1:

#

# v_t = argmin f_t(u)

#

# u_t = argmin [

#

#     f_t(u)

#

#     + lambda_1 q(d_t)

#

#     + lambda_2 / 2 ||u-v_t||^2

#

# ]

#

# MAX:

#

# q(d_t) = rho ||d_t||_inf

# ============================================================



def smroo_max_step(
    y_t,
    u_prev,
    A_matrices,
    lambda_1,
    lambda_2,
):
    y_t = np.asarray(y_t, dtype=float)
    u_prev = np.asarray(u_prev, dtype=float)
    A_matrices = np.asarray(A_matrices, dtype=float)

    D = len(y_t)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------
    if y_t.shape != (D,):
        raise ValueError("y_t must be a 1-D vector.")

    if u_prev.shape != (D,):
        raise ValueError(
            "u_prev and y_t must have the same dimension."
        )

    if not np.all(np.isfinite(y_t)):
        raise ValueError(
            f"Non-finite y_t encountered: {y_t}"
        )

    if not np.all(np.isfinite(u_prev)):
        raise ValueError(
            f"Non-finite u_prev encountered: {u_prev}"
        )

    if A_matrices.shape != (D, D, D):
        raise ValueError(
            "A_matrices must have shape "
            f"({D}, {D}, {D}), but got {A_matrices.shape}."
        )

    if not np.all(np.isfinite(A_matrices)):
        raise ValueError(
            "A_matrices contains non-finite values."
        )

    if not simplex_feasible(u_prev):
        raise RuntimeError(
            "u_prev is not simplex-feasible: "
            f"u_prev={u_prev}, sum={np.sum(u_prev)}"
        )

    # --------------------------------------------------------
    # Hitting-cost minimizer
    # --------------------------------------------------------
    v_t = compute_v_t(y_t)

    # ========================================================
    # Exact epigraph reformulation of
    #
    #   rho * ||d_t||_inf
    #
    # Original primal objective:
    #
    #   f_t(u)
    #   + lambda_1 * rho * max_i |d_i(u)|
    #   + lambda_2 / 2 * ||u-v_t||^2
    #
    # Introduce scalar s and solve:
    #
    #   f_t(u)
    #   + lambda_1 * rho * s
    #   + lambda_2 / 2 * ||u-v_t||^2
    #
    # subject to
    #
    #   s >=  d_i(u)
    #   s >= -d_i(u)
    #
    # for every memory coordinate i.
    #
    # This is mathematically equivalent to the max-norm
    # objective, but removes the nonsmooth np.max()/abs()
    # expression from the objective seen by SLSQP.
    # ========================================================

    def unpack_x(x):
        x = np.asarray(x, dtype=float)
        return x[:D], float(x[D])

    def epigraph_objective(x):
        u, s = unpack_x(x)

        f = hitting_cost(u, y_t)

        regularization = (
            lambda_2
            / 2.0
            * np.sum((u - v_t) ** 2)
        )

        value = (
            f
            + lambda_1 * RHO * s
            + regularization
        )

        return float(value)

    # --------------------------------------------------------
    # Determine the number of memory coordinates.
    # --------------------------------------------------------
    reference_memory = np.asarray(
        memory_cost(
            u_prev,
            u_prev,
            A_matrices,
        ),
        dtype=float,
    ).reshape(-1)

    if reference_memory.size == 0:
        raise RuntimeError(
            "memory_cost returned an empty vector."
        )

    if not np.all(np.isfinite(reference_memory)):
        raise RuntimeError(
            "Non-finite memory cost while building "
            "S-MROO-MAX epigraph constraints: "
            f"{reference_memory}"
        )

    n_memory = int(reference_memory.size)

    # --------------------------------------------------------
    # Constraints for x = [u, s].
    # --------------------------------------------------------
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: float(
                np.sum(np.asarray(x[:D], dtype=float))
                - 1.0
            ),
        }
    ]

    for memory_index in range(n_memory):

        def positive_epigraph_constraint(
            x,
            index=memory_index,
        ):
            u, s = unpack_x(x)
            d = np.asarray(
                memory_cost(
                    u,
                    u_prev,
                    A_matrices,
                ),
                dtype=float,
            ).reshape(-1)

            if d.size != n_memory:
                raise RuntimeError(
                    "memory_cost dimension changed during "
                    "S-MROO-MAX optimization: "
                    f"expected {n_memory}, got {d.size}."
                )

            if not np.all(np.isfinite(d)):
                return -1e30

            # s - d_i >= 0
            return float(s - d[index])

        def negative_epigraph_constraint(
            x,
            index=memory_index,
        ):
            u, s = unpack_x(x)
            d = np.asarray(
                memory_cost(
                    u,
                    u_prev,
                    A_matrices,
                ),
                dtype=float,
            ).reshape(-1)

            if d.size != n_memory:
                raise RuntimeError(
                    "memory_cost dimension changed during "
                    "S-MROO-MAX optimization: "
                    f"expected {n_memory}, got {d.size}."
                )

            if not np.all(np.isfinite(d)):
                return -1e30

            # s + d_i >= 0  <=>  s >= -d_i
            return float(s + d[index])

        constraints.append(
            {
                "type": "ineq",
                "fun": positive_epigraph_constraint,
            }
        )

        constraints.append(
            {
                "type": "ineq",
                "fun": negative_epigraph_constraint,
            }
        )

    # --------------------------------------------------------
    # Variable bounds:
    #   0 <= u_i <= 1
    #   s >= 0
    # --------------------------------------------------------
    bounds = simplex_bounds(D) + [(0.0, None)]

    # --------------------------------------------------------
    # Multiple feasible starting points.
    # --------------------------------------------------------
    primal_starts = [
        ("u_prev", u_prev.copy()),
        ("v_t", v_t.copy()),
        (
            "uniform",
            uniform_simplex_point(D),
        ),
    ]

    epigraph_starts = []

    for start_name, u0 in primal_starts:
        d0 = np.asarray(
            memory_cost(
                u0,
                u_prev,
                A_matrices,
            ),
            dtype=float,
        ).reshape(-1)

        if d0.size != n_memory:
            continue

        if not np.all(np.isfinite(d0)):
            continue

        s0 = float(np.max(np.abs(d0)))

        # Start slightly inside the epigraph constraints.
        margin = max(
            1e-10,
            1e-8 * max(1.0, abs(s0)),
        )

        x0 = np.concatenate(
            [
                np.asarray(u0, dtype=float),
                np.array([s0 + margin], dtype=float),
            ]
        )

        epigraph_starts.append(
            (start_name, x0)
        )

    if len(epigraph_starts) == 0:
        raise RuntimeError(
            "Could not construct a valid starting point "
            "for S-MROO-MAX."
        )

    # --------------------------------------------------------
    # Candidate validation for the D+1 dimensional epigraph
    # solve. result_is_usable() cannot be used here because
    # it expects the whole result.x to be a simplex vector.
    # --------------------------------------------------------
    def epigraph_result_is_usable(result):
        if result is None or not hasattr(result, "x"):
            return False

        x = np.asarray(result.x, dtype=float)

        if x.shape != (D + 1,):
            return False

        if not np.all(np.isfinite(x)):
            return False

        u = x[:D]
        s = float(x[D])

        if not simplex_feasible(u):
            return False

        if not np.isfinite(s) or s < -SIMPLEX_TOL:
            return False

        try:
            objective_value = float(result.fun)
        except (TypeError, ValueError):
            return False

        if not np.isfinite(objective_value):
            return False

        d = np.asarray(
            memory_cost(
                u,
                u_prev,
                A_matrices,
            ),
            dtype=float,
        ).reshape(-1)

        if d.size != n_memory:
            return False

        if not np.all(np.isfinite(d)):
            return False

        # Numerical tolerance for the epigraph inequalities.
        epigraph_tol = max(
            1e-7,
            1e-6 * max(
                1.0,
                abs(s),
                float(np.max(np.abs(d))),
            ),
        )

        if np.any(np.abs(d) > s + epigraph_tol):
            return False

        return True

    # --------------------------------------------------------
    # SLSQP attempts.
    # --------------------------------------------------------
    solver_settings = [
        (2000, 1e-9),
        (3000, 1e-9),
        (5000, 1e-8),
    ]

    results = []

    for attempt_index, (
        start_name,
        x0,
    ) in enumerate(epigraph_starts):

        maxiter, ftol = solver_settings[
            min(
                attempt_index,
                len(solver_settings) - 1,
            )
        ]

        result = minimize(
            fun=epigraph_objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": ftol,
                "maxiter": maxiter,
                "disp": False,
            },
        )

        results.append(
            (start_name, result)
        )

        if (
            result.success
            and epigraph_result_is_usable(result)
        ):
            # The problem is convex under the current quadratic
            # hitting/memory formulation, so a successful feasible
            # SLSQP solution is enough. Retain retry logic only for
            # numerical failures.
            break

    successful_results = [
        solver_result
        for _, solver_result in results
        if (
            solver_result.success
            and epigraph_result_is_usable(
                solver_result
            )
        )
    ]

    if len(successful_results) == 0:
        diagnostics = []

        for start_name, solver_result in results:
            candidate = np.asarray(
                getattr(
                    solver_result,
                    "x",
                    np.array([], dtype=float),
                ),
                dtype=float,
            )

            if candidate.size >= D:
                candidate_u = candidate[:D]
                sum_u = float(np.sum(candidate_u))
                min_u = float(np.min(candidate_u))
                max_u = float(np.max(candidate_u))
            else:
                sum_u = np.nan
                min_u = np.nan
                max_u = np.nan

            candidate_s = (
                float(candidate[D])
                if candidate.size > D
                else np.nan
            )

            diagnostics.append(
                (
                    f"start={start_name}, "
                    f"success={getattr(solver_result, 'success', None)}, "
                    f"status={getattr(solver_result, 'status', None)}, "
                    f"message={getattr(solver_result, 'message', None)}, "
                    f"nit={getattr(solver_result, 'nit', None)}, "
                    f"fun={getattr(solver_result, 'fun', None)}, "
                    f"sum_u={sum_u}, "
                    f"min_u={min_u}, "
                    f"max_u={max_u}, "
                    f"s={candidate_s}, "
                    f"x={candidate}"
                )
            )

        raise RuntimeError(
            "S-MROO-MAX epigraph primal update failed "
            "after all SLSQP attempts. "
            + " | ".join(diagnostics)
        )

    best_result = min(
        successful_results,
        key=lambda solver_result: float(
            solver_result.fun
        ),
    )

    best_x = np.asarray(
        best_result.x,
        dtype=float,
    )

    u_t = clean_simplex_solution(
        best_x[:D]
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------
    if not simplex_feasible(u_t):
        raise RuntimeError(
            "S-MROO-MAX solver returned an infeasible "
            "action after normalization: "
            f"u_t={u_t}, sum={np.sum(u_t)}"
        )

    if not np.all(np.isfinite(u_t)):
        raise RuntimeError(
            "S-MROO-MAX solver returned non-finite action: "
            f"{u_t}"
        )

    # --------------------------------------------------------
    # Realized memory cost
    # --------------------------------------------------------
    d_t = np.asarray(
        memory_cost(
            u_t,
            u_prev,
            A_matrices,
        ),
        dtype=float,
    )

    if not np.all(np.isfinite(d_t)):
        raise RuntimeError(
            "S-MROO-MAX produced non-finite memory cost: "
            f"d_t={d_t}"
        )

    return (
        u_t,
        v_t,
        d_t,
    )


# ============================================================

# RUN S-MROO-MAX

#

# If lambda_1 and lambda_2 are not supplied:

#

# use the theoretical parameter pair.

#

# Otherwise:

#

# use the explicitly supplied values.

# ============================================================



def run_smroo_max(

    demand,

    parameter_pair=1,

    R=None,

    lambda_1=None,

    lambda_2=None,

):



    demand = np.asarray(

        demand,

        dtype=float,

    )



    # --------------------------------------------------------

    # Validate demand

    # --------------------------------------------------------



    if demand.ndim != 2:



        raise ValueError(

            "demand must be a 2-D array with shape (T, D). "

            f"Received shape {demand.shape}."

        )



    if not np.all(

        np.isfinite(

            demand

        )

    ):



        raise ValueError(

            "demand contains NaN or infinite values."

        )



    T, D = (

        demand.shape

    )



    if T <= 0:



        raise ValueError(

            "demand must contain at least one time step."

        )



    if D <= 0:



        raise ValueError(

            "demand must contain at least one dimension."

        )



    # --------------------------------------------------------

    # Frame size

    #

    # For the full-horizon experiment we use R = T.

    # --------------------------------------------------------



    if R is None:



        R = T



    R = int(

        R

    )



    if R <= 0:



        raise ValueError(

            "R must be positive."

        )



    # --------------------------------------------------------

    # Compute theoretical parameters

    # --------------------------------------------------------



    (

        theoretical_lambda_1,

        theoretical_lambda_2,

        gamma_R_q,

        L_q,

    ) = get_smroo_max_parameters(



        D=D,



        R=R,



        parameter_pair=parameter_pair,

    )



    # --------------------------------------------------------

    # Use theoretical values unless overridden

    # --------------------------------------------------------



    if lambda_1 is None:



        lambda_1 = (

            theoretical_lambda_1

        )



    if lambda_2 is None:



        lambda_2 = (

            theoretical_lambda_2

        )



    lambda_1 = float(

        lambda_1

    )



    lambda_2 = float(

        lambda_2

    )



    # --------------------------------------------------------

    # Validate parameters

    # --------------------------------------------------------



    if not np.isfinite(

        lambda_1

    ):



        raise ValueError(

            "lambda_1 must be finite."

        )



    if not np.isfinite(

        lambda_2

    ):



        raise ValueError(

            "lambda_2 must be finite."

        )



    if lambda_1 <= 0.0:



        raise ValueError(

            "lambda_1 must be positive."

        )



    if lambda_2 < 0.0:



        raise ValueError(

            "lambda_2 must be nonnegative."

        )



    # --------------------------------------------------------

    # Initial allocation

    # --------------------------------------------------------



    u_prev = (

        np.ones(

            D,

            dtype=float,

        )

        /

        float(

            D

        )

    )



    # --------------------------------------------------------

    # Histories

    # --------------------------------------------------------



    actions = []



    hitting_history = []



    memory_history = []



    switching_history = []



    # ========================================================

    # MAIN ONLINE LOOP

    # ========================================================



    for t in range(

        T

    ):



        y_t = (

            demand[

                t

            ]

        )



        # ----------------------------------------------------

        # Matrix-valued memory configuration

        # ----------------------------------------------------



        A_matrices = np.asarray(

            A_MATRICES,

            dtype=float,

        )



        if A_matrices.shape != (D, D, D):

            raise RuntimeError(

                "A_MATRICES must have shape "

                f"({D}, {D}, {D}), but got {A_matrices.shape}."

            )



        if not np.all(

            np.isfinite(

                A_matrices

            )

        ):



            raise RuntimeError(

                "A_MATRICES contains non-finite values."

            )



        # ----------------------------------------------------

        # S-MROO-MAX update

        # ----------------------------------------------------



        try:



            (

                u_t,

                v_t,

                d_t,

            ) = smroo_max_step(



                y_t=y_t,



                u_prev=u_prev,



                A_matrices=A_matrices,



                lambda_1=lambda_1,



                lambda_2=lambda_2,

            )



        except Exception as error:



            raise RuntimeError(

                "S-MROO-MAX failed at "

                f"t={t}; "

                f"lambda_1={lambda_1}; "

                f"lambda_2={lambda_2}; "

                f"y_t={y_t}; "

                f"u_prev={u_prev}; "

                f"A_matrices_shape={A_matrices.shape}. "

                f"Original error: {error}"

            ) from error



        # ----------------------------------------------------

        # Store action

        # ----------------------------------------------------



        actions.append(

            u_t.copy()

        )



        # ----------------------------------------------------

        # Store hitting cost

        # ----------------------------------------------------



        current_hitting_cost = (

            hitting_cost(

                u_t,

                y_t,

            )

        )



        if not np.isfinite(

            current_hitting_cost

        ):



            raise RuntimeError(

                "Non-finite hitting cost at "

                f"t={t}: "

                f"{current_hitting_cost}"

            )



        hitting_history.append(

            current_hitting_cost

        )



        # ----------------------------------------------------

        # Store memory vector

        # ----------------------------------------------------



        memory_history.append(

            d_t.copy()

        )



        # ----------------------------------------------------

        # Scalar switching magnitude

        #

        # Diagnostic scalar derived from the realized vector-valued

        # matrix-memory cost. This is retained only for compatibility

        # with existing outputs; it is not the S-MROO-MAX max-norm

        # decision penalty or the final long-term cost.

        # ----------------------------------------------------



        switching_history.append(



            np.linalg.norm(

                d_t,

                ord=2,

            )



        )



        # ----------------------------------------------------

        # Next round

        # ----------------------------------------------------



        u_prev = (

            u_t.copy()

        )



    # ========================================================

    # CONVERT HISTORIES

    # ========================================================



    actions = np.asarray(

        actions,

        dtype=float,

    )



    hitting_history = np.asarray(

        hitting_history,

        dtype=float,

    )



    memory_history = np.asarray(

        memory_history,

        dtype=float,

    )



    switching_history = np.asarray(

        switching_history,

        dtype=float,

    )



    # ========================================================

    # FINAL CUMULATIVE COST

    #

    # Same evaluation objective:

    #

    # sum_t f_t

    #

    # +

    #

    # rho ||sum_t d_t||_inf

    # ========================================================



    hitting_cost_total = float(

        np.sum(

            hitting_history

        )

    )



    final_long_term_cost = float(

        long_term_cost(

            memory_history

        )

    )



    total_cost = (

        hitting_cost_total

        +

        final_long_term_cost

    )



    # --------------------------------------------------------

    # Final sanity checks

    # --------------------------------------------------------



    if not np.isfinite(

        hitting_cost_total

    ):



        raise RuntimeError(

            "Final hitting cost is non-finite."

        )



    if not np.isfinite(

        final_long_term_cost

    ):



        raise RuntimeError(

            "Final long-term cost is non-finite."

        )



    if not np.isfinite(

        total_cost

    ):



        raise RuntimeError(

            "Final total cost is non-finite."

        )



    # ========================================================

    # OUTPUT

    # ========================================================



    return {



        "actions":

            actions,



        "hitting_history":

            hitting_history,



        "memory_history":

            memory_history,



        "switching_history":

            switching_history,



        "hitting_cost":

            hitting_cost_total,



        "long_term_cost":

            final_long_term_cost,



        "total_cost":

            total_cost,



        "lambda_1":

            lambda_1,



        "lambda_2":

            lambda_2,



        "theoretical_lambda_1":

            theoretical_lambda_1,



        "theoretical_lambda_2":

            theoretical_lambda_2,



        "gamma_R_q":

            gamma_R_q,



        "L_q":

            L_q,



        "R":

            R,



        "parameter_pair":

            parameter_pair,

    }