"""PyQUBO assembly with stable external indices and reusable placeholders."""
from dataclasses import fields

from .problem import ProblemInstance


class ParameterizedProblem(ProblemInstance):
    def __init__(self, problem, approximate=()):
        from pyqubo import Binary, Constraint, Num, Placeholder

        super().__init__(**{f.name: getattr(problem, f.name) for f in fields(ProblemInstance)})
        self._labels = {f"x_{i}": i for i in range(self.n_variables)}
        self._placeholders = {name: f"penalty_{i}" for i, name in enumerate(self.penalty_qubos)}
        variables = {i: Binary(label) for label, i in self._labels.items()}

        def expression(qubo, offset):
            expr = Num(float(offset))
            for (i, j), coefficient in qubo.items():
                expr += coefficient * variables[i] * variables[j]
            return expr

        hamiltonian = expression(self.objective_qubo, self.objective_offset)
        for name, (qubo, offset) in self.penalty_qubos.items():
            penalty = expression(qubo, offset)
            # Approximate encodings can reward feasible points with negative energy.
            # Zero penalty is not a feasibility test for these encodings.
            if name not in approximate:
                penalty = Constraint(penalty, label=name)
            hamiltonian += Placeholder(self._placeholders[name]) * penalty
        self._model = hamiltonian.compile()

    def _feed(self, weights):
        weights = {**self.default_penalty_weights, **(weights or {})}
        if set(weights) - set(self.penalty_qubos):
            raise ValueError("Unknown penalty names")
        return {placeholder: weights[name] for name, placeholder in self._placeholders.items()}

    def to_qubo(self, penalty_weights=None):
        qubo, offset = self._model.to_qubo(feed_dict=self._feed(penalty_weights))
        mapped = {}
        for (a, b), value in qubo.items():
            i, j = sorted((self._labels[a], self._labels[b]))
            mapped[i, j] = mapped.get((i, j), 0.0) + float(value)
        return {k: v for k, v in mapped.items() if v != 0}, float(offset)

    def penalty_diagnostics(self, x, weights):
        sample = {label: int(x[self._labels[label]]) for label in self._model.variables}
        decoded = self._model.decode_sample(sample, vartype="BINARY", feed_dict=self._feed(weights))
        return {name: {"penalty_satisfied": bool(ok), "penalty_value": float(value)}
                for name, (ok, value) in decoded.constraints().items()}


def parameterize(problem, cfg, encodings=None):
    compiler = cfg.get("penalty_search", {}).get("compiler", "native")
    if compiler == "native":
        return problem
    if compiler != "pyqubo":
        raise ValueError(f"Unknown QUBO compiler: {compiler}")
    approximate = {name for name, choice in (encodings or {}).items()
                   if choice.get("encoding") == "unbalanced"}
    return ParameterizedProblem(problem, approximate)
