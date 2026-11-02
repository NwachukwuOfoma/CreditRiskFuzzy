from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import math

# Membership Functions

@dataclass
class TriangularMF:
    a: float
    b: float
    c: float

    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        if x == self.b:
            return 1.0
        if x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.c - x) / (self.c - self.b)


@dataclass
class TrapezoidalMF:
    a: float
    b: float
    c: float
    d: float

    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        if self.b <= x <= self.c:
            return 1.0
        if x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.d - x) / (self.d - self.c)


@dataclass
class FuzzyTerm:
    name: str
    mf: Callable[[float], float]


class FuzzyVariable:
    def __init__(self, name: str, universe: Tuple[float, float]):
        self.name = name
        self.universe = universe
        self.terms: Dict[str, FuzzyTerm] = {}

    def add_term(self, name: str, mf: Callable[[float], float]):
        self.terms[name] = FuzzyTerm(name, mf)

    def membership(self, term_name: str, x: float) -> float:
        return self.terms[term_name].mf(x)


@dataclass
class AntecedentAtom:
    var: FuzzyVariable
    term_name: str

    def degree(self, x: float) -> float:
        return self.var.membership(self.term_name, x)


@dataclass
class Rule:
    antecedent: List[AntecedentAtom]
    consequent: Tuple[FuzzyVariable, str]
    weight: float = 1.0

    def firing_strength(self, crisp_inputs: Dict[str, float]) -> float:
        degrees = []
        for atom in self.antecedent:
            x = crisp_inputs[atom.var.name]
            degrees.append(atom.degree(x))
        base = min(degrees) if degrees else 0.0
        return base * self.weight


class Defuzzifier:
    @staticmethod
    def centroid(samples: List[Tuple[float, float]]) -> float:
        num = sum(x * mu for x, mu in samples)
        den = sum(mu for _, mu in samples)
        return num / den if den > 0 else float("nan")

    @staticmethod
    def bisector(samples: List[Tuple[float, float]]) -> float:
        total = sum(mu for _, mu in samples)
        if total == 0:
            return float("nan")
        half = total / 2.0
        running = 0.0
        for x, mu in samples:
            running += mu
            if running >= half:
                return x
        return samples[-1][0]

    @staticmethod
    def mean_of_maxima(samples: List[Tuple[float, float]]) -> float:
        if not samples:
            return float("nan")
        m = max(mu for _, mu in samples)
        if m == 0:
            return float("nan")
        xs = [x for x, mu in samples if math.isclose(mu, m, rel_tol=1e-9, abs_tol=1e-12)]
        return sum(xs) / len(xs)


class MamdaniFIS:
    def __init__(self, inputs: List[FuzzyVariable], output: FuzzyVariable, rules: List[Rule]):
        self.inputs = {v.name: v for v in inputs}
        self.output = output
        self.rules = rules

    def _aggregate_output_mf(self, crisp_inputs: Dict[str, float], resolution: int = 401) -> List[Tuple[float, float]]:
        xmin, xmax = self.output.universe
        xs = [xmin + i * (xmax - xmin) / (resolution - 1) for i in range(resolution)]
        aggregated = [0.0] * resolution
        for rule in self.rules:
            strength = rule.firing_strength(crisp_inputs)
            cons_term = rule.consequent[1]
            mf = self.output.terms[cons_term].mf
            for i, x in enumerate(xs):
                aggregated[i] = max(aggregated[i], min(strength, mf(x)))
        return list(zip(xs, aggregated))

    def infer(self, crisp_inputs: Dict[str, float], method: str = "centroid", resolution: int = 401) -> Tuple[float, List[Tuple[float, float]]]:
        samples = self._aggregate_output_mf(crisp_inputs, resolution=resolution)
        if method == "centroid":
            y = Defuzzifier.centroid(samples)
        elif method == "bisector":
            y = Defuzzifier.bisector(samples)
        elif method == "mom":
            y = Defuzzifier.mean_of_maxima(samples)
        else:
            raise ValueError(f"Unknown defuzzification method: {method}")
        return y, samples


def build_credit_risk_fis() -> MamdaniFIS:
    income = FuzzyVariable("Income", (0, 200))
    income.add_term("low", TrapezoidalMF(0, 0, 35, 55))
    income.add_term("medium", TriangularMF(40, 80, 120))
    income.add_term("high", TrapezoidalMF(100, 140, 200, 200))

    dti = FuzzyVariable("DTI", (0, 60))
    dti.add_term("low", TrapezoidalMF(0, 0, 10, 20))
    dti.add_term("medium", TriangularMF(15, 25, 35))
    dti.add_term("high", TrapezoidalMF(30, 40, 60, 60))

    credit = FuzzyVariable("CreditScore", (300, 850))
    credit.add_term("poor", TrapezoidalMF(300, 300, 520, 600))
    credit.add_term("fair", TriangularMF(560, 640, 720))
    credit.add_term("good", TrapezoidalMF(680, 740, 850, 850))

    risk = FuzzyVariable("DefaultRisk", (0, 100))
    risk.add_term("low", TrapezoidalMF(0, 0, 20, 35))
    risk.add_term("medium", TriangularMF(30, 50, 70))
    risk.add_term("high", TrapezoidalMF(65, 80, 100, 100))

    def A(var: FuzzyVariable, term: str) -> AntecedentAtom:
        return AntecedentAtom(var, term)

    rules: List[Rule] = [
        Rule([A(dti, "high"), A(credit, "poor")], (risk, "high"), weight=1.0),
        Rule([A(dti, "high"), A(credit, "fair"), A(income, "low")], (risk, "high"), weight=0.9),
        Rule([A(dti, "medium"), A(credit, "poor")], (risk, "high"), weight=0.9),
        Rule([A(dti, "medium"), A(credit, "fair")], (risk, "medium"), weight=1.0),
        Rule([A(dti, "low"), A(credit, "good")], (risk, "low"), weight=1.0),
        Rule([A(dti, "low"), A(credit, "fair"), A(income, "high")], (risk, "low"), weight=0.9),
        Rule([A(dti, "high"), A(income, "high"), A(credit, "good")], (risk, "medium"), weight=0.6),
        Rule([A(dti, "low"), A(credit, "poor")], (risk, "medium"), weight=0.7),
    ]

    return MamdaniFIS(inputs=[income, dti, credit], output=risk, rules=rules)

def credit_risk_demo():
    fis = build_credit_risk_fis()
    samples = [
        {"Income": 45, "DTI": 38, "CreditScore": 610},
        {"Income": 150, "DTI": 12, "CreditScore": 780},
        {"Income": 60, "DTI": 25, "CreditScore": 650},
        {"Income": 30, "DTI": 45, "CreditScore": 580},
        {"Income": 90, "DTI": 18, "CreditScore": 720},
    ]
    methods = ["centroid", "bisector", "mom"]
    for s in samples:
        print("Inputs:", s)
        for m in methods:
            y, _ = fis.infer(s, method=m)
            print(f"  {m}: DefaultRisk = {y:.2f}")
        print()


if __name__ == "__main__":
    credit_risk_demo()
