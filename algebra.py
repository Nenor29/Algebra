from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import product
import numpy as np
from scipy import linalg
from scipy.linalg import null_space
@dataclass
class Vector:
    components: Dict[str, float]
    @classmethod
    def basis(cls, index: int) -> 'Vector':
        return cls({f"e{index}": 1})
    def __add__(self, other):
        result = {}
        all_bases = set(self.components.keys()) | set(other.components.keys())
        for base in all_bases:
            result[base] = self.components.get(base, 0) + other.components.get(base, 0)
        return Vector(result)
    def __sub__(self, other):
        result = {}
        all_bases = set(self.components.keys()) | set(other.components.keys())
        for base in all_bases:
            result[base] = self.components.get(base, 0) - other.components.get(base, 0)
        return Vector(result)
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector({base: coeff * other for base, coeff in self.components.items()})
        elif isinstance(other, Vector):
            result = {}
            for base1, coeff1 in self.components.items():
                for base2, coeff2 in other.components.items():
                    product = algebra.multiply_basis(base1, base2)
                    for base, value in product.components.items():
                        if base not in result:
                            result[base] = 0
                        result[base] += coeff1 * coeff2 * value
            return Vector(result)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        all_bases = set(self.components.keys()) | set(other.components.keys())
        return all(abs(self.components.get(base, 0) - other.components.get(base, 0)) < 1e-10 
                  for base in all_bases)
    def __str__(self):
        terms = []
        for base, coeff in sorted(self.components.items()):
            if abs(coeff) < 1e-10:  
                continue
            elif coeff == 1:
                terms.append(base)
            elif coeff == -1:
                terms.append(f"-{base}")
            else:
                if coeff > 0:
                    terms.append(f"{coeff}{base}")
                else:
                    terms.append(f"-{abs(coeff)}{base}")
        if not terms:
            return "0"
        result = terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"
        return result
class CustomAlgebra:
    def __init__(self, dimension: int, multiplication_rules: Dict[Tuple[str, str], Dict[str, float]]):
        self.dimension = dimension
        self.multiplication_rules = multiplication_rules
        self.basis_vectors = {f"e{i}": Vector.basis(i) for i in range(1, dimension + 1)}
    def multiply_basis(self, base1: str, base2: str) -> Vector:
        key = (base1, base2)
        if key in self.multiplication_rules:
            return Vector(self.multiplication_rules[key])
        return Vector({})
    def verify_properties(self):
        print("\nMultiplication Table:")
        for i in range(1, self.dimension + 1):
            for j in range(1, self.dimension + 1):
                base1 = f"e{i}"
                base2 = f"e{j}"
                result = self.multiply_basis(base1, base2)
                print(f"{base1} * {base2} = {result}")
def create_example_algebra(dimension):
    dimension = dimension
    rules = {
        ('e1', 'e2'): {'e3': 1},
        ('e1', 'e3'): {'e1': -2},
        ('e2', 'e1'): {'e3': -1},
        ('e2', 'e3'): {'e2': 2},
        ('e3', 'e1'): {'e1': 2},
        ('e3', 'e2'): {'e2': -2},
    }
    return CustomAlgebra(dimension, multiplication_rules=rules)
from itertools import permutations
from functools import lru_cache
@lru_cache(maxsize=None)
def generate_structures(n):
    if n == 1:
        return ['{}']  
    result = []
    for i in range(n-1, 0, -1):
        left_parts = generate_structures(i)
        right_parts = generate_structures(n - i)
        for l in left_parts:
            for r in right_parts:
                result.append(f'({l} * {r})')
    return result
def expression(*vectors, return_difference=False, return_terms=False, return_terms_as_strings=False):
    n = len(vectors)
    var_names = [chr(97 + i) for i in range(n)] 
    local_vars = dict(zip(var_names, vectors))  
    expressions = []
    structures = generate_structures(n)  
    for struct in structures:
        for perm in permutations(var_names): 
            expr_str = struct.format(*perm)
            expressions.append(expr_str)
    terms = []
    for expr_str in expressions:
        expr = eval(expr_str, {}, local_vars)
        terms.append(expr)
    if return_terms:
        return terms
    if return_terms_as_strings:
        return expressions  
    left_side = sum(terms, Vector({}))
    right_side = Vector({})
    if return_difference:
        return left_side - right_side
    return left_side == right_side
def build_identity_matrix(algebra, num_vars):
    basis_names = list(algebra.basis_vectors.keys())
    combinations = list(product(basis_names, repeat=num_vars))
    
    result_matrix = []
    for names in combinations:
        vectors = [algebra.basis_vectors[name] for name in names]
        terms = expression(*vectors, return_terms=True)
        as_strings = expression(*vectors, return_terms_as_strings=True)
        result_matrix.append(terms)
    
    return result_matrix, as_strings
def analyze_matrix(results_matrix):
    unique_keys = set()
    for row in results_matrix:
        for vec in row:
            unique_keys.update(vec.components.keys())
    component_matrices = {key: [] for key in unique_keys}
    for vector in component_matrices:
        for row in results_matrix:
            component_matrices[vector].append([vec.components.get(f'{vector}', 0) for vec in row])
    combined_matrices = [matrix for matrix in component_matrices.values()]
    return tuple(filter(None, combined_matrices))
from sympy import Matrix
def solve_equations(matrix):
    A = np.array([row for row in matrix if any(row)])  
    A_sym = Matrix(A)
    rref_matrix, _ = A_sym.rref()
    null_space = A_sym.nullspace()
    return rref_matrix, null_space
def combine_terms(solutions, as_strings):
    if solutions is None or len(solutions) == 0:
        print("No solutions found.")
        return None
    terms = as_strings
    equations = []
    for idx, solution in enumerate(solutions):
        coeffs = np.array(solution).flatten()
        result_terms = []
        formatted_coeff = format_coefficient(coeffs[0])
        if coeffs[0] == 1:
            result_terms.append(f"{terms[0]}")
        elif coeffs[0] == -1:
            result_terms.append(f"- {terms[0]}")
        elif abs(coeffs[0]) > 1e-10:
            result_terms.append(f"{formatted_coeff}{terms[0]}")
        for coeff, term in zip(coeffs[1:], terms[1:]):
            formatted_coeff = format_coefficient(coeff)
            if abs(coeff) > 1e-10:
                if coeff == 1:
                    if result_terms:
                        result_terms.append(f"+ {term}")
                    else:
                        result_terms.append(f"{term}")
                elif coeff == -1:
                    result_terms.append(f"- {term}")
                elif coeff > 0 and coeff != 1:
                    if result_terms:
                        result_terms.append(f"+ {formatted_coeff}{term}")
                    else:
                        result_terms.append(f"{formatted_coeff}{term}")
                else:
                    result_terms.append(f"{formatted_coeff}{term}")
        if result_terms != "":
            equation = f"Identity {idx + 1}: " + '  '.join(result_terms) + ' = 0'
        equations.append(equation)
        print(equation)
    
    return equations
def format_coefficient(coeff):
    if coeff.is_integer():
        return f"{int(coeff)}"
    return f"{coeff:.4f}"
import numpy as np
if __name__ == "__main__":
    algebra = create_example_algebra(dimension=3)
    matrix, as_strings = build_identity_matrix(algebra, 3)
    matrices = analyze_matrix(matrix)
    e_matrix = np.vstack(matrices)
    e_matrix = e_matrix[~np.all(e_matrix == 0, axis=1)]
    rref, null_space_vectors = solve_equations(e_matrix)
    numpy_solutions = []
    for vec in null_space_vectors:
        numpy_solutions.append(np.array(vec).astype(float))
    equations = combine_terms(numpy_solutions, as_strings)
    
