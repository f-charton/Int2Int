
from abc import ABC, abstractmethod
import numpy as np
import math
from logging import getLogger

logger = getLogger()


class Generator(ABC):
    def __init__(self, params):
        super().__init__()

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass

# empty for now
class Sequence(Generator):
    def __init__(self, params, dims):
        super().__init__(params)

        self.operation = params.operation
        self.maxint = params.maxint
        self.minint = params.minint
        self.dims = dims
        self.modulus = params.modulus

        
    def integer_sequence(self, len, rng, type=None, max=None):
        maxint = self.maxint if max is None else max
        return rng.randint(self. minint, maxint + 1, len)

    def integer_matrix(self, n, p, rng):
        maxint = (int)(self.maxint + 0.5)
        return rng.randint(- maxint, maxint + 1, (n, p))

    def generate(self, rng, type=None):
        if self.operation in ["fraction_simplify","fraction_round"]:
            integers = self.integer_sequence(3, rng)
            if self.operation == "fraction_simplify":
                g = math.gcd(integers[1],integers[2])
                if integers[0] == 1:
                    integers[0] = rng.randint(2, self.maxint + 1)
                inp = [integers[0] * integers[1] // g, integers[0] * integers[2] // g ]
                out = [integers[1] // g , integers[2] // g]
            else:
                m1 = min(integers[1],integers[2])
                m2 = max(integers[1],integers[2])
                if m2 == m1:
                    m1 = m2 - 1
                inp = [integers[0] * m2 + m1, m2]
                out = integers[0]
            return inp, out

        if self.operation in ["fraction_add", "fraction_compare", "fraction_determinant", "fraction_product"]:
            inp = self.integer_sequence(4, rng)
            if self.operation == "fraction_add":
                num = inp[0] * inp[3] + inp[1] * inp[2]
                den = inp[1] * inp[3]
                g = math.gcd(num, den)
                out = [int(num // g), int(den // g)]
            elif self.operation == "fraction_product":
                num = inp[0] * inp[2]
                den = inp[1] * inp[3]
                g = math.gcd(num, den)
                out = [int(num // g), int(den // g)]
            elif self.operation == "fraction_determinant":
                out = inp[0] * inp[3] - inp[1] * inp[2]    
            else: 
                cmp = inp[0] * inp[3] - inp[1] * inp[2]
                out = 1 if cmp > 0 else 0
            return inp, out
        if self.operation in ["modular_add","modular_mul"]:
                inp = self.integer_sequence(2, rng, type)
                out = (inp[0] + inp[1]) % self.modulus if self.operation =="modular_add" else (inp[0] * inp[1]) % self.modulus
                return inp, out
        if self.operation in ["gcd"]:
            inp = self.integer_sequence(2, rng, type)
            out = math.gcd(inp[0], inp[1])
            return inp, out
        if self.operation == "matrix_rank":
            maxrank = min(self.dims[0], self.dims[1])
            rank = rng.randint(1, maxrank + 1)
            
            P = self.integer_matrix(self.dims[0], rank, rng)
            Q = self.integer_matrix(rank, self.dims[1], rng)
            input = P @ Q
            check_rank = np.linalg.matrix_rank(input)
            if check_rank != rank:
                return None
            return input, rank

        return None

    def evaluate(self, src, tgt, hyp):
        return 0, 0, 0, 0
