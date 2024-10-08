from abc import ABC, abstractmethod
import numpy as np

class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, val):
        pass
   
    def decode(self, lst):
        v, p = self.parse(lst)
        if p == 0:
            return None
        return v

class SymbolicInts(Encoder):
    """
    one token per int from min to max (0 to 1 for binary, -10 to 10 for bounded ints, 0 to Q-1 for modular)
    optionally: add a prefix, e.g. E-100 E100 for exponents, N1 N5 for dimensions 
    """
    def __init__(self, min, max, prefix = ''):
        super().__init__()
        self.prefix = prefix
        self.symbols = [self.prefix + str(i) for i in range(min, max+1)]

    def encode(self, value):
        return [self.prefix+str(value)]

    def parse(self, lst):
        if len(lst) == 0 or (not lst[0] in self.symbols):
            return None, 0
        return  int(lst[0][len(self.prefix):]), 1


class PositionalInts(Encoder):
    """
    Single integers, in base params.base (positive base), with the sign
    """
    def __init__(self, base=10):
        super().__init__()
        self.base = base
        self.symbols = ['+', '-'] + [str(i) for i in range(self.base)]

    def encode(self, value):
        if value != 0:
            prefix = []
            w = abs(value)
            while w > 0:
                prefix.append(str(w % self.base))
                w = w // self.base
            prefix = prefix[::-1]
        else:
            prefix =['0']
        prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def parse(self,lst):
        if len(lst) <= 1 or (lst[0] != '+' and lst[0] != '-'):
            return None, 0
        res = 0
        pos = 1
        for x in lst[1:]:
            if not (x.isdigit()):
                break
            res = res * self.base + int(x)
            pos += 1
        if pos < 2: return None, pos
        return -res if lst[0] == '-' else res, pos

class NumberArray(Encoder):
    """
    Array of integers, in base params.base (any shape)
    TODO modify to support float, complex (rationals), different subencoders
    """
    def __init__(self, params, max_dim, dim_prefix, tensor_dim, code='pos_int'):
        super().__init__()
        self.tensor_dim = tensor_dim
        self.symbols = []
        self.dimencoder = SymbolicInts(1, max_dim, dim_prefix)
        self.symbols.extend(self.dimencoder.symbols)
        if code == 'pos_int':
            self.subencoder = PositionalInts(params.base)
        else:
            self.subencoder = SymbolicInts(params.min_int, params.max_int)
        self.symbols.extend(self.subencoder.symbols)

    def encode(self, vector):
        lst = []
        assert len(np.shape(vector)) == self.tensor_dim
        for d in np.shape(vector):
            lst.extend(self.dimencoder.encode(d))
        for val in np.nditer(np.array(vector)):
            lst.extend(self.subencoder.encode(val))
        return lst

    def decode(self, lst):
        shap = [] 
        h = lst
        for _ in range(self.tensor_dim):
            v, _ = self.dimencoder.parse(h)
            if v is None:
                return None
            shap.append(v)
            h = h[1:]
        m = np.zeros(tuple(shap), dtype=int)
        for val in np.nditer(m, op_flags=['readwrite']):
            v, pos = self.subencoder.parse(h)
            if v is None:
                return None
            h = h[pos:]
            val[...] = v      
        return m

