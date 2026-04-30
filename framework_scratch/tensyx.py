from typing import List, Literal, Tuple, Union

import numpy as np

inf = np.inf

class Tensor:
    """
    A Tensor object represent a multidimensional array. The class uses np.ndarray class for the multidimensional array implementation and implements automatic 
    derivation using graph.
    """
    __array_priority__ = 1
    def __init__(self, tensor: np.ndarray, parents: List = [], requires_grad: bool = True):
        """
        Initializes the Tensor class.

        :param np.ndarray **tensor**: Multidimensional array.
        :param List[Tensor] **parents**: .
        :param bool **requires_grad**: Boolean that specifies if the tensor gradient is calculated. Set to `True`.
        """
        self.tensor = tensor
        if self.tensor.dtype in [np.float128, np.float64]:
            self.tensor = self.tensor.astype(np.float32)

        self.parents = parents

        self.requires_grad = requires_grad

        if requires_grad:
            self.grad = np.zeros_like(tensor)
            self._backward = lambda: None
        else:
            self.parents = []

    def __add__(self, B: Union[int, float, np.ndarray]):
        """
        Built-in Python method that defines add operator (+) between a Tensor object and an integer, a floating point, a ndarray or a Tensor.

        :param Union[int, float, np.ndarray] **B**: Object to add to the Tensor.
        """
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor + B, parents=[self], requires_grad=self.requires_grad)
        elif isinstance(B, Tensor):
            parents = []
            if self.requires_grad:
                parents.append(self)
            if B.requires_grad:
                parents.append(B)
            C = Tensor(self.tensor + B.tensor, parents=parents, requires_grad=self.requires_grad or B.requires_grad)
        
        if C.requires_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad.shape != C.grad.shape:
                        self.grad += elementary_operation_broadcasting_backward(self.grad, C.grad)
                    else:
                        self.grad += C.grad
                if isinstance(B, Tensor):
                    if B.requires_grad:
                        if B.grad.shape != C.grad.shape:
                            B.grad += elementary_operation_broadcasting_backward(B.grad, C.grad)
                        else:
                            B.grad += C.grad
            
            C._backward = _backward

        return C

    def __radd__(self, B: Union[int, float, np.ndarray]):
        return self.__add__(B)        
    
    def __sub__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor - B, parents=[self], requires_grad=self.requires_grad)
        elif isinstance(B, Tensor):
            parents = []
            if self.requires_grad:
                parents.append(self)
            if B.requires_grad:
                parents.append(B)
            C = Tensor(self.tensor - B.tensor, parents=parents, requires_grad=self.requires_grad or B.requires_grad)
        
        if C.requires_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad.shape != C.grad.shape:
                        self.grad += elementary_operation_broadcasting_backward(self.grad, C.grad)
                    else:
                        self.grad += C.grad
                if isinstance(B, Tensor):
                    if B.requires_grad:
                        if B.grad.shape != C.grad.shape:
                            B.grad -= elementary_operation_broadcasting_backward(B.grad, C.grad)
                        else:
                            B.grad -= C.grad            
            C._backward = _backward

        return C
    
    def __rsub__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(B - self.tensor, parents=[self], requires_grad=self.requires_grad)
        elif isinstance(B, Tensor):
            parents = []
            if self.requires_grad:
                parents.append(self)
            if B.requires_grad:
                parents.append(B)
            C = Tensor(B.tensor - self.tensor, parents=parents, requires_grad=self.requires_grad or B.requires_grad)
        
        if C.requires_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad.shape != C.grad.shape:
                        self.grad -= elementary_operation_broadcasting_backward(self.grad, C.grad)
                    else:
                        self.grad -= C.grad
                if isinstance(B, Tensor):
                    if B.requires_grad:
                        if B.grad.shape != C.grad.shape:
                            B.grad += elementary_operation_broadcasting_backward(B.grad, C.grad)
                        else:
                            B.grad += C.grad   
            
            C._backward = _backward

        return C

    def __mul__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor * B, parents=[self], requires_grad=self.requires_grad)
        elif isinstance(B, Tensor):
            parents = []
            if self.requires_grad:
                parents.append(self)
            if B.requires_grad:
                parents.append(B)
            C = Tensor(self.tensor * B.tensor, parents=parents, requires_grad=self.requires_grad or B.requires_grad)

        if C.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(B, (int, float, np.ndarray)):
                        self.grad += B * C.grad
                    elif isinstance(B, Tensor):
                        self.grad += B.tensor * C.grad
                if isinstance(B, Tensor):
                    if B.requires_grad:
                        B.grad += self.tensor * C.grad
            
            C._backward = _backward

        return C
    
    def __rmul__(self, B: Union[int, float, np.ndarray]):
        return self.__mul__(B)
    
    def __truediv__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor / B, parents=[self], requires_grad=self.requires_grad)
        elif isinstance(B, Tensor):
            parents = []
            if self.requires_grad:
                parents.append(self)
            if B.requires_grad:
                parents.append(B)

            C = Tensor(self.tensor / B.tensor, parents=parents, requires_grad=self.requires_grad or B.requires_grad)

        if C.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(B, (int, float)):
                        self.grad += C.grad / B
                    elif isinstance(B, Tensor):
                        self.grad += C.grad / B.tensor
                if isinstance(B, Tensor):
                    if B.requires_grad:
                        B.grad += - (self.tensor * C.grad / (B.tensor**2)).sum(axis=-1, keepdims=True)
            
            C._backward = _backward

        return C
    
    def __matmul__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor @ B, parents=[self], requires_grad=self.requires_grad)
        elif isinstance(B, Tensor):
            parents = []
            if self.requires_grad:
                parents.append(self)
            if B.requires_grad:
                parents.append(B)
            C = Tensor(self.tensor @ B.tensor, parents=parents, requires_grad=self.requires_grad or B.requires_grad)
        else:
            return NotImplemented
        
        if C.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(B, (int, float)):
                        self.grad += C.grad @ B
                    elif isinstance(B, np.ndarray):
                        self.grad += C.grad @ B.T
                    elif isinstance(B, Tensor):
                        self.grad += C.grad @ B.tensor.T
                if B.requires_grad:
                    B.grad += self.tensor.T @ C.grad

            C._backward = _backward
        
        return C

    def __rmatmul__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(B @ self.tensor, parents=[self], requires_grad=self.requires_grad)
        
        if C.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(B, (int, float)):
                        self.grad += B @ self.tensor
                    elif isinstance(B, np.ndarray):
                        self.grad += B.T @ C.grad

            C._backward = _backward

        return C
    
    def __neg__(self):
        A = Tensor(-self.tensor, parents=[self], requires_grad=self.requires_grad)
        if A.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += -1 * A.grad
            A._backward = _backward
        return A
    
    # Logical operators

    def __and__(self, B):
        if isinstance(B, (int, float, np.ndarray)):
            return Tensor(self.tensor & B, requires_grad=False)
        elif isinstance(B, Tensor):
            return Tensor(self.tensor & B.tensor, requires_grad=False)
        else:
            return NotImplemented
    
    # Comparator operators
    def __lt__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor < B, requires_grad=False)
        elif isinstance(B, Tensor):
            C = Tensor(self.tensor < B.tensor, requires_grad=False)
        return C

    def __le__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor <= B, requires_grad=False)
        elif isinstance(B, Tensor):
            C = Tensor(self.tensor <= B.tensor, requires_grad=False)
        return C

    def __eq__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float)):
            C = Tensor(self.tensor == B, requires_grad=False)
        elif isinstance(B, np.ndarray):
            C = Tensor(self.tensor == B, requires_grad=False)
        elif isinstance(B, Tensor):
            if self.shape == B.shape:
                C = Tensor(self.tensor == B.tensor, requires_grad=False)
            else:
                return NotImplemented
        return C
        
    def __ne__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor != B, requires_grad=False)
        elif isinstance(B, Tensor):
            C = Tensor(self.tensor != B.tensor, requires_grad=False)
        return C

    def __gt__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = Tensor(self.tensor > B, requires_grad=False)
        elif isinstance(B, Tensor):
            C = Tensor(self.tensor > B.tensor, requires_grad=False)
        return C

    def __ge__(self, B: Union[int, float, np.ndarray]):
        if isinstance(B, (int, float, np.ndarray)):
            C = self.tensor >= B
        elif isinstance(B, Tensor):
            C = self.tensor >= B.tensor
        return C

    def __repr__(self):
        return self.tensor.__repr__()
    
    def __getitem__(self, i):
        return Tensor(self.tensor[i], requires_grad=False)
    
    def __setitem__(self, i, j):
        self.tensor[i] = j
        
    def __hash__(self):
        return id(self)
    
    def __len__(self):
        if len(self.tensor.shape) == 0:
            return 1
        else:
            return self.tensor.shape[0]

    @property
    def dtype(self):
        return self.tensor.dtype
    
    @property
    def shape(self):
        return self.tensor.shape
    
    @property
    def T(self):
        A = Tensor(self.tensor.T, parents=self.parents, requires_grad=self.requires_grad)

        if A.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += A.grad.T
                
            A._backward = _backward

        return A

    def item(self, loc=None):
        if loc is None:
            loc = 0
        return self.tensor.item(loc)

    def astype(self, type):
        self.tensor = self.tensor.astype(type)
        return self
    
    def sum(self, axis: Union[int, None]=None, keepdims: bool = False):
        A = Tensor(self.tensor.sum(axis, keepdims=keepdims), parents=[self], requires_grad=self.requires_grad)

        if A.requires_grad:
            def _backward():
                if axis is not None:
                    repeat = self.shape[axis]
                self.grad += A.grad if axis is None else np.tile(A.grad, reps=repeat).reshape(self.shape)

            A._backward = _backward
        return A
    
    def mean(self, axis: Union[int, None]=None, keepdims: bool = False):
        A = Tensor(self.tensor.mean(axis, keepdims=keepdims), parents=[self], requires_grad=self.requires_grad)

        if A.requires_grad:
            def _backward():
                if axis is not None:
                    repeat = self.shape[axis]
                    p = repeat
                else:
                    p = 1
                    for i in self.shape:
                        p *= i
                
                self.grad += A.grad / p if axis is None else np.tile(A.grad, reps=repeat).reshape(self.shape) / p

            A._backward = _backward
        return A
    
    def min(self, axis=None, keepdims: bool = False):
        return Tensor(np.min(self.tensor, axis=axis, keepdims=keepdims), requires_grad=False)
    
    def max(self, axis=None, keepdims: bool = False):
        return Tensor(np.max(self.tensor, axis=axis, keepdims=keepdims), requires_grad=False)
    
    def argmax(self, axis=None, keepdims: bool = False):
        return Tensor(np.argmax(self.tensor, axis, keepdims=keepdims), requires_grad=False)
    
    def unique(self, type = float) -> list[int]:
        return np.unique(self.tensor.astype(type)).tolist()
    
    def logits(self):
        if len(self.parents) == 1:
            return self.parents[0]
        else:
            raise Exception(f"Tensor has multiples parents ({len(self.parents)}).")
        
    def clip(self, min, max):
        self.tensor = self.tensor.clip(min, max)
        return self
    
    def get_column_along_dim(self, dim: int, indices: Union[Tuple[int], List[int]]):
        if len(indices) != self.tensor.ndim - 1:
            raise
        indexer = list(indices)
        indexer.insert(dim, slice(None))
        return self.tensor[tuple(indexer)]
    
    def repeat(self, repeat: int, axis: int):
        A = Tensor(np.repeat(self.tensor, repeats=repeat, axis=axis), parents=[self], requires_grad=self.requires_grad)

        if A.requires_grad:
            def _backward():
                self.grad += A.tensor[self.shape]

            A._backward = _backward
        return A

    def expand_dims(self, axis: int):
        A = Tensor(np.expand_dims(self.tensor, axis=axis), parents=self.parents, requires_grad=self.requires_grad)

        if A.requires_grad:
            def _backward():
                self.grad += A.tensor.squeeze(axis=axis)
            
            A._backward = _backward()

        return A
    
    # Automatic differentiation
    def _clear_grads(self):
        tree = self.parents.copy()

        while not len(tree) == 0:
            A = tree.pop()
            A.grad = np.zeros_like(A.grad)
            for parent in A.parents:
                if parent not in tree:
                    tree.append(parent)
    
    def backward(self):
        self._clear_grads()

        done = set()
        topology_order = []

        def cross(A):
            done.add(A)
            for parent in A.parents:
                if parent not in done:
                    cross(parent)

            topology_order.append(A)
        
        cross(self)
        self.grad = np.ones_like(self.grad)
        for A in reversed(topology_order):
            A._backward()

def linspace(start, end, num: int, endpoint: bool = True, requires_grad: bool = False):
    return Tensor(tensor=np.linspace(start, end, num, endpoint=endpoint), requires_grad=requires_grad)

def zeros(shape: Tuple[int], requires_grad: bool = True):
    return Tensor(tensor=np.zeros(shape=shape), requires_grad=requires_grad)

class random:
    @staticmethod
    def rand(size: Union[Tuple[int, int], None] = None) -> Union[float, Tensor]:
        if size is None:
            return np.random.randn()
        else:
            return Tensor(np.random.random(size))
        
    def randint(low: int, high: int = None, size: Union[Tuple[int, int], None] = None) -> Union[int, Tensor]:
        if size is None:
            return np.random.randint(low, high, size)
        else:
            return np.random.randint(low, high, size)
             
    @staticmethod
    def uniform(a: Union[int, float] = 0., b: Union[int, float] = 1.0, size: Union[Tuple[int, int], None] = None) -> Union[float, Tensor]:
        if size is None:
            return np.random.uniform(a, b, size)
        else:
            return Tensor(np.random.uniform(a, b, size))
    
    @staticmethod
    def normal(mean: Union[int, float] = 0., std: Union[int, float] = 1.0, size: Union[Tuple[int, int], None] = None) -> Union[float, Tensor]:
        if size is None:
            return np.random.normal(mean, std, size)
        else:
            return Tensor(np.random.normal(mean, std, size))

# Trigonometry Functions

def cos(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(np.cos(X.tensor), parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad -= np.sin(X.tensor) * Y.grad

            Y._backward = _backward

        return Y
    elif isinstance(X, np.ndarray):
        return np.cos(X)
    else:
        return NotImplemented

def sin(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(np.sin(X.tensor), parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad += np.cos(X.tensor) * Y.grad

            Y._backward = _backward

        return Y
    elif isinstance(X, np.ndarray):
        return np.sin(X)
    else:
        return NotImplemented

def arctan(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(np.arctan(X.tensor), parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad += 1 / (1 + X.tensor**2) * Y.grad
            
            Y._backward = _backward

        return Y
    elif isinstance(X, np.ndarray):
        return np.arctan(X)
    else:
        return NotImplemented
    
# Exponentiel, logarithm

def square(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(np.square(X.tensor), parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad += 2 * X.tensor * Y.grad
            
            Y._backward = _backward
            
        return Y
    elif isinstance(X, np.ndarray):
        return np.square(X)
    else:
        return NotImplemented
    
def exp(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(np.exp(X.tensor), parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad += np.exp(X.tensor) * Y.grad
            
            Y._backward = _backward
            
        return Y
    elif isinstance(X, np.ndarray):
        return np.exp(X)
    else:
        return NotImplemented
    
def log(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(np.log(X.tensor), parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                if X.tensor.all() != 0:
                    X.grad += (1 / X.tensor) * Y.grad
                else:
                    return NotImplemented
            
            Y._backward = _backward
            
        return Y
    elif isinstance(X, np.ndarray):
        return np.log(X)
    else:
        return NotImplemented

# Hyperbolic Functions

def tanh(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(np.tanh(X.tensor), parents=[X], requires_grad=X.requires_grad)
        if Y.requires_grad:
            def _backward():
                X.grad += (1 - Y.tensor**2) * Y.grad
            Y._backward = _backward

        return Y
    elif isinstance(X, np.ndarray):
        return np.tanh(X)
    else:
        return NotImplemented

# Activation Functions

def identity(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(X.tensor, parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad += np.ones_like(X.tensor) * Y.grad
            Y._backward = _backward
            
        return Y 
    elif isinstance(X, np.ndarray):
        return X
    else:
        return NotImplemented

def heaviside(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(np.heaviside(X.tensor, 1 / 2), parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad +=  np.zeros_like(X.grad)
            
            Y._backward = _backward

        return Y
    elif isinstance(X, np.ndarray):
        return np.heaviside(X, 1 / 2)
    else:
        return NotImplemented
    
def sigmoid(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        #sig = np.where(X.tensor > 0, 1 / (1 + np.exp(-X.tensor)), np.exp(X.tensor) / (np.exp(X.tensor) + 1)).clip(1e-7, 1-1e-7)
        sig = np.empty_like(X.tensor)
        mask = X.tensor > 0

        sig[mask] = 1 / (1 + np.exp(-X.tensor[mask]))
        sig[~mask] = np.exp(X.tensor[~mask]) / (np.exp(X.tensor[~mask]) + 1)
        sig = sig.clip(1e-7, 1-1e-7)
        Y = Tensor(sig, parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad += sig * (1 - sig) * Y.grad
            
            Y._backward = _backward

        return Y
    elif isinstance(X, np.ndarray):
        return (1 / (1 + np.exp(-X)))
    else:
        return NotImplemented
    
def relu(X: Union[Tensor, np.ndarray]):
    if isinstance(X, Tensor):
        Y = Tensor(X.tensor * (X.tensor > 0), parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad += 1. * (X.tensor > 0) * Y.grad
            
            Y._backward = _backward

        return Y
    elif isinstance(X, np.ndarray):
        return X * (X > 0)
    else:
        return NotImplemented

def softmax(X: Union[Tensor, np.ndarray], dim: Union[int, None] = None):
    def _dimension(dim: Union[int, None] = None):
        if dim is None:
            dim = -1
        return dim
    
    if isinstance(X, Tensor):
        dim = _dimension(dim)
        y = np.zeros_like(X.tensor)
        for index in np.ndindex(*(X.shape[:dim] + X.shape[dim+1:])):
            full_index = index[:dim] + (slice(None),) + index[dim:]
            x = X.tensor[full_index]

            x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)

            shifted_x = x - np.max(x)
            exp_x = np.exp(shifted_x)
            softmax = exp_x / np.maximum(exp_x.sum(), 1e-100)

            y[full_index] = softmax
        y = y.clip(1e-7, 1-1e-7)
        Y = Tensor(y, parents=[X], requires_grad=X.requires_grad)

        if Y.requires_grad:
            def _backward():
                X.grad += y * (1 - y) * Y.grad
            
            Y._backward = _backward
        
        return Y
    else:
        return NotImplemented

def broadcasting_backward(A: np.ndarray, B: np.ndarray):
    indices = None
    sup = None

    if len(A.shape) < len(B.shape):
        sup = False
    elif len(A.shape) > len(B.shape):
        sup = True
    else:
        return sup, indices
    
    A_size = len(A.shape)
    B_size = len(B.shape)
    if A_size != 0 and B_size != 0:
        for i, j in zip(range(len(A.shape)), range(len(B.shape))):
            if i == A_size - 1:
                indices = [(k, B.shape[k], True) for k in range(j+1)]
                break
            if j == B_size - 1:
                indices = [(k, A.shape[k], True) for k in range(i+1)]
                break
    elif A_size ==0:
        return sup, [(k, B.shape[k], False) for k in range(B_size)]
    else:
        return sup, [(k, A.shape[k], False) for k in range(A_size)]

    if indices is not None:
        indices.extend([(k, A.shape[k], False) for k in range(i+1, len(A.shape))] if sup else [(k, B.shape[k], False) for k in range(j+1, len(B.shape))])
    return sup, indices


def elementary_operation_broadcasting_backward(A: np.ndarray, B: np.ndarray):
    sup, indices = broadcasting_backward(A, B)
    new = None
    if indices is None:
        for enu, (i, j) in enumerate(zip(A.shape, B.shape)):
            if i < j:
                if new is None:
                    new = np.delete(B, obj=[k for k in range(j - 1)], axis=enu)
                else:
                    new = np.delete(new, obj=[k for k in range(j - 1)], axis=enu)
        return new
    else:
        idx_kd = 0
        for index, dim_size, keepdims in indices:
            if sup:
                if new is None:
                    new = np.delete(A, [i for i in range(dim_size - 1)], axis=index-idx_kd)
                else:
                    new = np.delete(new, [i for i in range(dim_size - 1)], axis=index-idx_kd)
            else:
                if new is None:
                    new = np.delete(B, [i for i in range(dim_size - 1)], axis=index-idx_kd)
                else:
                    new = np.delete(new, [i for i in range(dim_size - 1)], axis=index-idx_kd)

            if not keepdims:
                new = new.squeeze(axis=index - idx_kd)
                idx_kd += 1
        return new