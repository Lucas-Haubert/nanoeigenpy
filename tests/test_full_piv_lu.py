import nanoeigenpy
import numpy as np

dim = 100
seed = 1
rng = np.random.default_rng(seed)

# Test nb::init<const MatrixType &>()
A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

fullpivlu = nanoeigenpy.FullPivLU(A)

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = fullpivlu.solve(B)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = fullpivlu.solve(b)
assert nanoeigenpy.is_approx(x, x_est)
assert nanoeigenpy.is_approx(A.dot(x_est), b)

# Others
rows = fullpivlu.rows()
cols = fullpivlu.cols()
assert cols == dim
assert rows == dim

fullpivlu_compute = fullpivlu.compute(A)

A_reconstructed = fullpivlu.reconstructedMatrix()
assert nanoeigenpy.is_approx(A_reconstructed, A)

nonzeropivots = fullpivlu.nonzeroPivots()
maxpivot = fullpivlu.maxPivot()
kernel = fullpivlu.kernel()
image = fullpivlu.image(A)
rcond = fullpivlu.rcond()
determinant = fullpivlu.determinant()
rank = fullpivlu.rank()
dimkernel = fullpivlu.dimensionOfKernel()
injective = fullpivlu.isInjective()
surjective = fullpivlu.isSurjective()
invertible = fullpivlu.isInvertible()
inverse = fullpivlu.inverse()
reconstructedmatrix = fullpivlu.reconstructedMatrix()

fullpivlu.setThreshold(1e-8)
assert fullpivlu.threshold() == 1e-8

ldlt1 = nanoeigenpy.LDLT()
ldlt2 = nanoeigenpy.LDLT()

id1 = ldlt1.id()
id2 = ldlt2.id()

assert id1 != id2
assert id1 == ldlt1.id()
assert id2 == ldlt2.id()

dim_constructor = 3

ldlt3 = nanoeigenpy.LDLT(dim_constructor)
ldlt4 = nanoeigenpy.LDLT(dim_constructor)

id3 = ldlt3.id()
id4 = ldlt4.id()

assert id3 != id4
assert id3 == ldlt3.id()
assert id4 == ldlt4.id()
