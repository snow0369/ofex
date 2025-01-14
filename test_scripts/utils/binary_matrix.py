import galois

gf = galois.GF(2)
mat_a = gf(
    [
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ]
)

print("=== NULL ===")
ns = mat_a.T.null_space()
print(ns)
print(type(ns))
