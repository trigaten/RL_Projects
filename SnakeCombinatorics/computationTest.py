import Combinatorics as comb 

print("Possible states at a given Length, L")
print("L | Theoretical Upper Bound | Actual")
for i in range(1, 10):
    print(str(i) + " : " + str("{:e}".format(comb.computeUpperStateBound(i))) + " | " + str(comb.computeStatesFromSize(i)))


# Possible states at a given Length, L
# L | Upper Bound | Actual
# 1 : 1 | 1
# 2 : 236 | 44
# 3 : 137700 | 2080
# 4 : 535692272 | 143920
# 5 : 16475056294100 | 22073136
