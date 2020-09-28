import nnetsauce as ns


fit_obj = ns.HypTan(0)

print("tanh(0)")
print(fit_obj.calculate())
print("\n")

fit_obj2 = ns.HypTan(5)

print("tanh(5)")
print(fit_obj2.calculate())
print("\n")

fit_obj3 = ns.HypTan(10)

print("tanh(10)")
print(fit_obj3.calculate())
print("\n")

fit_obj4 = ns.HypTan(0.5)

print("tanh(0.5)")
print(fit_obj4.calculate())
print("\n")

fit_obj5 = ns.HypTan(-0.5)

print("tanh(-0.5)")
print(fit_obj5.calculate())
print("\n")

