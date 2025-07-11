import matplotlib.pyplot as plt

pseudo_code = """
Function BO-Minimization (x0)
    Initialize bounds using test_x
    Define constraints:
        constraintSafety(x) -> ensures safety
        constraintUnsafe(x) -> checks unsafe regions
        constraintExpander(x) -> expands safe set
    If constraintSafety(x0) < 0, print "Safeness violated!"
    If constraintUnsafe(x0) < 0, print "Unsafeness violated!"
    If constraintExpander(x0) < 0, print "Expander violated!"

    Generate candidates using Scipy optimization:
        min_x, min_val = gen_candidates_scipy(
            x0, objective, bounds, constraints
        )
    Return min_x, min_val
End Function
"""

fig, ax = plt.subplots(figsize=(8, 4))
ax.text(0, 0.5, pseudo_code, fontsize=12, family="monospace", va="center", ha="left")
ax.axis("off")
plt.title("Pseudo-code for Safe Optimization")
plt.show()

print('Done')