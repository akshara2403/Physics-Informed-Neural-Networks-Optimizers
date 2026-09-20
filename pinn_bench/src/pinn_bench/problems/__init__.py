from . import heat, burgers, allen_cahn, kdv

PROBLEMS = {
    heat.NAME: heat,
    burgers.NAME: burgers,
    allen_cahn.NAME: allen_cahn,
    kdv.NAME: kdv,
}

# The three benchmarks that appear in the original roadmap. KdV is a novel
# addition from this project and is deliberately excluded here so any
# "roadmap vs. novel" comparison in reports stays honest.
ROADMAP_PROBLEMS = (heat.NAME, burgers.NAME, allen_cahn.NAME)
NOVEL_PROBLEMS = (kdv.NAME,)
