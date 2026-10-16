"""Survey populations and sample sizes.

Sample size per population for 95% confidence, 5% margin of error and an
80/20 split (p = 0.8), with the finite population correction:

    n0 = z^2 p (1 - p) / e^2        n = n0 / (1 + (n0 - 1) / N)

Inputs are the contact lists of newcomers and of reviewers of AI PRs, which
are not distributed because they contain e-mail addresses.
"""

import math

import pandas as pd

Z, P, ERROR = 1.96, 0.8, 0.05
OUTPUT = "results/survey_population.csv"


def sample_size(n_pop: int) -> int:
    if n_pop == 0:
        return 0
    n0 = Z * Z * P * (1 - P) / ERROR**2
    return math.ceil(n0 / (1 + (n0 - 1) / n_pop))


newc = pd.read_csv("novatos_unicos.csv", keep_default_na=False)
newc_email = newc["Email_Perfil"] != ""
maint = pd.read_csv("mantenedores_ia_unicos.csv", keep_default_na=False)
maint_email = maint["Revisor_Email"] != ""

groups = {
    "Newcomers (all)": (pd.Series(True, index=newc.index), newc_email),
    # survey groups: AI in the debut vs. no AI PR at all; newcomers who used AI
    # only after the debut are left out (disjoint groups)
    "Newcomers with AI in the debut": (newc["IA_Estreia"], newc_email),
    "Newcomers without any AI PR": (newc["Qtde_PRs_IA"] == 0, newc_email),
    "Newcomers excluded (AI only after the debut)": (
        ~newc["IA_Estreia"] & (newc["Qtde_PRs_IA"] > 0),
        newc_email,
    ),
    "Maintainers (reviewed an AI PR)": (pd.Series(True, index=maint.index), maint_email),
}

rows = []
for name, (mask, email) in groups.items():
    n_pop = int(mask.sum())
    rows.append(
        {
            "group": name,
            "N": n_pop,
            "N_with_email": int((mask & email).sum()),
            "n_95_5": sample_size(n_pop),
        }
    )

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(df.to_string(index=False))
