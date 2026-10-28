"""Survey results (RQ3).

Reads the three Google Forms exports and writes to results_survey/:
  survey_response_rate.csv           invitations x answers per audience
                                     (Newcomer row = both newcomer forms)
  survey_closed_answers.csv          frequency of each closed question, per
                                     form and for both newcomer forms together
  survey_likert_maintainers.csv      1..5 distribution, median and % agree per item
  survey_crosstabs.csv               crosstabs with "does AI undermine belonging?"
                                     and with willingness
  survey_highlights.csv              highlighted 2x2 crosstabs with Fisher's
                                     exact test (exploratory)
  survey_likert_per_belonging.csv    maintainers' Likert items, Yes vs. No
                                     (Mann-Whitney, Cliff's delta)
  survey_open_answers.csv            all open answers with their codes
  survey_themes.csv                  theme counts per audience and belonging answer
  survey_codebook.csv                codebook: category, definition, inclusion and
                                     exclusion criteria, example excerpts, counts
                                     and participant IDs (N1.., M1..) per theme

The open answers (WILLINGNESS, THEMES) were coded manually; the
codes are fixed here, keyed by the form timestamp.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

OUT_DIR = "results_survey"
os.makedirs(OUT_DIR, exist_ok=True)

FORMS = {
    "Newcomer_AI": "Form Contribuidores (Com IA).csv",
    "Newcomer_no_AI": "Form Contribuidores (Sem IA).csv",
    "Maintainer": "Form Mantenedores.csv",
}
# group names used in the (private) invitation log
LOG_GROUPS = {
    "Novato_IA": "Newcomer_AI",
    "Novato_sem_IA": "Newcomer_no_AI",
    "Mantenedor": "Maintainer",
}

# Short column names, in the order of the export.
COLS_NEWCOMER = [
    "ts",
    "consent",
    "projects",
    "motivation",
    "aspect",
    "belonging_factor",
    "undermines_belonging",
    "undermines_explanation",
    "used_ai",
    "how_used",
    "willingness",
    "noticed_ai",
]
COLS_MAINT = [
    "ts",
    "consent",
    "L_quality",
    "L_less_effort",
    "L_more_attention",
    "L_more_prs",
    "L_less_communication",
    "L_lacks_context",
    "L_more_satisfying",
    "L_trusts_less",
    "L_more_workload",
    "aspect",
    "belonging_factor",
    "undermines_belonging",
    "undermines_explanation",
]

MOTIVATION = {
    "I contribute to fix or improve problems I run into as a user": "Fix a problem I face as a user",
    "I believe in the open source philosophy / it's the more ethical way to build software": "Open source philosophy",
    "I want to learn new things and improve my skills": "Learn",
    "I use this software, so I feel it's my duty to give back": "Give back (I use the software)",
    "I enjoy coding. It's fun / a hobby for me": "Fun / hobby",
    "I want to build a portfolio / strengthen my CV for hiring": "Portfolio / CV",
    "I'm paid or receive a stipend to contribute": "Paid to contribute",
    "I want to support other contributors and share knowledge with the community": "Support others / share knowledge",
    "I like collaborating with this team and being part of an inclusive community": "Collaborate / inclusive community",
}
FACTOR = {
    "Having my contributions recognized and valued by the community": "Recognition",
    "Getting support from other contributors/maintainers when I need help": "Support from others",
    "Feeling like I'm a member of this project": "Feeling like a member",
    "Feeling as if this project is home to me": "Project as home",
    "Knowing who to ask for help, and being known by others in the project": "Knowing who to ask / being known",
}
ASPECT = {
    "Efficiency and flow (cognitive load reviewing generated code)": "Efficiency and flow",
    "Communication and collaboration": "Communication and collaboration",
    "Satisfaction and well-being": "Satisfaction and well-being",
    "No noticeable change": "No noticeable change",
}

# Answers excluded from the analysis (clearly invalid or without consent).
INVALID = {
    ("Newcomer_no_AI", "2026/10/02 8:44:28 PM GMT-3"),
    ("Newcomer_AI", "2026/10/05 11:47:42 AM GMT-3"),
    ("Newcomer_AI", "2026/10/14 12:03:17 PM GMT-3"),
}

# Project field reduced to the project name: these answers cited the
# respondent's own PR.
PROJECTS_REDACTED = {
    ("Newcomer_AI", "2026/09/30 12:11:27 PM GMT-3"): "MLflow, Apache Superset",
    ("Newcomer_AI", "2026/10/01 8:20:34 AM GMT-3"): "github-readme-stats",
}

# Open answers that name the respondent's employer or own project. The text to replace would
# identify the respondent, so the list stays in a local file that is not
# distributed (survey_redactions.json: group, ts, column, text, replacement).
REDACTIONS_FILE = "survey_redactions.json"
TEXT_REDACTED = {}
if os.path.exists(REDACTIONS_FILE):
    with open(REDACTIONS_FILE, encoding="utf-8") as f:
        TEXT_REDACTED = {
            (r["group"], r["ts"], r["column"]): (r["text"], r["replacement"]) for r in json.load(f)
        }

# Effect of AI use on the willingness to contribute (question 10), only for
# respondents who used AI: increased | no_effect | ambiguous.
WILLINGNESS = {
    ("Newcomer_AI", "2026/09/30 9:41:09 AM GMT-3"): "increased",
    ("Newcomer_AI", "2026/09/30 12:11:27 PM GMT-3"): "no_effect",
    ("Newcomer_AI", "2026/09/30 12:25:40 PM GMT-3"): "no_effect",
    ("Newcomer_AI", "2026/10/01 4:18:04 AM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/01 8:20:34 AM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/01 10:35:05 AM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/01 10:40:06 AM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/01 12:34:27 PM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/02 2:32:04 AM GMT-3"): "no_effect",
    ("Newcomer_AI", "2026/10/03 11:44:50 PM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/05 3:33:04 PM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/06 1:21:05 PM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/06 3:45:55 PM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/07 12:13:53 PM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/13 6:40:55 PM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/14 1:04:38 PM GMT-3"): "no_effect",
    ("Newcomer_AI", "2026/10/14 2:57:23 PM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/14 4:58:28 PM GMT-3"): "no_effect",
    ("Newcomer_AI", "2026/10/15 1:24:57 AM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/15 5:24:41 AM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/15 9:58:08 AM GMT-3"): "increased",
    ("Newcomer_AI", "2026/10/15 12:23:55 PM GMT-3"): "increased",
    ("Newcomer_no_AI", "2026/10/01 10:36:35 AM GMT-3"): "increased",
    ("Newcomer_no_AI", "2026/10/07 4:48:06 AM GMT-3"): "no_effect",
    ("Newcomer_no_AI", "2026/10/07 12:58:52 PM GMT-3"): "increased",
    ("Newcomer_no_AI", "2026/10/12 6:11:41 AM GMT-3"): "no_effect",
    ("Newcomer_no_AI", "2026/10/14 4:44:37 PM GMT-3"): "no_effect",
    ("Newcomer_no_AI", "2026/10/15 12:14:09 PM GMT-3"): "increased",
    ("Newcomer_no_AI", "2026/10/15 4:58:27 PM GMT-3"): "no_effect",
    ("Newcomer_no_AI", "2026/10/16 3:29:12 AM GMT-3"): "increased",
}

# Codebook of the open answers (belonging explanation, AI use and willingness).
# Coding is multi-label. Each theme has a category, a definition, inclusion and
# exclusion criteria, the themes it is easily confused with ("close but no")
# and example excerpts (participant ID, verbatim text after redaction), written
# to survey_codebook.csv.
CODEBOOK = {
    "lowers_barrier": dict(
        category="AI as an enabler",
        definition="without AI the person would not have contributed, or AI made it much "
        "easier to start (skill, time or effort)",
        include="the person says they would not have had the skill, time or patience to "
        "contribute without AI, or that AI lowered the effort to start",
        exclude="AI helped to understand the code or the workflow, without saying the "
        "contribution would not have happened (onboarding)",
        close="onboarding; opens_community",
        examples=[
            ("N11", "without AI I wouldn't be technically able to do any of these things"),
            ("N30", "I wouldn't have the time otherwise."),
            ("M31", "the bar for contribution is lower now, but the bar for trust is higher"),
        ],
    ),
    "onboarding": dict(
        category="AI as an enabler",
        definition="AI helped to understand the code and the project workflow",
        include="AI was used to navigate the repository, understand the codebase, its "
        "conventions or the contribution workflow",
        exclude="AI only wrote or reviewed the code, with no mention of understanding the "
        "project (tool)",
        close="lowers_barrier; tool",
        examples=[
            ("N5", "it helped me understand the project, its workflow, and the codebase more quickly"),
            ("N35", "AI was used to navigate the repo, understand its culture and conventions"),
            ("N37", "it just made the onboarding process easier"),
        ],
    ),
    "opens_community": dict(
        category="AI as an enabler",
        definition="AI widens the community (users become developers)",
        include="AI lets people who could not contribute before, such as users or "
        "beginners, join the developer community",
        exclude="the effect is only on the respondent's own contribution (lowers_barrier)",
        close="lowers_barrier",
        examples=[
            ("N9", "using AI could enhance the community by expanding the community, by "
            "allowing the community of users to move into the community of developers"),
            ("N38", "make open-source projects more accessible to beginners"),
        ],
    ),
    "teammate": dict(
        category="AI as an enabler",
        definition="AI as a collaborator or peer, not as a tool",
        include="AI is described as a collaborator, peer or member of the team",
        exclude="AI is described as a tool under human control (tool)",
        close="tool",
        examples=[
            ("N28", "AI agents are equal collaborators no different from people"),
        ],
    ),
    "tool": dict(
        category="AI as a tool",
        definition="AI is just a tool; the human stays responsible",
        include="AI is called a tool like any other, and the human keeps the "
        "responsibility for the contribution",
        exclude="the answer makes the effect of AI depend on how it is used or reviewed "
        "(human_loop)",
        close="human_loop; teammate",
        examples=[
            ("N10", "LLMs are just tools"),
            ("N23", "AI is just a tool; what matters is quality and review, not who or what wrote the code"),
            ("M20", "I think AI agents are just tools rather than things in our community"),
        ],
    ),
    "human_loop": dict(
        category="AI as a tool",
        definition="depends on how it is used / on the human reviewing it",
        include="the effect of AI depends on how it is used, on a human reviewing and "
        "understanding the output, or on the autonomy given to the agent",
        exclude="AI is called a neutral tool with no condition attached (tool)",
        close="tool; slop",
        examples=[
            ("N16", "If it still goes through the course of a maintainer or contributor, receives "
            "critique and is addressed properly then I believe it's healthy AI usage."),
            ("M16", "When used properly with a Human in the loop, it's not any different from human-only PRs."),
            ("M22", "when it's used by people who do not understand what they're doing and wouldn't "
            "be able to do it by themselves, it's usually a net negative"),
        ],
    ),
    "slop": dict(
        category="AI as a threat to the community",
        definition="low-effort contributions, noise, credit seeking",
        include="low-effort or low-value AI contributions, noise, volume that overloads "
        "review, or contributors seeking credit",
        exclude="the concern is the loss of human contact or learning, not the quality "
        "or volume of contributions (interaction)",
        close="interaction; human_loop",
        examples=[
            ("N3", "AI enables low effort submissions which introduces noise into the system."),
            ("N39", "A lot of low value PRs and issues are being opened"),
            ("M18", "AI enables people who seem to just want credit / name recognition."),
        ],
    ),
    "interaction": dict(
        category="AI as a threat to the community",
        definition="loss of interaction, learning or human contact",
        include="AI reduces communication between people, the learning that comes from "
        "it, or the human connection in reviews",
        exclude="the concern is the quality or volume of AI contributions (slop) or "
        "the respondent's ownership of their own work (detachment)",
        close="slop; detachment",
        examples=[
            ("N26", "You can learn from it, but its a very different feeling than learning from "
            "other people working on the project."),
            ("M13", "If the day is filled with only good AI PR reviews, it can be really draining and lonely."),
            ("M46", "It removes the processes that cause contributors to develop expertise"),
        ],
    ),
    "detachment": dict(
        category="AI as a threat to the community",
        definition="AI weakens the sense of ownership of one's own work",
        include="writing code with AI reduces the respondent's sense of ownership of "
        "their own contribution",
        exclude="the loss concerns contact with other people (interaction)",
        close="interaction",
        examples=[
            ("M42", "With AI you have very little sense of ownership."),
        ],
    ),
    "impostor_syndrome": dict(
        category="Contributor confidence",
        definition="mentions impostor syndrome or lack of confidence",
        include="the respondent mentions impostor syndrome or a lack of confidence in "
        "their own ability, with or without AI",
        exclude="AI only saved time or effort, with no mention of confidence (lowers_barrier)",
        close="lowers_barrier",
        examples=[
            ("N4", "I sometimes struggle with impostor syndrome"),
            ("N35", "the assurance of correctness and validation from AI agent certainly removed "
            "the imposter syndrome"),
        ],
    ),
}
THEMES = {
    ("Newcomer_AI", "2026/09/30 9:41:09 AM GMT-3"): ["lowers_barrier", "opens_community"],
    ("Newcomer_AI", "2026/09/30 12:11:27 PM GMT-3"): ["tool", "onboarding"],
    ("Newcomer_AI", "2026/09/30 12:25:40 PM GMT-3"): ["slop"],
    ("Newcomer_AI", "2026/10/01 4:18:04 AM GMT-3"): ["impostor_syndrome"],
    ("Newcomer_AI", "2026/10/01 8:20:34 AM GMT-3"): ["onboarding"],
    ("Newcomer_AI", "2026/10/01 10:40:06 AM GMT-3"): ["tool"],
    ("Newcomer_AI", "2026/10/01 12:34:27 PM GMT-3"): [
        "lowers_barrier",
        "opens_community",
        "human_loop",
        "interaction",
    ],
    ("Newcomer_AI", "2026/10/02 2:32:04 AM GMT-3"): ["tool"],
    ("Newcomer_AI", "2026/10/03 11:44:50 PM GMT-3"): ["lowers_barrier", "tool"],
    ("Newcomer_AI", "2026/10/05 11:47:42 AM GMT-3"): ["tool"],
    ("Newcomer_AI", "2026/10/05 3:33:04 PM GMT-3"): ["lowers_barrier"],
    ("Newcomer_AI", "2026/10/06 1:21:05 PM GMT-3"): ["slop"],
    ("Newcomer_AI", "2026/10/06 3:45:55 PM GMT-3"): ["human_loop", "lowers_barrier", "onboarding"],
    ("Newcomer_AI", "2026/10/07 12:13:53 PM GMT-3"): ["tool"],
    ("Newcomer_AI", "2026/10/07 8:01:20 PM GMT-3"): ["interaction"],
    ("Newcomer_AI", "2026/10/13 6:40:55 PM GMT-3"): ["tool", "onboarding"],
    ("Newcomer_AI", "2026/10/14 1:04:38 PM GMT-3"): ["tool"],
    ("Newcomer_AI", "2026/10/14 2:57:23 PM GMT-3"): ["human_loop", "onboarding"],
    ("Newcomer_AI", "2026/10/14 4:58:28 PM GMT-3"): ["teammate"],
    ("Newcomer_AI", "2026/10/15 1:24:57 AM GMT-3"): ["lowers_barrier"],
    ("Newcomer_AI", "2026/10/15 5:24:41 AM GMT-3"): ["lowers_barrier", "onboarding"],
    ("Newcomer_AI", "2026/10/15 9:58:08 AM GMT-3"): ["tool", "human_loop"],
    ("Newcomer_AI", "2026/10/15 12:23:55 PM GMT-3"): ["tool"],
    ("Newcomer_AI", "2026/10/21 6:30:17 AM GMT-3"): ["human_loop", "lowers_barrier"],
    ("Newcomer_AI", "2026/10/21 10:11:26 AM GMT-3"): ["human_loop"],
    ("Newcomer_no_AI", "2026/10/01 10:36:35 AM GMT-3"): ["interaction"],
    ("Newcomer_no_AI", "2026/10/07 4:48:06 AM GMT-3"): ["tool", "slop"],
    ("Newcomer_no_AI", "2026/10/07 12:58:52 PM GMT-3"): ["human_loop", "slop"],
    ("Newcomer_no_AI", "2026/10/12 6:11:41 AM GMT-3"): ["human_loop"],
    ("Newcomer_no_AI", "2026/10/14 3:45:47 PM GMT-3"): ["interaction", "human_loop", "slop"],
    ("Newcomer_no_AI", "2026/10/14 4:44:37 PM GMT-3"): ["tool", "human_loop", "onboarding"],
    ("Newcomer_no_AI", "2026/10/15 12:13:12 PM GMT-3"): ["tool"],
    ("Newcomer_no_AI", "2026/10/15 12:14:09 PM GMT-3"): ["interaction", "onboarding", "impostor_syndrome"],
    ("Newcomer_no_AI", "2026/10/15 4:58:27 PM GMT-3"): ["human_loop", "slop", "onboarding"],
    ("Newcomer_no_AI", "2026/10/16 3:29:12 AM GMT-3"): ["tool", "opens_community", "human_loop", "onboarding"],
    ("Newcomer_no_AI", "2026/10/19 5:05:19 PM GMT-3"): ["slop"],
    ("Maintainer", "2026/09/30 10:29:04 AM GMT-3"): ["tool"],
    ("Maintainer", "2026/09/30 11:04:19 AM GMT-3"): ["human_loop"],
    ("Maintainer", "2026/09/30 11:32:43 AM GMT-3"): ["interaction"],
    ("Maintainer", "2026/09/30 12:12:38 PM GMT-3"): ["interaction"],
    ("Maintainer", "2026/09/30 4:08:37 PM GMT-3"): ["tool"],
    ("Maintainer", "2026/10/01 8:00:32 PM GMT-3"): ["slop", "interaction"],
    ("Maintainer", "2026/10/05 10:51:20 AM GMT-3"): ["interaction"],
    ("Maintainer", "2026/10/05 2:27:04 PM GMT-3"): ["slop"],
    ("Maintainer", "2026/10/05 4:51:24 PM GMT-3"): ["human_loop"],
    ("Maintainer", "2026/10/06 9:57:28 AM GMT-3"): ["slop"],
    ("Maintainer", "2026/10/06 10:52:12 AM GMT-3"): ["interaction"],
    ("Maintainer", "2026/10/06 11:11:59 AM GMT-3"): ["tool"],
    ("Maintainer", "2026/10/07 8:36:53 AM GMT-3"): ["human_loop"],
    ("Maintainer", "2026/10/09 8:58:51 AM GMT-3"): ["tool", "human_loop"],
    ("Maintainer", "2026/10/12 1:20:37 PM GMT-3"): ["slop"],
    ("Maintainer", "2026/10/14 9:00:36 AM GMT-3"): ["interaction"],
    ("Maintainer", "2026/10/14 11:33:17 AM GMT-3"): ["tool"],
    ("Maintainer", "2026/10/14 11:50:40 AM GMT-3"): ["slop"],
    ("Maintainer", "2026/10/14 12:06:00 PM GMT-3"): ["lowers_barrier", "human_loop"],
    ("Maintainer", "2026/10/15 9:51:13 AM GMT-3"): ["interaction", "human_loop"],
    ("Maintainer", "2026/10/15 11:09:28 AM GMT-3"): ["slop", "interaction", "lowers_barrier"],
    ("Maintainer", "2026/10/15 11:18:01 AM GMT-3"): ["interaction"],
    ("Maintainer", "2026/10/15 11:48:26 AM GMT-3"): ["slop", "interaction"],
    ("Maintainer", "2026/10/15 12:16:12 PM GMT-3"): ["slop"],
    ("Maintainer", "2026/10/15 1:02:04 PM GMT-3"): ["detachment"],
    ("Maintainer", "2026/10/15 10:37:42 PM GMT-3"): ["tool"],
    ("Maintainer", "2026/10/16 9:44:49 AM GMT-3"): ["interaction"],
}

# Maintainers who answer No (AI does not undermine belonging) but explain it
# with a concern.
NO_WITH_CAVEAT = {
    "2026/09/30 11:32:43 AM GMT-3",
    "2026/10/05 10:51:20 AM GMT-3",
    "2026/10/05 4:51:24 PM GMT-3",
    "2026/10/06 10:52:12 AM GMT-3",
    "2026/10/07 8:36:53 AM GMT-3",
    "2026/10/14 9:00:36 AM GMT-3",
    "2026/10/14 12:06:00 PM GMT-3",
    "2026/10/15 11:18:01 AM GMT-3",
    "2026/10/15 12:16:12 PM GMT-3",
}


def load():
    blocks = []
    for group, file in FORMS.items():
        d = pd.read_csv(file)
        d.columns = COLS_MAINT if group == "Maintainer" else COLS_NEWCOMER
        d.insert(0, "group", group)
        blocks.append(d)
    df = pd.concat(blocks, ignore_index=True)
    df["invalid"] = [(g, t) in INVALID for g, t in zip(df.group, df.ts)]
    df["motivation"] = df.motivation.map(MOTIVATION)
    df["belonging_factor"] = df.belonging_factor.map(FACTOR)
    df["aspect"] = df.aspect.map(ASPECT)
    df["newcomer"] = df.group != "Maintainer"
    df["willingness_code"] = [WILLINGNESS.get((g, t)) for g, t in zip(df.group, df.ts)]
    df["themes"] = [";".join(THEMES.get((g, t), [])) for g, t in zip(df.group, df.ts)]
    # Participant IDs used in the paper: N1.. for newcomers (both forms) and M1..
    # for maintainers, valid answers only, in order of answer time.
    df["pid"] = ""
    when = pd.to_datetime(df.ts.str.replace(" GMT-3", ""), format="%Y/%m/%d %I:%M:%S %p")
    for prefix, sel in [("N", df.newcomer), ("M", ~df.newcomer)]:
        idx = when[sel & ~df.invalid].sort_values().index
        df.loc[idx, "pid"] = [f"{prefix}{i}" for i in range(1, len(idx) + 1)]
    return df


def response_rate(df):
    log = pd.read_csv("survey_envios/envios_log.csv")
    env = (
        log[log.Status == "enviado"]
        .drop_duplicates(["Onda", "Login"])
        .groupby("Grupo")
        .size()
        .rename(LOG_GROUPS)
    )
    resp = df.groupby("group").size()
    val = df[~df.invalid].groupby("group").size()
    t = (
        pd.DataFrame({"invitations": env, "answers": resp, "valid_answers": val})
        .fillna(0)
        .astype(int)
    )
    # Newcomer = both newcomer forms together, the group reported in the paper.
    newc = t.loc[[g for g in t.index if g.startswith("Newcomer_")]].sum()
    t.loc["Total"] = t.sum()
    t.loc["Newcomer"] = newc
    t["rate_%"] = (100 * t.answers / t.invitations).round(1)
    t.index.name = "group"
    t.to_csv(f"{OUT_DIR}/survey_response_rate.csv")
    return t


def closed(df):
    questions = [
        "motivation",
        "aspect",
        "belonging_factor",
        "undermines_belonging",
        "used_ai",
        "willingness_code",
    ]
    rows = []
    v = df[~df.invalid]
    for group, g in list(v.groupby("group")) + [("Newcomer", v[v.newcomer])]:
        for p in questions:
            s = g[p].dropna()
            for v, n in s.value_counts().items():
                rows.append(
                    {
                        "group": group,
                        "question": p,
                        "answer": v,
                        "n": n,
                        "total": len(s),
                        "pct": round(100 * n / len(s), 1),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/survey_closed_answers.csv", index=False)
    return out


def likert(df):
    m = df[(df.group == "Maintainer") & ~df.invalid]
    rows = []
    for c in [c for c in m.columns if c.startswith("L_")]:
        s = m[c].dropna().astype(int)
        d = {f"n{k}": int((s == k).sum()) for k in range(1, 6)}
        rows.append(
            {
                "item": c,
                "n": len(s),
                **d,
                "median": s.median(),
                "pct_disagree_1_2": round(100 * (s <= 2).mean(), 1),
                "pct_neutral_3": round(100 * (s == 3).mean(), 1),
                "pct_agree_4_5": round(100 * (s >= 4).mean(), 1),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/survey_likert_maintainers.csv", index=False)
    return out


def crosstabs(df):
    v = df[~df.invalid]
    newc = v[v.newcomer]
    used = newc[newc.used_ai == "Yes"]
    tables = [
        ("newcomers", "motivation", "undermines_belonging", newc),
        ("newcomers", "belonging_factor", "undermines_belonging", newc),
        ("maintainers", "belonging_factor", "undermines_belonging", v[~v.newcomer]),
        ("newcomers", "used_ai", "undermines_belonging", newc),
        ("newcomers", "group", "undermines_belonging", newc),
        ("newcomers", "aspect", "undermines_belonging", newc),
        ("maintainers", "aspect", "undermines_belonging", v[~v.newcomer]),
        ("newcomers_who_used_ai", "motivation", "willingness_code", used),
        ("newcomers_who_used_ai", "belonging_factor", "willingness_code", used),
        ("newcomers_who_used_ai", "willingness_code", "undermines_belonging", used),
    ]
    rows = []
    for audience, lin, col, d in tables:
        ct = pd.crosstab(d[lin], d[col])
        for a in ct.index:
            for b in ct.columns:
                rows.append(
                    {
                        "audience": audience,
                        "row": lin,
                        "column": col,
                        "row_value": a,
                        "column_value": b,
                        "n": int(ct.loc[a, b]),
                        "row_total": int(ct.loc[a].sum()),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/survey_crosstabs.csv", index=False)
    return out


def highlights(df):
    """Highlighted 2x2 crosstabs. Exploratory: the subsets were chosen after
    looking at the data and n is small, so Fisher's test is descriptive only.
    """
    v = df[~df.invalid]
    newc = v[v.newcomer]
    used = newc[newc.used_ai == "Yes"]
    cases = [
        (
            "Newcomers who want to learn: AI undermines belonging",
            newc,
            newc.motivation == "Learn",
            newc.undermines_belonging == "Yes",
        ),
        (
            "Newcomers who value support from others: AI undermines belonging",
            newc,
            newc.belonging_factor == "Support from others",
            newc.undermines_belonging == "Yes",
        ),
        (
            "Newcomers who fix their own problem and used AI: willingness increased",
            used,
            used.motivation == "Fix a problem I face as a user",
            used.willingness_code == "increased",
        ),
        (
            "Maintainers x newcomers: AI undermines belonging",
            v,
            ~v.newcomer,
            v.undermines_belonging == "Yes",
        ),
    ]
    rows = []
    m = v[~v.newcomer & (v.undermines_belonging == "No") & v.undermines_explanation.notna()]
    assert NO_WITH_CAVEAT <= set(m.ts), (
        "NO_WITH_CAVEAT has timestamps outside the explained 'No' answers"
    )
    k = int(m.ts.isin(NO_WITH_CAVEAT).sum())
    rows.append(
        {
            "crosstab": "Maintainers who answer No and explain: explanation with a concern",
            "in_breakdown": f"{k}/{len(m)}",
            "outside_breakdown": "",
            "pct_breakdown": round(100 * k / len(m), 1),
            "pct_outside": np.nan,
            "p_fisher": np.nan,
        }
    )
    for name, d, group, outcome in cases:
        a, b = int((group & outcome).sum()), int((group & ~outcome).sum())
        c, e = int((~group & outcome).sum()), int((~group & ~outcome).sum())
        _, p = fisher_exact([[a, b], [c, e]])
        rows.append(
            {
                "crosstab": name,
                "in_breakdown": f"{a}/{a + b}",
                "outside_breakdown": f"{c}/{c + e}",
                "pct_breakdown": round(100 * a / (a + b), 1) if a + b else np.nan,
                "pct_outside": round(100 * c / (c + e), 1) if c + e else np.nan,
                "p_fisher": round(p, 3),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/survey_highlights.csv", index=False)
    return out


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] > y).sum() - (x[:, None] < y).sum()) / (len(x) * len(y))


def likert_per_belonging(df):
    m = df[(df.group == "Maintainer") & ~df.invalid]
    yes, no = m[m.undermines_belonging == "Yes"], m[m.undermines_belonging == "No"]
    rows = []
    for c in [c for c in m.columns if c.startswith("L_")]:
        x, y = yes[c].dropna().astype(int), no[c].dropna().astype(int)
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        rows.append(
            {
                "item": c,
                "n_yes": len(x),
                "n_no": len(y),
                "median_yes": x.median(),
                "median_no": y.median(),
                "pct_agree_yes": round(100 * (x >= 4).mean(), 1),
                "pct_agree_no": round(100 * (y >= 4).mean(), 1),
                "cliffs_delta": round(cliffs_delta(x, y), 3),
                "p": round(p, 3),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/survey_likert_per_belonging.csv", index=False)
    return out


def open_ended(df):
    cols = [
        "pid",
        "group",
        "ts",
        "invalid",
        "projects",
        "motivation",
        "belonging_factor",
        "undermines_belonging",
        "undermines_explanation",
        "used_ai",
        "how_used",
        "willingness",
        "willingness_code",
        "noticed_ai",
        "themes",
    ]
    out = df[cols].copy()
    for (g, ts), name in PROJECTS_REDACTED.items():
        sel = (out.group == g) & (out.ts == ts)
        assert sel.sum() == 1, (g, ts)
        out.loc[sel, "projects"] = name
    for (g, ts, c), (text, repl) in TEXT_REDACTED.items():
        sel = (out.group == g) & (out.ts == ts)
        assert sel.sum() == 1 and text in out.loc[sel, c].iloc[0], (g, ts, c)
        out.loc[sel, c] = out.loc[sel, c].str.replace(text, repl, regex=False)
    for c in ["undermines_explanation", "how_used", "willingness", "noticed_ai", "projects"]:
        out[c] = out[c].astype("string").str.replace(r"\s*\n\s*", " / ", regex=True)
    out.to_csv(f"{OUT_DIR}/survey_open_answers.csv", index=False)

    t = (
        df[~df.invalid & (df.themes != "")]
        .assign(theme=lambda d: d.themes.str.split(";"))
        .explode("theme")
    )
    t["audience"] = np.where(t.newcomer, "newcomers", "maintainers")
    cnt = (
        t.groupby(["audience", "theme", "undermines_belonging"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"Yes": "undermines_yes", "No": "undermines_no"})
    )
    cnt["total"] = cnt.sum(axis=1)
    cnt = cnt.sort_values(["audience", "total"], ascending=[True, False])
    cnt.to_csv(f"{OUT_DIR}/survey_themes.csv")

    assert set(t.theme) == set(CODEBOOK), set(t.theme) ^ set(CODEBOOK)
    text = out.set_index("pid")[["undermines_explanation", "how_used", "willingness", "noticed_ai"]]
    coded = t.groupby("theme")["pid"].apply(set)
    for k, c in CODEBOOK.items():
        for pid, excerpt in c["examples"]:
            assert pid in coded[k], (k, pid)
            assert any(excerpt in str(v) for v in text.loc[pid]), (k, pid, excerpt)
    n = t.groupby(["theme", "audience"]).size().unstack(fill_value=0)
    book = pd.DataFrame(
        [
            {
                "category": c["category"],
                "theme": k,
                "definition": c["definition"],
                "inclusion": c["include"],
                "exclusion": c["exclude"],
                "close_but_no": c["close"],
                "examples": " | ".join(f'"{e}" ({pid})' for pid, e in c["examples"]),
                "newcomers": n.loc[k, "newcomers"],
                "maintainers": n.loc[k, "maintainers"],
                "participants": " ".join(
                    sorted(coded[k], key=lambda p: (p[0] != "N", int(p[1:])))
                ),
            }
            for k, c in CODEBOOK.items()
        ]
    )
    book.to_csv(f"{OUT_DIR}/survey_codebook.csv", index=False)
    return cnt


if __name__ == "__main__":
    df = load()
    missing = [k for k in list(WILLINGNESS) + list(THEMES) if k not in set(zip(df.group, df.ts))]
    assert not missing, f"codes without a matching answer: {missing}"
    no_code = df[
        ~df.invalid
        & df.newcomer
        & (df.used_ai == "Yes")
        & df.willingness.notna()
        & df.willingness_code.isna()
    ]
    assert no_code.empty, f"answers without a willingness code: {no_code.ts.tolist()}"

    pd.set_option("display.width", 200)
    print(response_rate(df), "\n")
    closed(df)
    print(likert(df)[["item", "median", "pct_disagree_1_2", "pct_agree_4_5"]], "\n")
    crosstabs(df)
    print(highlights(df).to_string(index=False), "\n")
    print(likert_per_belonging(df).to_string(index=False), "\n")
    print(open_ended(df))
