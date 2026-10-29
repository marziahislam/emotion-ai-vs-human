import re
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

## align codeL in behaviorData.mat with the % fear proportion from stim file name (1 = most fearful, 7 = happiest)
CODEL_TO_FEAR = {1: 100, 2: 70, 3: 60, 4: 50, 5: 40, 6: 30, 7: 0}

# assess deepface model
model = pd.read_csv("results/deepface_probs.csv")
model["identity"] = model["file"].str.split("-").str[0] # "SF1-fe70-ha30.tif" -> identity "SF1", fear_pct 70
model["fear_pct"] = model["file"].apply(lambda f: int(re.search(r"fe(\d+)", f).group(1)))

model_curve = model.groupby("fear_pct")["p_fear_over_fh"].agg(["mean", "std"])

# assess human data from behaviorData.mat 
data = sio.loadmat("data/behaviorData.mat", squeeze_me=True, struct_as_record=False)
participants = data["combinedData"]   ## 66 neurotypical people (eye-tracking + EEG + fMRI)

per_person = []   # one row per (participant, morph level)
for pid, p in enumerate(participants):
    levels = np.atleast_1d(p.codeL)
    answers = np.atleast_1d(p.vResp)
    answered = answers > 0                      # 0 = no response, leave those out
    for code, fear_pct in CODEL_TO_FEAR.items():
        trials = answered & (levels == code)
        if trials.sum() == 0:                   # skip if this person never answered at this level
            continue
        per_person.append({
            "participant": pid,
            "fear_pct": fear_pct,
            "prop_fear": np.mean(answers[trials] == 1),   # 1 = said "fear"
        })
per_person = pd.DataFrame(per_person)

human_curve = per_person.groupby("fear_pct")["prop_fear"].agg(["mean", "sem"])

# table using pandas (side by side human vs model)
comparison = pd.DataFrame({
    "human_prop_fear": human_curve["mean"],
    "human_sem": human_curve["sem"],
    "model_p_fear": model_curve["mean"],
    "model_std_across_identities": model_curve["std"],
})
comparison["difference_model_minus_human"] = comparison["model_p_fear"] - comparison["human_prop_fear"]
comparison.to_csv("results/human_vs_deepface.csv")
print(comparison.round(3).to_string())

## plot using matplotlib
plt.figure(figsize=(6, 4.5))
x = comparison.index

plt.errorbar(x, comparison["human_prop_fear"], yerr=comparison["human_sem"],
             marker="o", capsize=3, label="Humans (n = 66)")
plt.plot(x, comparison["model_p_fear"], marker="s", label="DeepFace (mean of 4 identities)")

for ident, g in model.groupby("identity"):
    g = g.sort_values("fear_pct")
    plt.plot(g["fear_pct"], g["p_fear_over_fh"], linestyle=":", alpha=0.35, color="gray")

plt.axhline(0.5, color="lightgray", linewidth=0.8)
plt.xlabel("% fear in morphed face")
plt.ylabel("P(fear)")
plt.title("Fear judgments: humans vs. DeepFace")
plt.ylim(-0.02, 1.02)
plt.legend()
plt.tight_layout()
plt.savefig("results/human_vs_deepface.png", dpi=200)
print("\nSaved results/human_vs_deepface.csv and results/human_vs_deepface.png")