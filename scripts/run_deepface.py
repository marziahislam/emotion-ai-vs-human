import os
import pandas as pd
from deepface import DeepFace


STIMULI = "data/stimuli"
rows = []

for file_name in sorted(os.listdir(STIMULI)):
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
        continue

    try: 
        result = DeepFace.analyze(
            img_path=os.path.join(STIMULI, file_name),
            actions=["emotion"],
            enforce_detection=False,
        )[0]
        detected_emotion = result["emotion"]

        ## add deepface result for image into results table
        row = {"file": file_name, "prediction_label": result["dominant_emotion"]}
        ## add entries for corresponding key (emotion) and value (score) pairs
        row.update({f"p_{k}": v / 100 for k, v in detected_emotion.items()}) 
        row["p_fear_over_fh"] = detected_emotion["fear"] / (detected_emotion["fear"] + detected_emotion["happy"])
        rows.append(row)
        print(f"{file_name}: {row['prediction_label']}  p_fear_over_fh= {row['p_fear_over_fh']:.3f}")

    except Exception as e:
        print(f"Failed on {file_name}: {e}")


os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv("results/deepface_probs.csv", index=False)
print(f"Saved {len(rows)} images to results/deepface_probs.csv")