"""
Customer Support Ticket Classification - full project script
============================================================
Goal: read a customer support ticket, decide which ISSUE it describes
(login, network, hardware, software bug, data loss, security, general)
and route it to the right department.

This is a TEXT CLASSIFICATION problem.

How to run (from inside the Ticket_Classification folder):
    pip install -r requirements.txt
    python main.py

It expects the data at:  dataset/customer_support_tickets.csv

Everything the script creates is saved in the  results/  folder.
(The same steps, with longer explanations, are in
 notebooks/Ticket_Classification.ipynb)
"""

import matplotlib

matplotlib.use("Agg")  # lets us save charts without opening windows

import os
import random
import re
import time
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def display(obj):  # print tables as plain text
    print(obj.to_string() if hasattr(obj, "to_string") else obj)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
print("Libraries loaded.")


# --- 0.2 Settings and a chart helper ---
ROOT = "." if os.path.exists("dataset/customer_support_tickets.csv") else ".."
DATA_PATH = os.path.join(ROOT, "dataset", "customer_support_tickets.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

RANDOM_STATE = 42
GENERAL = "General Troubleshooting"
MIXED = "Mixed"


def save_plot(filename):
    """Save the current chart to results/visualizations and show it."""
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, filename), dpi=120)
    if matplotlib.get_backend().lower() == "agg":
        plt.close()  # running as a script: no window to show
    else:
        plt.show()  # running in Jupyter: show the chart


print("Project folder:", os.path.abspath(ROOT))

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Save the ticket data as "
        "dataset/customer_support_tickets.csv and run again."
    )

df = pd.read_csv(DATA_PATH)

REQUIRED = ["Ticket ID", "Ticket Description", "Product Purchased",
            "Ticket Type", "Ticket Priority", "Ticket Channel", "Ticket Status"]
missing_cols = [c for c in REQUIRED if c not in df.columns]
if missing_cols:
    raise ValueError(f"The dataset is missing these expected columns: {missing_cols}")

print("Shape (rows, columns):", df.shape)
show_cols = [c for c in ["Ticket Type", "Ticket Subject", "Product Purchased",
                         "Ticket Priority", "Ticket Channel"] if c in df.columns]
display(df[show_cols].head())

print("Share of empty cells (top 5 columns):")
print(df.isna().mean().round(3).sort_values(ascending=False).head(5))

print("\nDuplicate ticket IDs:", df["Ticket ID"].duplicated().sum())
df = df.drop_duplicates(subset="Ticket ID").copy()


# --- 2.2 Build a clean text column ---
def clean_text(description, product):
    text = str(description).replace("{product_purchased}", str(product))
    text = text.replace("{error_message}", "an error message")
    return re.sub(r"\s+", " ", text).strip()


df["text"] = [clean_text(d, p) for d, p in zip(df["Ticket Description"], df["Product Purchased"])]
print("Example cleaned ticket:")
print(df["text"].iloc[2])

cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
audit = []
for col in ["Ticket Type", "Ticket Priority", "Ticket Channel", "Ticket Status"]:
    quick = make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True), LogisticRegression(max_iter=500))
    acc = cross_val_score(quick, df["text"], df[col], cv=cv5, n_jobs=-1).mean()
    audit.append({"column": col, "accuracy": round(acc, 3), "random chance": round(1 / df[col].nunique(), 3)})
audit = pd.DataFrame(audit)

audit.set_index("column").plot.barh(figsize=(8, 3.5), color=["#2a6f97", "#bbbbbb"])
plt.title("Can the ticket text predict this column?")
plt.xlabel("5-fold CV accuracy")
save_plot("01_data_audit.png")
display(audit)

ISSUE_RULES = {
    "Account Access": r"\baccount\b|password|logging into|log in|login|credentials|locked|unlock",
    "Network": r"network|wi-?fi|internet connection|disconnect|connect to any",
    "Hardware": (r"hardware|strange noises|not turning on|flickering|charger|not charging"
                 r"|battery|repair or replacement|used to last much longer"),
    "Software Bug": r"software bug|application crashes|\bcrashes\b|glitch|freezes frequently",
    "Data Loss": (r"data loss issue|deleted important data|lost all the data|files and documents"
                  r"|disappeared|recover the deleted|recover the lost|lost data|retrieve them"),
    "Security": r"security of my|data is safe",
}
COMPILED = {k: re.compile(v, re.I) for k, v in ISSUE_RULES.items()}

DEPARTMENTS = {
    "Account Access": "Account & Security Team", "Security": "Account & Security Team",
    "Network": "Network Support", "Hardware": "Hardware & Repairs",
    "Software Bug": "Software Support", "Data Loss": "Data Recovery Team",
    GENERAL: "General Support (Tier 1)",
}


def weak_label(text):
    hits = {k: len(rx.findall(text)) for k, rx in COMPILED.items()}
    best = max(hits.values())
    if best == 0:
        return GENERAL
    winners = [k for k, v in hits.items() if v == best]
    return winners[0] if len(winners) == 1 else MIXED


df["issue"] = df["text"].map(weak_label)
print("Ambiguous tickets removed:", (df["issue"] == MIXED).sum())
df = df[df["issue"] != MIXED].copy()
df["department"] = df["issue"].map(DEPARTMENTS)


# --- 4.2 Urgency cues and a "smart priority" ---
URGENT_RE = re.compile(r"urgent|as soon as possible|asap|affecting my work|hindering my productivity", re.I)
REPEAT_RE = re.compile(r"contacted customer support multiple times|issue remains unresolved", re.I)

df["urgent"] = df["text"].str.contains(URGENT_RE)
df["repeat_contact"] = df["text"].str.contains(REPEAT_RE)
df["smart_priority"] = np.select(
    [df["urgent"] & df["repeat_contact"], df["urgent"] | df["repeat_contact"]],
    ["Escalate", "High"],
    default="Normal",
)
df["text_length"] = df["text"].str.len()

print("Rows used:", len(df))
print("\nTickets per issue:")
print(df["issue"].value_counts())

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
order = df["issue"].value_counts().index
sns.countplot(data=df, y="issue", order=order, color="steelblue", ax=ax[0])
ax[0].set_title("Tickets per issue (class balance)")
sns.countplot(data=df, x="smart_priority", order=["Normal", "High", "Escalate"], color="darkorange", ax=ax[1])
ax[1].set_title("Smart priority levels")
save_plot("02_class_balance_and_priority.png")


# --- 5.2 Ticket length by issue ---
plt.figure(figsize=(9, 4))
sns.boxplot(data=df, x="issue", y="text_length", order=order)
plt.title("Ticket text length by issue")
plt.xticks(rotation=25)
save_plot("03_text_length_by_issue.png")


# --- 5.3 Issue mix per product ---
top_products = df["Product Purchased"].value_counts().head(10).index
mix = pd.crosstab(df["Product Purchased"], df["issue"], normalize="index").loc[top_products]
plt.figure(figsize=(10, 5))
sns.heatmap(mix, annot=True, fmt=".2f", cmap="Blues")
plt.title("Issue mix per product (share of each product's tickets)")
save_plot("04_issue_mix_per_product.png")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["issue"], test_size=0.2, stratify=df["issue"], random_state=RANDOM_STATE
)
print(len(X_train), "training tickets |", len(X_test), "test tickets")


# --- 6.2 Augmentation (training data only) ---
"""Training-time text augmentation.

The raw dataset repeats a small set of template sentences, so a model trained
on it only learns those exact words. This bank adds many alternative phrasings
per issue. Sentences are mixed 1-2 at a time with openers, closers and a
product name to create extra *training-only* tickets. {p} is replaced by a
product name.
"""

BANK = {
    "Account Access": [
        "I am not able to sign in to my {p} profile.",
        "My {p} login is being rejected and I am locked out.",
        "The reset email for my {p} passcode never arrives.",
        "I can't get into my {p} account after changing the password.",
        "The sign-in page says my credentials are wrong but I'm sure they are right.",
        "My profile was locked after too many attempts, how do I unlock it?",
        "I forgot my PIN and the recovery link does not work.",
        "The two-factor verification code is not being sent so I cannot log in.",
        "Unable to authenticate on my {p}, it keeps asking me to sign in again.",
        "My username is not recognised and I cannot access my profile.",
        "I can't open my user profile, it says access denied.",
        "The password reset link has expired and I am locked out of {p}.",
    ],
    "Network": [
        "My {p} will not connect to the wireless network.",
        "The signal keeps dropping and the device goes offline.",
        "There is no internet connection on my {p} although other devices work on the same router.",
        "It can't find my home Wi-Fi and the connection fails every time.",
        "The {p} loses connectivity every few minutes.",
        "I cannot get online, the router is fine but the {p} shows no signal.",
        "Bluetooth and wireless pairing both fail on my {p}.",
        "The device keeps dropping off the network and reconnecting.",
        "Internet is very slow only on my {p} and pages time out.",
        "The {p} says connected but there is no internet access.",
        "I've restarted the router but the {p} still won't join the network.",
        "There are constant connection drops when streaming on my {p}.",
    ],
    "Hardware": [
        "My {p} will not power on at all, it seems completely dead.",
        "There is a grinding noise coming from the {p}.",
        "The screen goes black and shows flickering lines.",
        "The battery drains within an hour even though I use the original charger.",
        "My {p} overheats and shuts down suddenly.",
        "The buttons on the {p} are stuck and not responding.",
        "The {p} has a cracked screen after a small fall.",
        "It will not switch on and the charging light does not come up.",
        "The speaker is crackling and the sound is distorted.",
        "The device does not charge no matter which cable I use.",
        "I think the {p} is physically damaged and needs a repair.",
        "The display is dim with strange colour stripes.",
    ],
    "Software Bug": [
        "The {p} app keeps crashing when I open the settings page.",
        "The program hangs and then closes by itself.",
        "There is a bug where the application freezes every few minutes.",
        "After the latest update the {p} software behaves strangely and keeps glitching.",
        "An error pops up and the app stops responding.",
        "The application shuts down whenever I tap a certain feature.",
        "The software on my {p} is lagging and not responding.",
        "The interface gets stuck on a loading screen forever.",
        "The {p} app force closes immediately after launching.",
        "Some features inside the software do not work as they should, it looks like a bug.",
        "The firmware update broke a function that used to work.",
        "The program throws an exception and quits.",
    ],
    "Data Loss": [
        "My photos and documents vanished from the {p} after a restart.",
        "I accidentally erased saved projects, can they be restored?",
        "All my files are gone and I need help to get them back.",
        "Everything stored on my {p} was wiped after the update.",
        "I deleted a folder by mistake and want to recover it.",
        "My saved settings and backups have disappeared.",
        "After the crash all the content on my {p} is missing, can it be restored?",
        "The memory was reset and my personal files were lost.",
        "I need to restore my deleted contacts and messages.",
        "Files that I saved yesterday are no longer there.",
        "Is it possible to retrieve what I removed by accident from my {p}?",
        "My backup is empty and the data I stored is gone.",
    ],
    "Security": [
        "I am worried about the privacy of my information on the {p}.",
        "How do I make sure my personal data is protected and secure?",
        "I think someone may have hacked my {p}, is my data safe?",
        "I received a suspicious message and I'm concerned about a security breach.",
        "Is my information encrypted on this device?",
        "Could someone else access my private data through the {p}?",
        "I want to know how my data is stored and who can see it.",
        "My {p} shows a warning about unauthorized access, please check security.",
        "I'm concerned my personal information might leak.",
        "Please confirm that my details are kept confidential and secure.",
    ],
    GENERAL: [
        "Something is wrong with my {p}, please assist.",
        "I have a problem with my product and would like some help.",
        "It is not working properly even after following the manual.",
        "I need some assistance with my {p}.",
        "Could you please look into an issue I am facing?",
        "My {p} is not behaving as expected.",
        "I've tried the usual steps but nothing helps.",
        "Can someone from support get in touch regarding my {p}?",
        "There's an issue and I'm not sure what is causing it.",
        "Please advise on what to do next with my {p}.",
    ],
}
OPENERS = ["Hello,", "Hi team,", "Hi,", "", "", "Good day,"]
CLOSERS = ["Thanks.", "Please help.", "Any advice?", "Thank you!", "", "", "I would appreciate a quick reply."]


def make_augmented(products, per_issue=300, seed=42):
    """Return a DataFrame(text, issue) of synthetic training tickets."""
    rng = random.Random(seed)
    rows = []
    for issue, sentences in BANK.items():
        n = per_issue // 2 if issue == GENERAL else per_issue
        for _ in range(n):
            parts = rng.sample(sentences, rng.choice([1, 1, 2]))
            prod = rng.choice(products)
            text = " ".join([rng.choice(OPENERS)] + [s.format(p=prod) for s in parts] + [rng.choice(CLOSERS)])
            rows.append({"text": re.sub(r"\s+", " ", text).strip(), "issue": issue})
    return pd.DataFrame(rows)


print("Augmentation bank ready:", sum(len(v) for v in BANK.values()), "alternative sentences.")

aug = make_augmented(sorted(df["Product Purchased"].unique()), per_issue=300, seed=RANDOM_STATE)
X_train_aug = pd.concat([X_train, aug["text"]], ignore_index=True)
y_train_aug = pd.concat([y_train, aug["issue"]], ignore_index=True)

print("Real training tickets:", len(X_train), "| after augmentation:", len(X_train_aug))
display(aug.sample(3, random_state=1))

def make_pipe(clf):
    # TF-IDF converts text to numbers, then the classifier learns from them
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("model", clf)])


models = {
    "KNN": make_pipe(KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine", algorithm="brute")),
    "SVM": make_pipe(SVC(kernel="linear", C=5, class_weight="balanced")),
    "Decision Tree": make_pipe(DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
    "Random Forest": make_pipe(RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE,
                                                      class_weight="balanced")),
}

rows, preds = [], {}
for name, pipe in models.items():
    t0 = time.time()
    pipe.fit(X_train_aug, y_train_aug)  # the model learns here
    p = pipe.predict(X_test)  # predictions on tickets it has never seen
    preds[name] = p
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, p),
                 "Precision": precision_score(y_test, p, average="macro", zero_division=0),
                 "Recall": recall_score(y_test, p, average="macro", zero_division=0),
                 "F1 (macro)": f1_score(y_test, p, average="macro", zero_division=0),
                 "Train time (s)": time.time() - t0})
    print(f"Trained: {name}")

results = pd.DataFrame(rows).round(3).sort_values("F1 (macro)", ascending=False)
results.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
display(results)

results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 (macro)"]].plot.bar(
    figsize=(9, 4.5), rot=0, ylim=(0, 1.08))
plt.title("KNN vs SVM vs Decision Tree vs Random Forest")
plt.legend(loc="lower right")
save_plot("05_model_comparison.png")

best_name = results.iloc[0]["Model"]
print("Best model (by macro F1):", best_name)

gap = []
for name, pipe in models.items():
    gap.append({"Model": name,
                "Train accuracy": accuracy_score(y_train, pipe.predict(X_train)),
                "Test accuracy": accuracy_score(y_test, pipe.predict(X_test))})
gap = pd.DataFrame(gap).round(3)
gap["Gap"] = (gap["Train accuracy"] - gap["Test accuracy"]).round(3)
display(gap)

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(y_test, preds[best_name], ax=ax, xticks_rotation=40,
                                        cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix ({best_name})")
save_plot("06_confusion_matrix.png")


# --- 9.2 Report for each issue ---
print(classification_report(y_test, preds[best_name], zero_division=0))

unseen_rows = [
    ("I can't sign in to my profile and the reset email never arrives.", 'Account Access'),
    ('My login keeps getting rejected and now the profile is locked out.', 'Account Access'),
    ('Forgot my passcode and the reset link is not working.', 'Account Access'),
    ('The device will not join my home wireless and keeps dropping the signal.', 'Network'),
    ('No internet on the gadget although my phone works on the same router.', 'Network'),
    ('It cannot find any wifi networks nearby.', 'Network'),
    ('The unit makes a grinding sound and will not power up.', 'Hardware'),
    ('Display shows flickering lines and then goes black.', 'Hardware'),
    ('Battery drains within an hour even with the original charger.', 'Hardware'),
    ('Device is dead and does not switch on at all.', 'Hardware'),
    ('The program keeps crashing whenever I open the settings page.', 'Software Bug'),
    ('There is a bug in the app, it hangs and closes by itself.', 'Software Bug'),
    ('The application glitches and freezes every few minutes.', 'Software Bug'),
    ('Photos and documents vanished from my device after the restart.', 'Data Loss'),
    ('I accidentally erased my saved projects, can they be restored?', 'Data Loss'),
    ('All my files are gone, I need help to get them back.', 'Data Loss'),
    ('I am worried about the privacy of my information on this device.', 'Security'),
    ('Is my personal data protected? I want to make sure it is secure.', 'Security'),
    ('Something is wrong with my product, please help.', 'General Troubleshooting'),
    ('I have a problem and need some assistance.', 'General Troubleshooting'),
    ('It does not work properly even after I tried the manual steps.', 'General Troubleshooting'),
    ('Cannot get into my user area, the system keeps saying my details are invalid.', 'Account Access'),
    ('Locked out after typing the wrong secret code too many times.', 'Account Access'),
    ('Never received the verification message to reopen my access.', 'Account Access'),
    ('Connection to the web is unstable, the gadget goes offline repeatedly.', 'Network'),
    ('My unit refuses to link up with the household router.', 'Network'),
    ('Streaming buffers constantly because the connection keeps dying.', 'Network'),
    ('A rattling noise is coming from inside and the thing overheats quickly.', 'Hardware'),
    ('Screen is cracked and shows coloured lines.', 'Hardware'),
    ("It won't charge, the port seems damaged.", 'Hardware'),
    ('The power button is jammed and nothing turns on.', 'Hardware'),
    ('Every time I tap the camera feature the program quits unexpectedly.', 'Software Bug'),
    ('The software is buggy, it stalls and throws errors constantly.', 'Software Bug'),
    ('The app is lagging and closes on its own.', 'Software Bug'),
    ('My saved pictures and notes are no longer there after the update.', 'Data Loss'),
    ('I wiped a folder by mistake and would like it restored.', 'Data Loss'),
    ('Everything I stored has been erased from the memory.', 'Data Loss'),
    ('Someone might have broken into my device, how can I protect my information?', 'Security'),
    ("I'd like to know if my private details are kept confidential.", 'Security'),
    ("The product isn't working as it should, kindly assist.", 'General Troubleshooting'),
    ("I'm facing a difficulty and need your guidance.", 'General Troubleshooting'),
    ('Please contact me about my problem.', 'General Troubleshooting'),
]

unseen = pd.DataFrame(unseen_rows, columns=["text", "expected_issue"])
unseen_scores = {}
for name, pipe in models.items():
    unseen[name] = pipe.predict(unseen["text"])
    unseen_scores[name] = (unseen[name] == unseen["expected_issue"]).mean()

unseen_table = pd.DataFrame({"Model": list(unseen_scores),
                             "Accuracy on new wording": list(unseen_scores.values())}).round(3)
display(unseen_table)

wrong = unseen[unseen[best_name] != unseen["expected_issue"]][["text", "expected_issue", best_name]]
print(f"{len(wrong)} of {len(unseen)} new-wording tickets were wrong for {best_name}:")
display(wrong)

rf = models["Random Forest"]
words = rf.named_steps["tfidf"].get_feature_names_out()
importance = pd.Series(rf.named_steps["model"].feature_importances_, index=words).sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 5))
sns.barplot(x=importance.values, y=importance.index, color="darkorange")
plt.title("Most important words (Random Forest)")
plt.xlabel("Importance")
save_plot("07_important_words.png")

def predict_ticket(text, product=None, model_name=None):
    model = models[model_name or best_name]
    cleaned = clean_text(text, product or "product")
    issue = model.predict([cleaned])[0]
    urgent = bool(URGENT_RE.search(cleaned))
    repeat = bool(REPEAT_RE.search(cleaned))
    priority = "Escalate" if urgent and repeat else "High" if urgent or repeat else "Normal"
    return {"issue": issue, "department": DEPARTMENTS[issue], "priority": priority}


examples = ["I can't sign in to my profile and need this fixed urgently.",
            "The unit makes a grinding sound and will not power up.",
            "Photos and documents vanished from my device after the restart.",
            "My device will not join the wireless network."]
display(pd.DataFrame([{"ticket": t, **predict_ticket(t)} for t in examples]))

joblib.dump(models[best_name], os.path.join(RESULTS_DIR, "best_model.joblib"))

best_row = results.iloc[0]
report = f"""TICKET CLASSIFICATION - FINAL REPORT
====================================

Tickets used after cleaning : {len(df)}
Training / testing tickets  : {len(X_train)} / {len(X_test)} (+{len(aug)} augmented training tickets)

MODEL COMPARISON (real test tickets)
{results.to_string(index=False)}

ACCURACY ON NEW WORDING (42 hand-written tickets)
{unseen_table.to_string(index=False)}

Best model by macro F1: {best_name}
- Accuracy {best_row['Accuracy'] * 100:.1f}%, macro F1 {best_row['F1 (macro)'] * 100:.1f}%.

Charts are saved in results/visualizations/
The best model is saved as results/best_model.joblib
"""

with open(os.path.join(RESULTS_DIR, "final_report.txt"), "w") as f:
    f.write(report)

print(report)

