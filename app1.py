"""
Imagined-speech EEG -> word -> speech   (Streamlit app)

Run:   pip install -r requirements.txt
       streamlit run app.py
Data:  upload Imagined_speech_EEG_edf.zip in the sidebar, or point to the extracted folder.
"""
import glob, io, itertools, os, re, tempfile, warnings, zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, filtfilt, welch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

mne.set_log_level("ERROR"); warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})

CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
FS, N = 128, 1536
BANDS = ["delta 1-4", "theta 4-8", "alpha 8-13", "beta-lo 13-20", "beta-hi 20-30", "gamma 30-45"]
EDGES = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]


# ----------------------------------------------------------------------------- data
def find_root(path):
    for d, subs, _ in os.walk(path):
        if {"Char", "Digit", "Image"} <= set(subs):
            return d
    return None


@st.cache_resource(show_spinner="Extracting zip…")
def extract_zip(data: bytes):
    tmp = tempfile.mkdtemp(prefix="eeg_")
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
    return tmp


def read_trial(path):
    d = mne.io.read_raw_edf(path, preload=True).get_data(picks=CH) * 1e6
    return d[:, :N] if d.shape[1] >= N else np.pad(d, ((0, 0), (0, N - d.shape[1])), "edge")


@st.cache_data(show_spinner="Reading EDF files…")
def load_task(root, task):
    X, y, g = [], [], []
    for f in sorted(glob.glob(os.path.join(root, task, "*.edf"))):
        m = re.match(r".*name(\d+)_(.+)\.edf", f)
        if not m:
            continue
        X.append(read_trial(f)); g.append(int(m[1]))
        y.append(m[2].capitalize() if task == "Image" else m[2])
    return np.array(X), np.array(y), np.array(g)


# ----------------------------------------------------------------------------- features
_b, _a = butter(4, [1, 45], btype="band", fs=FS)


def bandpass(X):
    return filtfilt(_b, _a, X - X.mean(-1, keepdims=True), axis=-1)


def features(X):
    f, P = welch(bandpass(X), fs=FS, nperseg=128, axis=-1)
    F = [np.log(P[..., (f >= lo) & (f < hi)].mean(-1) + 1e-8) for lo, hi in EDGES]
    return np.concatenate(F, -1).reshape(len(X), -1)          # 14 ch x 6 bands = 84


def subject_norm(F, g):
    F = F.copy()
    for s in np.unique(g):
        m = g == s
        F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-8)
    return F


# ----------------------------------------------------------------------------- models
def get_models():
    sc = StandardScaler
    return {
        "SVM (RBF)": make_pipeline(sc(), SVC(C=1, probability=True, random_state=0)),
        "SVM (Linear)": make_pipeline(sc(), SVC(kernel="linear", C=0.01, probability=True, random_state=0)),
        "Logistic Regression": make_pipeline(sc(), LogisticRegression(C=0.05, max_iter=3000)),
        "LDA (shrinkage)": make_pipeline(sc(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "Naive Bayes": make_pipeline(sc(), GaussianNB()),
        "k-NN": make_pipeline(sc(), KNeighborsClassifier(7, weights="distance")),
        "Random Forest": RandomForestClassifier(200, random_state=0),
        "Extra Trees": ExtraTreesClassifier(200, random_state=0),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, max_depth=3,
                                                            min_samples_leaf=4, random_state=0),
        "MLP (neural net)": make_pipeline(sc(), MLPClassifier((64,), alpha=1.0, max_iter=800, random_state=0)),
        "Top-20 features + LogReg": make_pipeline(sc(), SelectKBest(f_classif, k=20),
                                                  LogisticRegression(C=0.1, max_iter=3000)),
        "Ensemble (soft vote)": VotingClassifier([
            ("svm", make_pipeline(sc(), SVC(C=1, probability=True, random_state=0))),
            ("lr", make_pipeline(sc(), LogisticRegression(C=0.05, max_iter=3000))),
            ("rf", RandomForestClassifier(200, random_state=0)),
            ("et", ExtraTreesClassifier(200, random_state=0))], voting="soft"),
    }


MODELS = get_models()


def make_model(name):
    return clone(MODELS[name])


def cv_splits(F, y, g, seed):
    return StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(F, y, g)


@st.cache_data(show_spinner=False)
def compare_models(F, y, g, names, reps):
    rows, folds = [], {}
    bar = st.progress(0.0, text="Comparing models…")
    for i, n in enumerate(names):
        acc = []
        for r in range(reps):
            for tr, te in cv_splits(F, y, g, r):
                acc.append((make_model(n).fit(F[tr], y[tr]).predict(F[te]) == y[te]).mean())
        folds[n] = acc
        rows.append({"Model": n, "Accuracy": np.mean(acc), "Std (folds)": np.std(acc)})
        bar.progress((i + 1) / len(names), text=f"Done: {n}")
    bar.empty()
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True), folds


@st.cache_data(show_spinner="Collecting out-of-fold predictions…")
def oof_predict(F, y, g, name, reps):
    classes = np.array(sorted(set(y)))
    yt, pr, fa = [], [], []
    for r in range(reps):
        for tr, te in cv_splits(F, y, g, r):
            mdl = make_model(name).fit(F[tr], y[tr]); p = mdl.predict_proba(F[te])
            yt += list(y[te]); pr.append(p); fa.append((mdl.classes_[p.argmax(1)] == y[te]).mean())
    return np.array(yt), np.vstack(pr), fa, classes


@st.cache_data(show_spinner="Leave-one-subject-out…")
def loso(F, y, g, name):
    return {int(g[te][0]): (make_model(name).fit(F[tr], y[tr]).predict(F[te]) == y[te]).mean()
            for tr, te in LeaveOneGroupOut().split(F, y, g)}


@st.cache_data(show_spinner="Running permutation test…")
def permutation_test(F, y, g, name, n_perm, obs_seed=0):
    rng = np.random.RandomState(0); null = []
    for _ in range(n_perm):
        yp = y.copy()
        for s in np.unique(g):
            idx = np.where(g == s)[0]; yp[idx] = rng.permutation(y[idx])
        a = [(make_model(name).fit(F[tr], yp[tr]).predict(F[te]) == yp[te]).mean()
             for tr, te in cv_splits(F, yp, g, int(rng.randint(1000)))]
        null.append(np.mean(a))
    return np.array(null)


def speak(text):
    try:
        from gtts import gTTS
        buf = io.BytesIO(); gTTS(text=text, lang="en").write_to_fp(buf)
        st.audio(buf.getvalue(), format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"Text-to-speech unavailable (needs internet + gTTS): {e}")


# ----------------------------------------------------------------------------- UI
st.set_page_config(page_title="Imagined-speech EEG", page_icon="🧠", layout="wide")
st.title("🧠BCI based Brain signals to speech conversion using machine learning")
st.caption("Emotiv EPOC (14 ch, 128 Hz). Accuracy is always measured on **unseen subjects** "
           "(subject-wise CV). Each subject has only one trial per word, so expect modest numbers.")

with st.sidebar:
    st.header("1. Data")
    src = st.radio("Source", ["Upload zip", "Folder path"], horizontal=True)
    root = None
    if src == "Upload zip":
        up = st.file_uploader("Imagined_speech_EEG_edf.zip", type="zip")
        if up is not None:
            root = find_root(extract_zip(up.getvalue()))
    else:
        p = st.text_input("Folder", os.environ.get("EEG_DATA_DIR", "./Imagined_speech_EEG_edf"))
        root = find_root(p) if os.path.isdir(p) else None
    if root is None:
        st.info("Provide the dataset to begin."); st.stop()

    task = st.selectbox("Task", ["Image", "Digit", "Char"], help="Image = 10 words, Digit = 0-9, Char = 10 letters")
    X, y, g = load_task(root, task)
    labels = sorted(set(y))
    default = {"Image": ["Car", "Scooter"], "Digit": ["4", "9"], "Char": ["A", "S"]}[task]
    words = st.multiselect("Words (pick 2 or 3)", labels, default=default, max_selections=3)
    if len(words) < 2:
        st.warning("Select at least 2 words."); st.stop()

    st.header("2. Models")
    all_names = list(MODELS)
    chosen = st.multiselect("Models to compare", all_names,
                            default=["SVM (RBF)", "Logistic Regression", "LDA (shrinkage)", "Naive Bayes",
                                     "k-NN", "Extra Trees"])
    reps = st.slider("CV repeats", 1, 10, 3)

    st.header("3. Time window")
    t0, t1 = st.slider("Seconds of each trial to use", 0, 12, (0, 12),
                       help="Some trials have a large burst in the first seconds (movement/blink/onset). "
                            "Try 3-12 s and see how accuracy changes.")
    if t1 - t0 < 2:
        st.warning("Use a window of at least 2 s."); st.stop()

# Selected subset
msk = np.isin(y, words)
Xm, ym, gm = X[msk], y[msk], g[msk]
Fall = subject_norm(features(X[..., t0 * FS:t1 * FS]), g)
Fm = Fall[msk]
K = len(words); classes = sorted(words); colors = plt.cm.tab10(np.arange(K))
mu_, sd_ = st.columns(2)
mu_.metric("Trials", len(ym)); sd_.metric("Subjects", len(np.unique(gm)))

tab_data, tab_cmp, tab_det, tab_pred = st.tabs(["📊 Data", "🏁 Model comparison", "🔬 Detailed results", "🗣️ Predict & speak"])

# ---- Data tab
with tab_data:
    Xf = bandpass(Xm); t = np.arange(N) / FS
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Example trial per word")
        fig, axes = plt.subplots(K, 1, figsize=(7, 2.2 * K), sharex=True); axes = np.atleast_1d(axes)
        for ax, c, col in zip(axes, classes, colors):
            i = np.where(ym == c)[0][0]
            for j, ch in enumerate(["AF3", "F7", "T7", "O1"]):
                ax.plot(t, Xf[i, CH.index(ch)] + j * 80, color=col, lw=.6)
            ax.set_yticks([0, 80, 160, 240]); ax.set_yticklabels(["AF3", "F7", "T7", "O1"]); ax.set_title(c, loc="left", fontsize=9)
        axes[-1].set_xlabel("Time (s)"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        st.subheader("Average spectrum per word")
        f, P = welch(Xf, fs=FS, nperseg=256, axis=-1); P = P.mean(1)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for c, col in zip(classes, colors):
            Pc = 10 * np.log10(P[ym == c]); mu = Pc.mean(0); se = Pc.std(0) / np.sqrt(len(Pc))
            ax.plot(f, mu, color=col, label=c); ax.fill_between(f, mu - se, mu + se, color=col, alpha=.2)
        ax.set_xlim(1, 45); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    Fz = Fm.reshape(len(Fm), 14, 6)
    st.subheader("Scalp maps: word-mean band power (z-scored within subject)")
    info = mne.create_info(CH, FS, "eeg"); info.set_montage("standard_1020")
    vmax = np.abs(np.stack([Fz[ym == c].mean(0) for c in classes])).max()
    fig, axes = plt.subplots(K, 6, figsize=(12, 2.2 * K + .5)); axes = np.atleast_2d(axes)
    for r, c in enumerate(classes):
        mu = Fz[ym == c].mean(0)
        for b in range(6):
            im, _ = mne.viz.plot_topomap(mu[:, b], info, axes=axes[r, b], show=False, vlim=(-vmax, vmax), cmap="RdBu_r")
            if r == 0: axes[r, b].set_title(BANDS[b], fontsize=8)
        axes[r, 0].set_ylabel(c, rotation=0, labelpad=22)
    fig.colorbar(im, ax=axes, shrink=.6); st.pyplot(fig); plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Band power per word")
        fig, ax = plt.subplots(figsize=(6, 3.5)); w = .8 / K
        for k, (c, col) in enumerate(zip(classes, colors)):
            v = Fz[ym == c].mean(1)
            ax.bar(np.arange(6) + k * w, v.mean(0), w, yerr=v.std(0) / np.sqrt(len(v)), color=col, label=c, capsize=2)
        ax.set_xticks(np.arange(6) + w * (K - 1) / 2); ax.set_xticklabels(BANDS, rotation=25, fontsize=8)
        ax.axhline(0, color="k", lw=.5); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c4:
        st.subheader("PCA of features")
        Z = PCA(2).fit_transform(Fm); fig, ax = plt.subplots(figsize=(6, 3.5))
        for c, col in zip(classes, colors): ax.scatter(*Z[ym == c].T, color=col, label=c, s=25, alpha=.8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ---- Comparison tab
with tab_cmp:
    st.write(f"Words: **{', '.join(classes)}** · chance = **{1/K:.0%}** · {reps}×5 subject-wise folds")
    if not chosen:
        st.info("Pick at least one model in the sidebar.")
    elif st.button("▶ Run comparison", type="primary"):
        st.session_state["cmp"] = compare_models(Fm, ym, gm, tuple(chosen), reps)
        st.session_state["cmp_key"] = (task, tuple(words))
    if st.session_state.get("cmp_key") == (task, tuple(words)) and "cmp" in st.session_state:
        df, folds = st.session_state["cmp"]
        c1, c2 = st.columns([1, 1.3])
        with c1:
            show = df.copy(); show["Accuracy"] = (show["Accuracy"] * 100).round(1).astype(str) + "%"
            show["Std (folds)"] = (show["Std (folds)"] * 100).round(1).astype(str) + " pts"
            st.dataframe(show, hide_index=True)
            st.caption("Differences of a few points are within noise (fold-to-fold std is ~10 pts with 23 subjects).")
        with c2:
            fig, ax = plt.subplots(figsize=(7, .5 * len(df) + 1.5))
            d = df.iloc[::-1]
            ax.barh(d["Model"], d["Accuracy"], xerr=d["Std (folds)"], color="steelblue", capsize=3)
            ax.axvline(1 / K, color="r", ls="--", label="chance"); ax.set_xlim(0, 1); ax.legend()
            ax.set_xlabel("Accuracy on unseen subjects"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.boxplot([folds[n] for n in df["Model"]])
            ax.set_xticklabels(list(df["Model"]), rotation=35, ha="right", fontsize=8)
            ax.axhline(1 / K, color="r", ls="--"); ax.set_ylabel("Fold accuracy")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ---- Detailed tab
with tab_det:
    default_model = st.session_state["cmp"][0].iloc[0]["Model"] if st.session_state.get("cmp_key") == (task, tuple(words)) and "cmp" in st.session_state else "SVM (RBF)"
    model_name = st.selectbox("Model", all_names, index=all_names.index(default_model), key="model_detail")
    yt, pr, fa, cls = oof_predict(Fm, ym, gm, model_name, reps)
    yp = cls[pr.argmax(1)]
    st.metric("Accuracy (unseen subjects)", f"{np.mean(yt == yp):.1%}", f"{np.mean(yt == yp) - 1 / K:+.1%} vs chance")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ConfusionMatrixDisplay(confusion_matrix(yt, yp, labels=cls, normalize="true"), display_labels=cls).plot(
            ax=ax, cmap="Blues", values_format=".2f", colorbar=False); ax.set_title("Confusion matrix")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        for k, c in enumerate(cls):
            fpr, tpr, _ = roc_curve(yt == c, pr[:, k]); ax.plot(fpr, tpr, label=f"{c} (AUC {auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=.7); ax.set_title("ROC (one-vs-rest)"); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c3:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.hist(fa, bins=12, color="steelblue"); ax.axvline(1 / K, color="r", ls="--"); ax.axvline(np.mean(fa), color="k")
        ax.set_title("Fold accuracy"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("Per-subject accuracy (leave-one-subject-out)")
    sa = loso(Fm, ym, gm, model_name)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.bar([str(s) for s in sa], list(sa.values()), color="steelblue"); ax.axhline(1 / K, color="r", ls="--")
    ax.set_xlabel("Held-out subject"); ax.set_ylabel("Accuracy"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("Is it real? Permutation test")
    n_perm = st.slider("Permutations", 20, 300, 100, 20)
    if st.button("Run permutation test"):
        null = permutation_test(Fm, ym, gm, model_name, n_perm); obs = np.mean(fa)
        pval = (np.sum(null >= obs) + 1) / (len(null) + 1)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(null, bins=20, color="grey", label="shuffled labels"); ax.axvline(obs, color="r", lw=2, label=f"real {obs:.1%} (p={pval:.3f})")
        ax.legend(); ax.set_xlabel("Accuracy"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("Sanity check: does accuracy depend on the start of the trial?")
    st.caption("Some words (e.g. Scooter) show a big burst in the first seconds — movement/blink/onset transient. "
               "If accuracy collapses when the start is removed, the classifier was mostly detecting that artifact.")
    if st.button("Run time-window check"):
        wins = [(0, 12), (0, 3), (3, 12), (5, 12), (6, 12)]; out = []
        for a_, b_ in wins:
            Fw = subject_norm(features(X[..., a_ * FS:b_ * FS]), g)[msk]; s_ = []
            for r in range(3):
                for tr, te in cv_splits(Fw, ym, gm, r):
                    s_.append((make_model(model_name).fit(Fw[tr], ym[tr]).predict(Fw[te]) == ym[te]).mean())
            out.append((f"{a_}-{b_}s", np.mean(s_), np.std(s_)))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar([o[0] for o in out], [o[1] for o in out], yerr=[o[2] for o in out], color="steelblue", capsize=3)
        ax.axhline(1 / K, color="r", ls="--", label="chance"); ax.set_ylim(0, 1); ax.legend(); ax.set_ylabel("Accuracy")
        for k, o in enumerate(out): ax.text(k, o[1] + .02, f"{o[1]:.0%}", ha="center")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        Xf2 = bandpass(Xm)
        st.write({c: f"first 2 s RMS {np.sqrt((Xf2[ym == c][..., :256] ** 2).mean()):.0f} µV vs rest {np.sqrt((Xf2[ym == c][..., 256:] ** 2).mean()):.0f} µV" for c in classes})

# ---- Predict tab
with tab_pred:
    pm = st.selectbox("Model", all_names, index=0, key="model_pred")
    mode = st.radio("Input", ["Held-out subject from the dataset", "Upload one EDF file"], horizontal=True)
    if mode.startswith("Held-out"):
        subj = st.selectbox("Hold out subject", sorted(np.unique(gm)))
        tr = gm != subj; te = np.where(~tr)[0]
        mdl = make_model(pm).fit(Fm[tr], ym[tr])
        rows = []
        for i in te:
            pb = mdl.predict_proba(Fm[i:i + 1])[0]; rows.append({"True": ym[i], "Predicted": mdl.classes_[pb.argmax()],
                                                                    **{f"P({c})": round(p, 2) for c, p in zip(mdl.classes_, pb)}})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        pick = st.selectbox("Speak the prediction for trial", range(len(te)), format_func=lambda i: f"{rows[i]['True']} → {rows[i]['Predicted']}")
        if st.button("🔊 Speak"):
            speak(f"The imagined word is {rows[pick]['Predicted']}")
    else:
        st.caption("Upload **one** `.edf` file (Emotiv EPOC, 14 channels, ~12 s). A single file cannot be normalised per subject, "
                   "so this mode uses a model trained on plain (un-normalised) log band-power features. "
                   "That is **much less accurate** than the held-out-subject mode above.")
        fl = st.file_uploader("EDF file", type="edf", accept_multiple_files=False)
        if fl is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    tmp.write(fl.getvalue())
                new_trial = read_trial(tmp.name)[None]
            except Exception as e:
                st.error(f"Could not read this EDF (needs the 14 EPOC channels {CH}): {e}"); st.stop()
            Fraw = features(X[..., t0 * FS:t1 * FS])[msk]                    # no subject normalisation
            Fn = features(new_trial[..., t0 * FS:t1 * FS])
            mdl = make_model(pm).fit(Fraw, ym); pb = mdl.predict_proba(Fn)[0]
            pred = mdl.classes_[pb.argmax()]

            est = np.mean([(make_model(pm).fit(Fraw[tr], ym[tr]).predict(Fraw[te]) == ym[te]).mean()
                           for tr, te in cv_splits(Fraw, ym, gm, 0)])
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted word", pred)
            c2.metric("Confidence", f"{pb.max():.0%}")
            c3.metric("Est. accuracy of this mode", f"{est:.0%}", f"{est - 1 / K:+.0%} vs chance")
            fig, ax = plt.subplots(figsize=(5, 2.6))
            ax.bar(mdl.classes_, pb, color=["seagreen" if c == pred else "lightgrey" for c in mdl.classes_])
            ax.axhline(1 / K, color="r", ls="--", lw=.8); ax.set_ylim(0, 1); ax.set_ylabel("Probability")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            if est < 1 / K + 0.1:
                st.warning("Accuracy in this single-file mode is close to chance for these words — treat the prediction as a guess.")
            if st.button("🔊 Speak", key="spk2"):
                speak(f"The imagined word is {pred}")