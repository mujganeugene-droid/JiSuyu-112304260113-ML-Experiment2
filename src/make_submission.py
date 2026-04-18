from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def _resolve_data_file(data_dir: Path, base_name: str) -> Path | None:
    candidates = [
        data_dir / base_name,
        data_dir / base_name / base_name,  # e.g. ./labeledTrainData.tsv/labeledTrainData.tsv
        data_dir / "data" / base_name,
        data_dir / "input" / base_name,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


_NON_LETTERS = re.compile(r"[^a-zA-Z]")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def _review_to_words(review_html: str) -> list[str]:
    text = BeautifulSoup(review_html, "lxml").get_text(" ")
    text = _NON_LETTERS.sub(" ", text)
    words = text.lower().split()
    return words


def _review_to_sentences(review_html: str) -> list[list[str]]:
    text = BeautifulSoup(review_html, "lxml").get_text(" ")
    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    sentences: list[list[str]] = []
    for sentence in raw_sentences:
        words = _NON_LETTERS.sub(" ", sentence).lower().split()
        if words:
            sentences.append(words)
    return sentences


def _review_to_text(review_html: str) -> str:
    text = BeautifulSoup(review_html, "lxml").get_text(" ")
    text = _NON_LETTERS.sub(" ", text)
    return " ".join(text.lower().split())


def _avg_word_vectors(tokenized_reviews: list[list[str]], keyed_vectors, vector_size: int) -> np.ndarray:
    features = np.zeros((len(tokenized_reviews), vector_size), dtype=np.float32)
    vocab = keyed_vectors.key_to_index

    for i, words in enumerate(tokenized_reviews):
        vectors = []
        for w in words:
            if w in vocab:
                vectors.append(keyed_vectors[w])
        if vectors:
            features[i] = np.mean(vectors, axis=0)
    return features


@dataclass(frozen=True)
class DataPaths:
    labeled: Path
    test: Path | None
    unlabeled: Path | None


def _find_data_paths(data_dir: Path) -> DataPaths:
    labeled = _resolve_data_file(data_dir, "labeledTrainData.tsv")
    if labeled is None:
        raise FileNotFoundError(
            "找不到 labeledTrainData.tsv。请把数据放到 --data-dir 指定目录下，"
            "或保持类似 ./labeledTrainData.tsv/labeledTrainData.tsv 的结构。"
        )

    test = _resolve_data_file(data_dir, "testData.tsv")
    unlabeled = _resolve_data_file(data_dir, "unlabeledTrainData.tsv")
    return DataPaths(labeled=labeled, test=test, unlabeled=unlabeled)


def _read_tsv(path: Path) -> pd.DataFrame:
    # Kaggle 原始数据常用 quoting=3 来忽略引号转义
    df = pd.read_csv(path, header=0, delimiter="\t", quoting=3)
    # quoting=3 会保留外层引号（例如 id 变成 "5814_8"），这里统一剥掉最外层一对引号
    for col in ("id", "review"):
        if col in df.columns and df[col].dtype == object:
            s = df[col].astype(str)
            mask = s.str.startswith('"') & s.str.endswith('"') & (s.str.len() >= 2)
            df[col] = s.where(~mask, s.str.slice(1, -1))
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Kaggle submission for IMDB sentiment (AUC).")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing TSV files.")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"), help="Output submission csv path.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Where to save models.")

    parser.add_argument("--max-train-rows", type=int, default=None, help="Debug: limit labeled rows.")
    parser.add_argument("--max-unlabeled-rows", type=int, default=None, help="Debug: limit unlabeled rows.")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Debug: limit test rows.")
    parser.add_argument("--cv-folds", type=int, default=0, help="If >1, run StratifiedKFold CV AUC on labeled set.")

    parser.add_argument(
        "--features",
        choices=(
            "w2v",
            "tfidf_word",
            "tfidf_char",
            "tfidf_both",
            "nbsvm_word",
            "nbsvm_char",
            "nbsvm_both",
            "ensemble",
            "blend_tfidf_nbsvm",
            "stack",
        ),
        default="w2v",
        help="Feature set to use.",
    )
    parser.add_argument(
        "--ensemble-alpha",
        type=float,
        default=0.5,
        help="Ensemble weight: alpha*tfidf + (1-alpha)*w2v. Only used when --features ensemble.",
    )
    parser.add_argument(
        "--ensemble-search",
        action="store_true",
        help="Search best alpha on CV OOF (if --cv-folds>1) or on validation split (otherwise).",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help="Blend weight: alpha*tfidf_both + (1-alpha)*nbsvm_both. Only used when --features blend_tfidf_nbsvm.",
    )
    parser.add_argument(
        "--blend-search",
        action="store_true",
        help="Search best blend alpha on CV OOF (if --cv-folds>1) or on validation split (otherwise).",
    )

    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-unlabeled", action="store_true", help="Do not use unlabeledTrainData.tsv even if present.")
    parser.add_argument("--tfidf-max-features", type=int, default=200000)
    parser.add_argument(
        "--stack-models",
        type=str,
        default="tfidf_both_c2,tfidf_both_c8,tfidf_word13,tfidf_char36,nbsvm_both",
    )
    parser.add_argument("--nb-max-features", type=int, default=200000)
    parser.add_argument("--nb-c", type=float, default=4.0)
    parser.add_argument("--nb-estimator", choices=("sgd", "lr"), default="sgd")
    parser.add_argument("--nb-word-ngram-max", type=int, default=2)
    parser.add_argument("--nb-char-ngram-min", type=int, default=3)
    parser.add_argument("--nb-char-ngram-max", type=int, default=4)
    parser.add_argument("--nb-max-iter", type=int, default=20)

    args = parser.parse_args()

    stack_models = [m.strip() for m in args.stack_models.split(",") if m.strip()]
    need_text = args.features in (
        "tfidf_word",
        "tfidf_char",
        "tfidf_both",
        "nbsvm_word",
        "nbsvm_char",
        "nbsvm_both",
        "ensemble",
        "blend_tfidf_nbsvm",
        "stack",
    )
    need_w2v = args.features in ("w2v", "ensemble") or (args.features == "stack" and "w2v" in stack_models)

    paths = _find_data_paths(args.data_dir)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] labeled: {paths.labeled}")
    print(f"[data] test: {paths.test if paths.test else '(not found)'}")
    print(f"[data] unlabeled: {paths.unlabeled if paths.unlabeled else '(not found)'}")

    train = _read_tsv(paths.labeled)
    if args.max_train_rows:
        train = train.head(args.max_train_rows)
    if "review" not in train.columns or "sentiment" not in train.columns:
        raise ValueError("labeledTrainData.tsv 必须包含 review 和 sentiment 两列。")

    test = _read_tsv(paths.test) if paths.test else None
    if test is not None and args.max_test_rows:
        test = test.head(args.max_test_rows)

    unlabeled = None
    if (not args.no_unlabeled) and paths.unlabeled:
        unlabeled = _read_tsv(paths.unlabeled)
        if args.max_unlabeled_rows:
            unlabeled = unlabeled.head(args.max_unlabeled_rows)

    # -------- Model & Features --------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import FeatureUnion, Pipeline

    y = train["sentiment"].astype(int).to_numpy()

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def _search_alpha(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> tuple[float, float]:
        best_alpha = 0.5
        best_auc = -1.0
        for a in np.linspace(0.0, 1.0, 21):
            auc_val = roc_auc_score(y_true, a * p_a + (1.0 - a) * p_b)
            if auc_val > best_auc:
                best_auc = auc_val
                best_alpha = float(a)
        return best_alpha, float(best_auc)

    def _fit_predict_w2v(train_idx: np.ndarray, valid_idx: np.ndarray):
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import StandardScaler

        w2v_clf = SkPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None)),
            ]
        )
        w2v_clf.fit(X_w2v[train_idx], y[train_idx])
        valid_proba = w2v_clf.predict_proba(X_w2v[valid_idx])[:, 1]
        return w2v_clf, valid_proba

    def _fit_predict_tfidf(train_idx: np.ndarray, valid_idx: np.ndarray):
        from sklearn.feature_extraction.text import TfidfVectorizer

        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=args.tfidf_max_features,
            dtype=np.float32,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=args.tfidf_max_features,
            dtype=np.float32,
        )

        if args.features == "tfidf_word":
            features = word
        elif args.features == "tfidf_char":
            features = char
        else:
            features = FeatureUnion([("word", word), ("char", char)])

        tfidf_clf = Pipeline(
            steps=[
                ("tfidf", features),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=4000,
                        solver="saga",
                        n_jobs=None,
                        random_state=args.seed,
                        C=4.0,
                    ),
                ),
            ]
        )
        tfidf_clf.fit(np.array(X_text, dtype=object)[train_idx], y[train_idx])
        valid_proba = tfidf_clf.predict_proba(np.array(X_text, dtype=object)[valid_idx])[:, 1]
        return tfidf_clf, valid_proba

    def _score_1d(est, X_any) -> np.ndarray:
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X_any)[:, 1]
        if hasattr(est, "decision_function"):
            return _sigmoid(np.asarray(est.decision_function(X_any), dtype=np.float64))
        return _sigmoid(np.asarray(est.predict(X_any), dtype=np.float64))

    def _make_tfidf_pipeline(kind: str, *, word_ngram_max: int = 2, char_ngram=(3, 5), C: float = 4.0):
        from sklearn.feature_extraction.text import TfidfVectorizer

        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, word_ngram_max),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=args.tfidf_max_features,
            dtype=np.float32,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=args.tfidf_max_features,
            dtype=np.float32,
        )
        if kind == "word":
            feats = word
        elif kind == "char":
            feats = char
        else:
            feats = FeatureUnion([("word", word), ("char", char)])

        return Pipeline(
            steps=[
                ("tfidf", feats),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=4000,
                        solver="saga",
                        n_jobs=None,
                        random_state=args.seed,
                        C=C,
                    ),
                ),
            ]
        )

    def _nbsvm_pipeline(kind: str):
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import SGDClassifier

        class _NBSVMScaler(BaseEstimator, TransformerMixin):
            def __init__(self, alpha: float = 1.0):
                self.alpha = alpha

            def fit(self, X, y_fit):
                X_csr = X.tocsr()
                y_arr = np.asarray(y_fit)
                pos = np.asarray(X_csr[y_arr == 1].sum(axis=0)).ravel() + self.alpha
                neg = np.asarray(X_csr[y_arr == 0].sum(axis=0)).ravel() + self.alpha
                self.r_ = np.log(pos / neg)
                return self

            def transform(self, X):
                return X.multiply(self.r_)

        word_vec = CountVectorizer(
            analyzer="word",
            ngram_range=(1, args.nb_word_ngram_max),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            max_features=args.nb_max_features,
            binary=True,
        )
        char_vec = CountVectorizer(
            analyzer="char_wb",
            ngram_range=(args.nb_char_ngram_min, args.nb_char_ngram_max),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            max_features=args.nb_max_features,
            binary=True,
        )

        if args.nb_estimator == "lr":
            estimator = LogisticRegression(
                max_iter=4000,
                solver="saga",
                n_jobs=None,
                random_state=args.seed,
                C=args.nb_c,
            )
        else:
            # Much faster for large sparse features
            estimator = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-5,
                max_iter=args.nb_max_iter,
                tol=1e-3,
                random_state=args.seed,
            )

        if kind == "word":
            feats = word_vec
        elif kind == "char":
            feats = char_vec
        else:
            feats = FeatureUnion(
                [
                    ("word", Pipeline([("vec", word_vec), ("nb", _NBSVMScaler())])),
                    ("char", Pipeline([("vec", char_vec), ("nb", _NBSVMScaler())])),
                ]
            )
            return Pipeline(
                steps=[
                    ("feats", feats),
                    ("clf", estimator),
                ]
            )

        return Pipeline(
            steps=[
                ("vec", feats),
                ("nb", _NBSVMScaler()),
                ("clf", estimator),
            ]
        )

    def _svm_tfidf_pipeline(kind: str, *, word_ngram_max: int = 2, char_ngram=(3, 5)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier

        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, word_ngram_max),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=args.tfidf_max_features,
            dtype=np.float32,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=args.tfidf_max_features,
            dtype=np.float32,
        )
        if kind == "word":
            feats = word
        elif kind == "char":
            feats = char
        else:
            feats = FeatureUnion([("word", word), ("char", char)])

        svm = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-5,
            max_iter=30,
            tol=1e-3,
            random_state=args.seed,
        )
        return Pipeline([("tfidf", feats), ("svm", svm)])

    if need_w2v:
        # -------- Word2Vec --------
        try:
            from gensim.models import Word2Vec
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("缺少 gensim：请先 pip install -r requirements.txt") from exc

        w2v_path = (
            args.artifacts_dir
            / f"word2vec_vs{args.vector_size}_w{args.window}_mc{args.min_count}_e{args.epochs}.model"
        )
        legacy_w2v = args.artifacts_dir / "word2vec.model"
        if w2v_path.exists():
            print(f"[w2v] loading cached model: {w2v_path}")
            w2v = Word2Vec.load(str(w2v_path))
        elif legacy_w2v.exists():
            print(f"[w2v] loading cached model: {legacy_w2v}")
            w2v = Word2Vec.load(str(legacy_w2v))
        else:
            sentences: list[list[str]] = []
            print("[w2v] tokenizing sentences...")
            for review_html in tqdm(train["review"].tolist(), desc="train"):
                sentences.extend(_review_to_sentences(review_html))
            if unlabeled is not None and "review" in unlabeled.columns:
                for review_html in tqdm(unlabeled["review"].tolist(), desc="unlabeled"):
                    sentences.extend(_review_to_sentences(review_html))
            if test is not None and "review" in test.columns:
                for review_html in tqdm(test["review"].tolist(), desc="test"):
                    sentences.extend(_review_to_sentences(review_html))

            print(f"[w2v] sentences: {len(sentences)}")
            print("[w2v] training...")
            w2v = Word2Vec(
                sentences=sentences,
                vector_size=args.vector_size,
                window=args.window,
                min_count=args.min_count,
                workers=args.workers,
                seed=args.seed,
            )
            w2v.train(sentences, total_examples=len(sentences), epochs=args.epochs)
            w2v.save(str(w2v_path))
            print(f"[w2v] saved: {w2v_path}")

        keyed_vectors = w2v.wv

        print("[feat] averaging word vectors (w2v)...")
        X_tokens = [_review_to_words(r) for r in tqdm(train["review"].tolist(), desc="train->words")]
        X_w2v = _avg_word_vectors(X_tokens, keyed_vectors, args.vector_size)

        test_X_w2v = None
        if test is not None:
            test_tokens = [_review_to_words(r) for r in tqdm(test["review"].tolist(), desc="test->words")]
            test_X_w2v = _avg_word_vectors(test_tokens, keyed_vectors, args.vector_size)

    if need_text:
        print("[feat] cleaning text (tfidf)...")
        X_text = [_review_to_text(r) for r in tqdm(train["review"].tolist(), desc="train->text")]
        test_X_text = None
        if test is not None:
            test_X_text = [_review_to_text(r) for r in tqdm(test["review"].tolist(), desc="test->text")]

    if args.features == "w2v":
        # -------- Eval --------
        fit_X = X_w2v
        test_X = test_X_w2v

        from sklearn.preprocessing import StandardScaler

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None)),
            ]
        )

    elif args.features in ("tfidf_word", "tfidf_char", "tfidf_both"):
        from sklearn.feature_extraction.text import TfidfVectorizer

        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=200000,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=200000,
        )

        if args.features == "tfidf_word":
            features = word
        elif args.features == "tfidf_char":
            features = char
        else:
            features = FeatureUnion([("word", word), ("char", char)])

        clf = Pipeline(
            steps=[
                ("tfidf", features),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=4000,
                        solver="saga",
                        n_jobs=None,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        fit_X = X_text
        test_X = test_X_text

    elif args.features in ("nbsvm_word", "nbsvm_char", "nbsvm_both"):
        kind = "word" if args.features == "nbsvm_word" else "char" if args.features == "nbsvm_char" else "both"
        clf = _nbsvm_pipeline(kind)
        fit_X = X_text
        test_X = test_X_text

    elif args.features == "blend_tfidf_nbsvm":
        if args.cv_folds and args.cv_folds > 1 and args.blend_search:
            print(f"[cv] {args.cv_folds}-fold OOF for blend...")
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
            oof_tfidf = np.zeros(len(y), dtype=np.float64)
            oof_nb = np.zeros(len(y), dtype=np.float64)
            tfidf_pipeline = None
            nb_pipeline = None
            for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
                args.features = "tfidf_both"
                tfidf_pipeline, p_t = _fit_predict_tfidf(tr, va)
                args.features = "blend_tfidf_nbsvm"
                nb_pipeline = _nbsvm_pipeline("both")
                nb_pipeline.fit(np.array(X_text, dtype=object)[tr], y[tr])
                p_n = nb_pipeline.predict_proba(np.array(X_text, dtype=object)[va])[:, 1]
                oof_tfidf[va] = p_t
                oof_nb[va] = p_n
                print(
                    f"[cv] fold {fold}: tfidf={roc_auc_score(y[va], p_t):.5f} "
                    f"nbsvm={roc_auc_score(y[va], p_n):.5f}"
                )
            auc_t = roc_auc_score(y, oof_tfidf)
            auc_n = roc_auc_score(y, oof_nb)
            alpha, auc_b = _search_alpha(y, oof_tfidf, oof_nb)
            print(f"[cv] oof tfidf AUC: {auc_t:.5f}")
            print(f"[cv] oof nbsvm AUC: {auc_n:.5f}")
            print(f"[cv] best alpha={alpha:.2f} blend AUC: {auc_b:.5f}")
            args.blend_alpha = alpha

        idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=args.seed, stratify=y)
        args.features = "tfidf_both"
        tfidf_model, p_t = _fit_predict_tfidf(idx_tr, idx_va)
        nb_model = _nbsvm_pipeline("both")
        nb_model.fit(np.array(X_text, dtype=object)[idx_tr], y[idx_tr])
        p_n = nb_model.predict_proba(np.array(X_text, dtype=object)[idx_va])[:, 1]
        alpha = float(args.blend_alpha)
        if args.blend_search and not (args.cv_folds and args.cv_folds > 1):
            alpha, best_auc = _search_alpha(y[idx_va], p_t, p_n)
            args.blend_alpha = alpha
            print(f"[valid] searched best alpha={alpha:.2f} blend AUC: {best_auc:.5f}")
        else:
            auc_b = roc_auc_score(y[idx_va], alpha * p_t + (1.0 - alpha) * p_n)
            print(f"[valid] blend alpha={alpha:.2f} AUC: {auc_b:.5f}")
        print(f"[valid] tfidf AUC: {roc_auc_score(y[idx_va], p_t):.5f}")
        print(f"[valid] nbsvm AUC: {roc_auc_score(y[idx_va], p_n):.5f}")

        if test is None:
            print("[submit] testData.tsv not found; skip submission generation.")
            return 0
        if test_X_text is None:
            raise RuntimeError("internal error: test features not prepared")

        print("[model] fitting tfidf_both on full labeled set...")
        args.features = "tfidf_both"
        tfidf_model.fit(np.array(X_text, dtype=object), y)
        print("[model] fitting nbsvm_both on full labeled set...")
        nb_model.fit(np.array(X_text, dtype=object), y)

        p_test_t = tfidf_model.predict_proba(np.array(test_X_text, dtype=object))[:, 1]
        p_test_n = nb_model.predict_proba(np.array(test_X_text, dtype=object))[:, 1]
        test_proba = args.blend_alpha * p_test_t + (1.0 - args.blend_alpha) * p_test_n

        submission = pd.DataFrame({"id": test["id"], "sentiment": test_proba})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.out, index=False)
        print(f"[submit] wrote: {args.out}")
        return 0

    elif args.features == "stack":
        models = stack_models
        if not models:
            raise ValueError("--stack-models 不能为空")
        if not (args.cv_folds and args.cv_folds > 1):
            raise ValueError("stack 模式需要 --cv-folds > 1（用于OOF stacking）")

        def make_model(name: str):
            if name == "tfidf_word":
                return _make_tfidf_pipeline("word"), "text"
            if name == "tfidf_char":
                return _make_tfidf_pipeline("char"), "text"
            if name == "tfidf_both":
                return _make_tfidf_pipeline("both"), "text"
            if name == "tfidf_word13":
                return _make_tfidf_pipeline("word", word_ngram_max=3), "text"
            if name == "tfidf_char36":
                return _make_tfidf_pipeline("char", char_ngram=(3, 6)), "text"
            if name == "tfidf_both_c2":
                return _make_tfidf_pipeline("both", C=2.0), "text"
            if name == "tfidf_both_c8":
                return _make_tfidf_pipeline("both", C=8.0), "text"
            if name == "svm_word":
                return _svm_tfidf_pipeline("word"), "text"
            if name == "svm_char":
                return _svm_tfidf_pipeline("char", char_ngram=(3, 6)), "text"
            if name == "svm_both":
                return _svm_tfidf_pipeline("both", char_ngram=(3, 6)), "text"
            if name == "nbsvm_word":
                return _nbsvm_pipeline("word"), "text"
            if name == "nbsvm_char":
                return _nbsvm_pipeline("char"), "text"
            if name == "nbsvm_both":
                return _nbsvm_pipeline("both"), "text"
            if name == "w2v":
                from sklearn.pipeline import Pipeline as SkPipeline
                from sklearn.preprocessing import StandardScaler

                return (
                    SkPipeline(
                        steps=[
                            ("scaler", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None)),
                        ]
                    ),
                    "w2v",
                )
            raise ValueError(f"未知 stacking base model: {name}")

        if test is None:
            print("[submit] testData.tsv not found; skip submission generation.")
            return 0

        X_text_arr = None
        if any(m in {"tfidf_word", "tfidf_char", "tfidf_both", "nbsvm_word", "nbsvm_char", "nbsvm_both"} for m in models):
            if "X_text" not in locals():
                raise RuntimeError("internal error: text features not prepared")
            X_text_arr = np.array(X_text, dtype=object)
            test_X_text_arr = np.array(test_X_text, dtype=object) if test_X_text is not None else None
            if test_X_text_arr is None:
                raise RuntimeError("internal error: test text not prepared")

        if "w2v" in models:
            if "X_w2v" not in locals() or test_X_w2v is None:
                raise RuntimeError("internal error: w2v features not prepared")

        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        oof = np.zeros((len(y), len(models)), dtype=np.float64)

        print(f"[stack] base models: {models}")
        for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
            for j, name in enumerate(models):
                est, kind = make_model(name)
                if kind == "text":
                    est.fit(X_text_arr[tr], y[tr])
                    p = _score_1d(est, X_text_arr[va])
                else:
                    est.fit(X_w2v[tr], y[tr])
                    p = _score_1d(est, X_w2v[va])
                oof[va, j] = p
            fold_auc = roc_auc_score(y[va], oof[va].mean(axis=1))
            print(f"[stack] fold {fold}: mean-proba AUC={fold_auc:.5f}")

        meta = LogisticRegression(max_iter=2000, solver="lbfgs")
        meta.fit(oof, y)
        meta_oof = meta.predict_proba(oof)[:, 1]
        print(f"[stack] meta OOF AUC: {roc_auc_score(y, meta_oof):.5f}")

        test_level1 = np.zeros((len(test), len(models)), dtype=np.float64)
        for j, name in enumerate(models):
            est, kind = make_model(name)
            if kind == "text":
                est.fit(X_text_arr, y)
                test_level1[:, j] = _score_1d(est, test_X_text_arr)
            else:
                est.fit(X_w2v, y)
                test_level1[:, j] = _score_1d(est, test_X_w2v)

        test_proba = meta.predict_proba(test_level1)[:, 1]
        submission = pd.DataFrame({"id": test["id"], "sentiment": test_proba})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.out, index=False)
        print(f"[submit] wrote: {args.out}")
        return 0

    else:
        # -------- Ensemble: tfidf_both + w2v --------
        if args.ensemble_search and args.cv_folds and args.cv_folds > 1:
            print(f"[cv] {args.cv_folds}-fold OOF for ensemble...")
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

            oof_tfidf = np.zeros(len(y), dtype=np.float64)
            oof_w2v = np.zeros(len(y), dtype=np.float64)

            for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
                args.features = "tfidf_both"
                tfidf_model, p_t = _fit_predict_tfidf(tr, va)
                args.features = "ensemble"
                _, p_w = _fit_predict_w2v(tr, va)
                oof_tfidf[va] = p_t
                oof_w2v[va] = p_w
                print(
                    f"[cv] fold {fold}: tfidf={roc_auc_score(y[va], p_t):.5f} "
                    f"w2v={roc_auc_score(y[va], p_w):.5f}"
                )

            auc_tfidf = roc_auc_score(y, oof_tfidf)
            auc_w2v = roc_auc_score(y, oof_w2v)
            alpha, auc_ens = _search_alpha(y, oof_tfidf, oof_w2v)
            print(f"[cv] oof tfidf AUC: {auc_tfidf:.5f}")
            print(f"[cv] oof w2v   AUC: {auc_w2v:.5f}")
            print(f"[cv] best alpha={alpha:.2f} ens AUC: {auc_ens:.5f}")
            args.ensemble_alpha = alpha

        # quick hold-out validation for a sanity AUC
        train_idx, valid_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=args.seed, stratify=y
        )
        args.features = "tfidf_both"
        tfidf_model, proba_tfidf = _fit_predict_tfidf(train_idx, valid_idx)
        args.features = "ensemble"
        w2v_model, proba_w2v = _fit_predict_w2v(train_idx, valid_idx)
        auc_t = roc_auc_score(y[valid_idx], proba_tfidf)
        auc_w = roc_auc_score(y[valid_idx], proba_w2v)
        alpha = float(args.ensemble_alpha)
        if args.ensemble_search and (not (args.cv_folds and args.cv_folds > 1)):
            alpha, best_auc = _search_alpha(y[valid_idx], proba_tfidf, proba_w2v)
            args.ensemble_alpha = alpha
            print(f"[valid] searched best alpha={alpha:.2f} ens AUC: {best_auc:.5f}")
        else:
            auc_e = roc_auc_score(y[valid_idx], alpha * proba_tfidf + (1.0 - alpha) * proba_w2v)
            print(f"[valid] ens alpha={alpha:.2f} AUC: {auc_e:.5f}")
        print(f"[valid] tfidf AUC: {auc_t:.5f}")
        print(f"[valid] w2v   AUC: {auc_w:.5f}")

        # fit both on full data and combine test probabilities
        print("[model] fitting tfidf_both on full labeled set...")
        args.features = "tfidf_both"
        tfidf_model.fit(np.array(X_text, dtype=object), y)
        print("[model] fitting w2v on full labeled set...")
        args.features = "ensemble"
        w2v_model.fit(X_w2v, y)

        if test is None:
            print("[submit] testData.tsv not found; skip submission generation.")
            return 0
        if test_X_text is None or test_X_w2v is None:
            raise RuntimeError("internal error: test features not prepared")

        p_test_t = tfidf_model.predict_proba(np.array(test_X_text, dtype=object))[:, 1]
        p_test_w = w2v_model.predict_proba(test_X_w2v)[:, 1]
        test_proba = args.ensemble_alpha * p_test_t + (1.0 - args.ensemble_alpha) * p_test_w

        submission = pd.DataFrame({"id": test["id"], "sentiment": test_proba})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.out, index=False)
        print(f"[submit] wrote: {args.out}")
        return 0

    # -------- Eval --------
    X_train, X_valid, y_train, y_valid = train_test_split(
        fit_X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    clf.fit(X_train, y_train)
    valid_proba = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_proba)
    print(f"[valid] AUC: {auc:.5f}")

    if args.cv_folds and args.cv_folds > 1:
        print(f"[cv] {args.cv_folds}-fold AUC...")
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        scores = cross_val_score(clf, fit_X, y, cv=cv, scoring="roc_auc", n_jobs=None)
        print(f"[cv] mean={scores.mean():.5f} std={scores.std():.5f}")

    # -------- Fit full --------
    print("[model] fitting on full labeled set...")
    clf.fit(fit_X, y)

    model_path = args.artifacts_dir / f"model_{args.features}.pkl"
    try:
        import joblib

        joblib.dump(clf, model_path)
        print(f"[model] saved: {model_path}")
    except Exception:
        print("[model] joblib not available; skip saving sklearn model.")

    # -------- Submission --------
    if test is None:
        print("[submit] testData.tsv not found; skip submission generation.")
        return 0

    if "review" not in test.columns or "id" not in test.columns:
        raise ValueError("testData.tsv 必须包含 id 和 review 两列。")

    if test_X is None:
        raise RuntimeError("internal error: test features not prepared")
    test_proba = clf.predict_proba(test_X)[:, 1]

    submission = pd.DataFrame({"id": test["id"], "sentiment": test_proba})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"[submit] wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
