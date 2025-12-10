import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import pronouncing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


WINDOW = 60
NA_LABEL = "NA"
MAX_LINE_LEN = 120
MIN_WORDS = 4
MIN_SECTION_LINES = 4


def preprocess_suffix(text: str) -> str:
    """Return the normalized ending for rhyme comparison."""
    if not isinstance(text, str):
        return ""

    tokens = re.findall(r"[A-Za-z']+", text)
    if not tokens:
        return ""

    token = tokens[-1].lower().strip("'")
    token = re.sub(r"[^a-z]", "", token)
    if not token:
        return ""

    phones = pronouncing.phones_for_word(token)
    if phones:
        rhyme = pronouncing.rhyming_part(phones[0])
        if rhyme:
            return rhyme

    if len(token) >= 3:
        return token[-3:]
    return token


def letter_to_number(letter: str) -> int:
    value = 0
    for ch in letter:
        if not ("A" <= ch <= "Z"):
            continue
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value


def number_to_letter(number: int) -> str:
    if number <= 0:
        return "A"
    chars = []
    while number:
        number -= 1
        chars.append(chr(number % 26 + ord("A")))
        number //= 26
    return "".join(reversed(chars))


def is_poetic_line(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > MAX_LINE_LEN:
        return False
    if len(stripped.split()) < MIN_WORDS:
        return False
    if any(ch in stripped for ch in {'"', "“", "”"}):
        return False
    return True


def group_poem_sections(texts: Sequence[str]) -> List[List[int]]:
    sections: List[List[int]] = []
    current: List[int] = []
    for idx, text in enumerate(texts):
        if is_poetic_line(text):
            current.append(idx)
        else:
            if len(current) >= MIN_SECTION_LINES:
                sections.append(current)
            current = []
    if len(current) >= MIN_SECTION_LINES:
        sections.append(current)
    return sections


def find_neighbor(idx: int, suffix: str, assigned: Dict[int, str], suffixes: Sequence[str], window: int) -> Optional[int]:
    for offset in range(1, window + 1):
        for neighbor in (idx - offset, idx + offset):
            if neighbor < 0 or neighbor >= len(suffixes):
                continue
            if neighbor in assigned and suffixes[neighbor] == suffix:
                return neighbor
    return None


def find_future_unlabeled(idx: int, suffix: str, assigned: Dict[int, str], suffixes: Sequence[str], window: int) -> Optional[int]:
    upper = min(len(suffixes) - 1, idx + window)
    for neighbor in range(idx + 1, upper + 1):
        if neighbor not in assigned and suffixes[neighbor] == suffix:
            return neighbor
    return None


def detect_rhyme_letters(df: pd.DataFrame, window: int = WINDOW, seed_gold: bool = False) -> List[str]:
    predictions = [NA_LABEL] * len(df)
    sections = group_poem_sections(df["text"].tolist())
    next_letter_value = 1

    for section in sections:
        if not section:
            continue

        rhymes: Dict[str, tuple[str, int]] = {}
        assigned: Dict[int, str] = {}
        suffixes: List[str] = []
        section_max_letter = 0

        for local_idx, global_idx in enumerate(section):
            text = df.at[global_idx, "text"]
            suffix = preprocess_suffix(text)
            suffixes.append(suffix)

            if seed_gold:
                label = str(df.at[global_idx, "rhyme"]).strip().upper()
                if label and label != NA_LABEL:
                    assigned[local_idx] = label
                    rhymes.setdefault(suffix, (label, len(section) - 1))
                    section_max_letter = max(section_max_letter, letter_to_number(label))

        if seed_gold and section_max_letter:
            next_letter_value = max(next_letter_value, section_max_letter + 1)

        for local_idx, global_idx in enumerate(section):
            if local_idx in assigned:
                predictions[global_idx] = assigned[local_idx]
                continue

            suffix = suffixes[local_idx]
            if not suffix:
                predictions[global_idx] = NA_LABEL
                continue

            if suffix in rhymes:
                letter, expiry = rhymes[suffix]
                if local_idx <= expiry:
                    assigned[local_idx] = letter
                    predictions[global_idx] = letter
                    if local_idx >= expiry:
                        del rhymes[suffix]
                    continue
                del rhymes[suffix]

            neighbor = find_neighbor(local_idx, suffix, assigned, suffixes, window)
            if neighbor is not None:
                letter = assigned[neighbor]
                assigned[local_idx] = letter
                current_letter, expiry = rhymes.get(suffix, (letter, local_idx))
                rhymes[suffix] = (letter, max(expiry, neighbor if neighbor > local_idx else local_idx))
                predictions[global_idx] = letter
                continue

            unlabeled_match = find_future_unlabeled(local_idx, suffix, assigned, suffixes, window)
            if unlabeled_match is not None:
                letter = number_to_letter(next_letter_value)
                next_letter_value += 1
                assigned[local_idx] = letter
                assigned[unlabeled_match] = letter
                rhymes[suffix] = (letter, unlabeled_match)
                predictions[global_idx] = letter
                continue

            predictions[global_idx] = NA_LABEL

    return predictions


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, names=["text", "device", "rhyme"])
    df = df.fillna({"text": "", "device": NA_LABEL, "rhyme": NA_LABEL})
    df["device"] = df["device"].astype(str)
    df["rhyme"] = df["rhyme"].astype(str).str.upper()
    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    labeled_mask = df["rhyme"] != NA_LABEL
    if labeled_mask.sum() == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = df.loc[labeled_mask, "rhyme"]
    y_pred = df.loc[labeled_mask, "predicted_rhyme"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign rhyme-scheme letters based on suffix matching.")
    parser.add_argument("--input", type=Path, default=Path("data/nlp_annots.tsv"), help="Path to the TSV dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/nlp_annots_with_rhyme.tsv"), help="Where to save the predictions TSV.")
    parser.add_argument("--window", type=int, default=WINDOW, help="Line search window when pairing rhymes.")
    parser.add_argument(
        "--seed-gold-labels",
        action="store_true",
        help="If set, seed the detector with existing gold labels (useful for regeneration rather than evaluation).",
    )
    args = parser.parse_args()

    df = load_dataset(args.input)
    predictions = detect_rhyme_letters(df, window=args.window, seed_gold=args.seed_gold_labels)
    df["predicted_rhyme"] = predictions

    df.to_csv(args.output, sep="\t", index=False)

    total = len(df)
    labeled_total = int((df["rhyme"] != NA_LABEL).sum())
    newly_assigned = sum(
        1 for original, predicted in zip(df["rhyme"], predictions) if original == NA_LABEL and predicted != NA_LABEL
    )
    metrics = compute_metrics(df)

    print(f"Processed {total} lines.")
    print(f"Existing rhyme labels: {labeled_total}")
    print(f"Newly assigned rhyme labels: {newly_assigned}")
    print(f"Accuracy (gold-labeled subset): {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro): {metrics['recall']:.4f}")
    print(f"F1 (macro): {metrics['f1']:.4f}")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
