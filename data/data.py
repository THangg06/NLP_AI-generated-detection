from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


LABEL_MAP = {"human": 0, "ai": 1}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Filter human/ai labels, map label ids, show stats, PCA plots, and wordclouds."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/315k_dataset_ai_detection.csv"),
		help="Input CSV file path.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/outputs"),
		help="Directory to store filtered CSV and generated plots.",
	)
	parser.add_argument(
		"--max-pca-samples",
		type=int,
		default=12000,
		help="Maximum rows used for PCA visualization to control memory/time.",
	)
	parser.add_argument(
		"--max-features",
		type=int,
		default=3000,
		help="Max TF-IDF features for PCA.",
	)
	parser.add_argument(
		"--max-wordcloud-samples",
		type=int,
		default=20000,
		help="Maximum rows per class used to build each wordcloud.",
	)
	parser.add_argument(
		"--ai-limit",
		type=int,
		default=10000,
		help="Maximum number of rows kept for label 'ai'.",
	)
	parser.add_argument(
		"--human-limit",
		type=int,
		default=10000,
		help="Maximum number of rows kept for label 'human'.",
	)
	return parser.parse_args()


def load_and_filter(input_path: Path, ai_limit: int, human_limit: int) -> pd.DataFrame:
	df = pd.read_csv(input_path)

	if "text" not in df.columns or "label" not in df.columns:
		raise ValueError("CSV must contain 'text' and 'label' columns.")

	df = df.copy()
	df["label"] = df["label"].astype(str).str.strip().str.lower()
	df = df[df["label"].isin(LABEL_MAP.keys())].copy()

	if df.empty:
		raise ValueError("No rows with label 'human' or 'ai' were found.")

	per_class_limits = {"ai": ai_limit, "human": human_limit}
	sampled_frames = []
	for label_name, limit in per_class_limits.items():
		label_df = df[df["label"] == label_name]
		if label_df.empty:
			continue
		if limit is not None and limit > 0:
			label_df = label_df.sample(n=min(limit, len(label_df)), random_state=42)
		sampled_frames.append(label_df)

	if not sampled_frames:
		raise ValueError("No rows left after applying ai/human limits.")

	df = pd.concat(sampled_frames, ignore_index=True)

	df["label_id"] = df["label"].map(LABEL_MAP).astype(int)
	df["text"] = df["text"].astype(str).fillna("")
	return df


def print_stats(df: pd.DataFrame) -> None:
	print("=== DATASET STATS ===")
	print(f"Total rows (human + ai): {len(df):,}")
	print("\nLabel counts:")
	print(df["label"].value_counts())
	print("\nLabel percentages (%):")
	print((df["label"].value_counts(normalize=True) * 100).round(2))

	text_lengths = df["text"].str.split().str.len()
	print("\nText length stats (words):")
	print(text_lengths.describe().round(2))


def make_pca_plots(
	df: pd.DataFrame,
	output_dir: Path,
	max_pca_samples: int,
	max_features: int,
) -> None:
	pca_df = df.sample(n=min(max_pca_samples, len(df)), random_state=42).copy()

	vectorizer = TfidfVectorizer(
		max_features=max_features,
		stop_words="english",
		ngram_range=(1, 2),
	)
	x_sparse = vectorizer.fit_transform(pca_df["text"])
	x_dense = x_sparse.toarray()

	pca2 = PCA(n_components=2, random_state=42)
	comps2 = pca2.fit_transform(x_dense)

	plt.figure(figsize=(9, 7))
	for label_name, label_id in LABEL_MAP.items():
		mask = pca_df["label_id"].values == label_id
		plt.scatter(comps2[mask, 0], comps2[mask, 1], s=10, alpha=0.55, label=label_name)
	plt.title("PCA 2D (TF-IDF)")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "pca_2d.png", dpi=200)
	plt.close()

	pca3 = PCA(n_components=3, random_state=42)
	comps3 = pca3.fit_transform(x_dense)

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")
	for label_name, label_id in LABEL_MAP.items():
		mask = pca_df["label_id"].values == label_id
		ax.scatter(
			comps3[mask, 0],
			comps3[mask, 1],
			comps3[mask, 2],
			s=10,
			alpha=0.5,
			label=label_name,
		)
	ax.set_title("PCA 3D (TF-IDF)")
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	ax.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "pca_3d.png", dpi=200)
	plt.close()


def make_wordclouds(df: pd.DataFrame, output_dir: Path, max_wordcloud_samples: int) -> None:
	for label_name in LABEL_MAP:
		label_df = df[df["label"] == label_name]
		if len(label_df) > max_wordcloud_samples:
			label_df = label_df.sample(n=max_wordcloud_samples, random_state=42)

		text_blob = " ".join(label_df["text"].astype(str).tolist())
		wc = WordCloud(
			width=1400,
			height=800,
			background_color="white",
			colormap="viridis",
			max_words=300,
		).generate(text_blob)

		plt.figure(figsize=(12, 7))
		plt.imshow(wc, interpolation="bilinear")
		plt.axis("off")
		plt.title(f"WordCloud - {label_name}")
		plt.tight_layout()
		plt.savefig(output_dir / f"wordcloud_{label_name}.png", dpi=200)
		plt.close()


def main() -> None:
	args = parse_args()

	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	df = load_and_filter(args.input, args.ai_limit, args.human_limit)
	print_stats(df)

	filtered_csv = output_dir / "human_ai_filtered.csv"
	df.to_csv(filtered_csv, index=False)

	make_pca_plots(df, output_dir, args.max_pca_samples, args.max_features)
	make_wordclouds(df, output_dir, args.max_wordcloud_samples)

	print("\nSaved files:")
	print(f"- {filtered_csv}")
	print(f"- {output_dir / 'pca_2d.png'}")
	print(f"- {output_dir / 'pca_3d.png'}")
	print(f"- {output_dir / 'wordcloud_human.png'}")
	print(f"- {output_dir / 'wordcloud_ai.png'}")


if __name__ == "__main__":
	main()
