
import csv
import os
import sys

def load_news_data(filepath):
    news_ids = set()
    titles = set()
    count = 0
    # Use pandas or just standard csv with tab delimiter
    # Reading manually to be robust against quoting issues if any, although csv module is best.
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                nid = parts[0]
                title = parts[3]
                news_ids.add(nid)
                titles.add(title)
                count += 1
    return news_ids, titles, count

def check_overlap(train_path, test_path):
    print(f"Loading Train: {train_path}")
    train_ids, train_titles, train_count = load_news_data(train_path)
    print(f"Train Count: {train_count}")

    print(f"Loading Test: {test_path}")
    test_ids, test_titles, test_count = load_news_data(test_path)
    print(f"Test Count: {test_count}")

    # ID Overlap
    id_overlap = train_ids.intersection(test_ids)
    print(f"\n--- Overlap Analysis ---")
    print(f"News ID Overlap Count: {len(id_overlap)}")
    if len(test_ids) > 0:
        print(f"News ID Overlap % (of Test): {len(id_overlap) / len(test_ids) * 100:.2f}%")
    
    # Title Overlap
    title_overlap = train_titles.intersection(test_titles)
    print(f"Title Overlap Count: {len(title_overlap)}")
    if len(test_titles) > 0:
        print(f"Title Overlap % (of Test): {len(title_overlap) / len(test_titles) * 100:.2f}%")

    if len(id_overlap) > 0:
        print("\nSample Overlapping IDs:", list(id_overlap)[:5])
    
    if len(title_overlap) > 0:
        print("Sample Overlapping Titles:", list(title_overlap)[:5])

if __name__ == "__main__":
    preferred_root = "data/work/adressa_one_week_mind_final"
    legacy_root = "adressa_one_week_mind_final"
    data_root = preferred_root
    if os.path.isdir(legacy_root) and not os.path.isdir(preferred_root):
        data_root = legacy_root
    train_file = f"{data_root}/train/news.tsv"
    test_file = f"{data_root}/test/news.tsv"
    check_overlap(train_file, test_file)
