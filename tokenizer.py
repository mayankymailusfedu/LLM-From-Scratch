# --- 1. Sampling Data Setup ---
sampling_data = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

print(f"Training Data: {sampling_data}")


# --- 2. Initial Vocabulary Creation ---
unique_chars = set()
for doc in sampling_data:
    # Iterate through every character in the text to build a set of unique characters.
    for char in doc:
        unique_chars.add(char)

vocab = list(unique_chars)
# Define a special token to mark the end of a word, which is crucial for separation.
end_of_word = "</w>"
vocab.append(end_of_word)
vocab.sort()

print(f"\nInitial vocabulary: {vocab}")
print(f"Vocabulary size: {len(vocab)}")


# --- 3. Word Splitting and Frequency Counting ---
word_splits = {}
for doc in sampling_data:
    # Split the document into a list of words using space as the delimiter.
    words = doc.split(' ')
    for word in words:
        if word:
            # Create the token unit: [word characters] + [end_of_word marker].
            char_list = list(word) + [end_of_word]
            word_tuple = tuple(char_list)
            # Count the occurrence of this specific word tuple.
            if word_tuple not in word_splits:
                word_splits[word_tuple] = 0
            word_splits[word_tuple] += 1

print(f"\nPre-tokenized word frequencies: {word_splits}")


# --- 4. Helper Function: Get Pair Statistics ---
def get_pair_stats(splits):
    # This function calculates the frequency of adjacent character pairs (bigrams) 
    # within the tokenized words.
    pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        # Iterate through the symbols to find consecutive pairs.
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] += freq
    return pair_counts


# pair_stats = get_pair_stats(word_splits)
# print(f"Pair Counts: {pair_stats}")


# --- 5. Helper Function: Merge Pairs (for BPE) ---
def merge_pairs(pairs_to_merge, splits):
    # Merges a specific pair (e.g., ('a', 'b')) into a single new token ('ab').
    new_splits = {}
    (first, second) = pairs_to_merge
    merged_token = first + second

    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        # Iterate through the current word tuple to apply the merge rule.
        while i < len(symbols):
            # Check if the current sequence matches the pair being merged.
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                # If a match is found, add the new merged token and skip both input characters.
                new_symbols.append(merged_token)
                i = i + 2
            else:
                # If no match, just append the current symbol.
                new_symbols.append(symbols[i])
                i = i + 1
        # Store the resulting tuple with its frequency.
        new_splits[tuple(new_symbols)] = freq
    return new_splits


# --- 6. Byte Pair Encoding (BPE) Execution ---
num_merges = 15  # The number of times the merging process will be run.
merges = {}      # Stores the mapping: (pair) -> (merged_token).
current_splits = word_splits.copy()  # Start with the initial tokenized words.

print(f"\nStart Byte Pair Encoding Merges")

for i in range(num_merges):
    print(f"\nMerge Iteration {i + 1}/{num_merges}")

    # Calculate pair frequencies based on the current token splits.
    pair_stats = get_pair_stats(current_splits)

    if not pair_stats:
        print(f"No more pairs to merge.")
        break

    # Sort pairs by frequency in descending order to select the most common pair first.
    sorted_pairs = sorted(pair_stats.items(), key=lambda item: item[1], reverse=True)
    print(f"Top {min(5, len(sorted_pairs))} pair frequencies: {sorted_pairs[:5]}")

    # Select the most frequent pair found.
    best_pair = max(pair_stats, key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Found best pairs: {best_pair} with frequency: {best_freq}")

    # Perform the merge operation to reduce vocabulary size.
    current_splits = merge_pairs(best_pair, current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"Merge {best_pair} into {new_token}")
    print(f"Split after merge: {current_splits}")

    # Record the successful merge rule for later use.
    merges[best_pair] = new_token
    print(f"Updated merges: {merges}")


print(f"\nByte Pair Encoding Merges Completed")

# --- 7. Final Results ---
print(f"\nFinal vocabulary: {vocab}")
print(f"Final vocabulary size: {len(vocab)}")

final_vocab_sort = sorted(list(set(vocab)))
print(f"Final vocabulary sorted: {final_vocab_sort}")

print(f"\nLearned merges: {merges}")
print(f"Final vocabulary sorted: ")