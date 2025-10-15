import string
def read_and_clean_file(filename):
    
    try:
        with open(filename,'r') as file:
            text = file.read()
            text = text.lower()
            
            text = text.translate(str.maketrans('','',string.punctuation))
            
            words = text.split()
            return words
    except FileNotFoundError:
        print(f"Error: The file '{filename} was not found")
        return []
    
def count_frequencies(words):
    dict = {}
    for word in words:
        dict[word] = dict.get(word,0) + 1
    return dict

def display_top_words(word_counts, n=10):
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    print(f"\n--- Top {n} Most Frequent Words ---")
    for word, count in sorted_words[:n]:
        print(f"{word}:{count}")
        
def main():
    filename = 'sample.txt'
    words = read_and_clean_file(filename)
    if words:
        word_counts = count_frequencies(words)
        display_top_words(word_counts)

if __name__ == "__main__":
    main()