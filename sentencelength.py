import re

def split_text_into_sentences(text):
    text = text.strip()
    text = re.sub(r'\n+', '', text)  # Completely remove all newlines
    text = text.replace('\r', ' ')
    text = text.replace('-', ' ')
    text = re.sub(r'\.\.+', '.', text)

    # Split the text into sentences using ., ?, and ! as delimiters
    sentences = re.split(r'(?<=[.!?:]) +', text)
    return sentences

def join_short_sentences(sentences, min_words=10):
    result = []
    buffer = ""

    for sentence in sentences:
        if len(sentence.split()) < min_words:
            # Replace ? ! . with a comma
            sentence = re.sub(r'[?!\.]', ',', sentence)
            buffer += " " + sentence
        else:
            if buffer:
                result.append(buffer.strip() + " " + sentence)
                buffer = ""
            else:
                result.append(sentence)
    
    if buffer:
        result.append(buffer.strip())
    
    return result

def split_long_sentence(sentence, max_words=30):
    # Split the sentence into words
    words = sentence.split()
    
    # Check if the sentence has more than `max_words`
    if len(words) <= max_words:
        return [sentence]  # Return the sentence as is if it is not too long
    
    # Find the best split point near a comma, colon, or semicolon
    split_point = len(words) // 2
    for i in range(split_point, len(words)):
        if words[i].endswith(',') or words[i].endswith(':') or words[i].endswith(';'):
            split_point = i + 1
            break
    else:
        for i in range(split_point, -1, -1):
            if words[i].endswith(',') or words[i].endswith(':') or words[i].endswith(';'):
                split_point = i + 1
                break
    
    # Split the sentence into two parts and add a period at the end of the first part
    first_part = ' '.join(words[:split_point]) + '.'
    second_part = ' '.join(words[split_point:])
    
    return [first_part, second_part]

def sentence_length(text, min_words=10, max_words=20):
    # Step 1: Split the text into sentences
    sentences = split_text_into_sentences(text)
    
    # Step 2: Join short sentences
    sentences = join_short_sentences(sentences, min_words)
    
    # Step 3: Split long sentences
    final_sentences = []
    for sentence in sentences:
        split_sentences = split_long_sentence(sentence, max_words)
        final_sentences.extend(split_sentences)
    
    # Merge processed sentences into a single text
    merged_text = '\n '.join(final_sentences)
    return merged_text

# Example usage:
text = """
Trung ương yêu cầu nhanh chóng triển khai đề án tiếp tục sắp xếp, không tổ chức công an cấp huyện.
"""

# Process the text
processed_sentences = sentence_length(text)


# Print the merged text
print(processed_sentences)
