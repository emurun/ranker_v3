import knowledge_base
import json

def extract_dataset_questions(n=20):
    """Extract questions and supporting paragraphs from the dataset."""
    subset = knowledge_base.load_musique_subset(n=n)
    data = []
    
    for i in range(len(subset)):
        item = subset[i]
        question_data = {
            'id': item['id'],
            'question': item['question'],
            'answer': item['answer'],
            'supporting_paragraphs': []
        }
        
        # Find supporting paragraphs
        for j, para in enumerate(item['paragraphs']):
            if para.get('is_supporting', False):
                text = para.get('paragraph_text', 'No text')
                # Truncate long paragraphs for readability
                if len(text) > 200:
                    text = text[:200] + '...'
                
                question_data['supporting_paragraphs'].append({
                    'title': para.get('title', 'No title'),
                    'text': text
                })
        
        data.append(question_data)
    
    # Save to file
    with open(f'dataset_questions_n{n}.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Extracted {len(data)} questions and supporting paragraphs to dataset_questions_n{n}.json")
    
    return data

if __name__ == "__main__":
    extract_dataset_questions(20)
