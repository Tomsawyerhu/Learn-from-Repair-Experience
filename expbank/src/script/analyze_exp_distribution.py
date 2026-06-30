import os
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def main():
    # Setting paths based on the script location: f:\software\projects\VulAgentEvolver\expbank\src\script
    # We want to access: f:\software\projects\VulAgentEvolver\expbank\src\exp_json
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    exp_json_dir = os.path.join(src_dir, 'exp_json')
    output_dir = os.path.join(exp_json_dir, 'anay')

    if not os.path.exists(exp_json_dir):
        print(f"Error: Directory {exp_json_dir} does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Group data by (template_used, model_name)
    grouped_data = defaultdict(lambda: {'scores': [], 'tags': [], 'lengths': []})

    files = [f for f in os.listdir(exp_json_dir) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files in {exp_json_dir}")

    for filename in files:
        filepath = os.path.join(exp_json_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Get grouping keys
                template = data.get('template_used', 'unknown_template')
                model = data.get('model_name', 'unknown_model')
                
                group = grouped_data[(template, model)]

                # Extract score
                if 'score' in data:
                    group['scores'].append(data['score'])
                
                # Extract tags
                if 'tag' in data:
                    t = data['tag']
                    if isinstance(t, list):
                        group['tags'].extend(t)
                    elif isinstance(t, str):
                        group['tags'].append(t)
                
                # Extract experience length
                if 'experience' in data and isinstance(data['experience'], str):
                    group['lengths'].append(len(data['experience']))

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    # Process each group
    for (template, model), data in grouped_data.items():
        print(f"Processing group: Template={template}, Model={model}")
        
        # Create a safe prefix for filenames
        safe_template = "".join([c for c in template if c.isalnum() or c in ('-', '_')]).strip()
        safe_model = "".join([c for c in model if c.isalnum() or c in ('-', '_')]).strip()
        prefix = f"{safe_template}_{safe_model}_"
        
        scores = data['scores']
        tags = data['tags']
        lengths = data['lengths']

        # 1. Analyze Score Distribution
        if scores:
            plt.figure(figsize=(10, 6))
            score_counts = Counter(scores)
            sorted_scores = sorted(score_counts.items()) 
            keys = [str(k) for k, v in sorted_scores]
            values = [v for k, v in sorted_scores]
            
            plt.bar(keys, values, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of Scores ({template} - {model})')
            plt.xlabel('Score')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            output_path_score = os.path.join(output_dir, f'{prefix}score_distribution.png')
            plt.savefig(output_path_score)
            plt.close()
            print(f"  Score distribution saved to {output_path_score}")
        else:
            print(f"  No scores found for {template}/{model}.")

        # 2. Analyze Tag Distribution
        if tags:
            plt.figure(figsize=(12, 8))
            tag_counts = Counter(tags)
            sorted_tags = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
            
            plt.bar(sorted_tags.keys(), sorted_tags.values(), color='lightgreen', edgecolor='black')
            plt.title(f'Distribution of Tags ({template} - {model})')
            plt.xlabel('Tag')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            output_path_tag = os.path.join(output_dir, f'{prefix}tag_distribution.png')
            plt.savefig(output_path_tag)
            plt.close()
            print(f"  Tag distribution saved to {output_path_tag}")
        else:
            print(f"  No tags found for {template}/{model}.")
            
        # 3. Analyze Statistics (Avg Length)
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        stats = {
            "template_used": template,
            "model_name": model,
            "sample_count": len(lengths),
            "average_experience_length": avg_length
        }
        
        output_path_stats = os.path.join(output_dir, f'{prefix}statistics.json')
        with open(output_path_stats, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"  Statistics saved to {output_path_stats}")

if __name__ == "__main__":
    main()
