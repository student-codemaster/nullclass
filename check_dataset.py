import os
p = os.path.normpath(os.path.join(os.path.dirname(__file__), 'customer_service_chatbot_LLM', 'dataset', 'dataset.csv'))
print('computed path:', p)
print('exists:', os.path.exists(p))
if os.path.exists(p):
    print('size:', os.path.getsize(p))
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f'line {i+1}:', line.strip())
