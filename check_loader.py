import sys, os, csv
sys.path.insert(0, 'customer_service_chatbot_LLM/src')
import importlib
import langchain_helper
importlib.reload(langchain_helper)
print('langchain_helper file:', langchain_helper.__file__)
# call the cached loader
try:
    data = langchain_helper._load_simple_dataset()
    print('loader rows:', len(data))
    if data:
        print('first prompt:', data[0]['prompt'])
except Exception as e:
    print('loader error', e)
# manual load using same path logic
p = os.path.normpath(os.path.join(os.path.dirname(langchain_helper.__file__), '..', 'dataset', 'dataset.csv'))
print('manual path:', p, 'exists', os.path.exists(p))
rows = []
try:
    with open(p, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i>9:
                break
            rows.append(r)
    print('manual rows read:', len(rows))
    if rows:
        print('manual first keys:', list(rows[0].keys()))
        print('manual first prompt snippet:', rows[0].get('prompt')[:80])
except Exception as e:
    print('manual load error', e)
