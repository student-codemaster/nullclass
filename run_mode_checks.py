import sys, os
sys.path.insert(0, 'customer_service_chatbot_LLM/src')

results = []

def try_call(desc, fn):
    try:
        v = fn()
        print(f"OK: {desc} -> {type(v)}")
        results.append((desc, 'ok', str(type(v))))
    except Exception as e:
        print(f"ERR: {desc} -> {e}")
        results.append((desc, 'err', str(e)))

# langchain_helper
import importlib
import langchain_helper
importlib.reload(langchain_helper)
try_call('langchain_helper._load_simple_dataset()', lambda: langchain_helper._load_simple_dataset())
try:
    chain = langchain_helper.get_qa_chain()
    # call chain with a sample question
    try:
        out = chain('What is the duration of the bootcamp?')
        print('langchain_helper.get_qa_chain call returned type', type(out))
        results.append(('langchain_helper.get_qa_chain.call', 'ok', type(out).__name__))
    except Exception as e:
        print('langchain_helper.get_qa_chain call error', e)
        results.append(('langchain_helper.get_qa_chain.call', 'err', str(e)))
except Exception as e:
    print('get_qa_chain error', e)
    results.append(('get_qa_chain', 'err', str(e)))

# interaction_store
try:
    import interaction_store as isdb
    isdb.init_db()
    rowid = isdb.log_interaction('test_user', 'hello', 'hi', source=[{'source':'test'}])
    recent = isdb.get_recent(1)
    print('interaction_store OK, last id', rowid, 'recent count', len(recent))
    results.append(('interaction_store', 'ok', f'id={rowid}'))
except Exception as e:
    print('interaction_store error', e)
    results.append(('interaction_store', 'err', str(e)))

# medical_qa
try:
    from modes.medical_qa import medical_retrieval as med
    ents = med.extract_medical_entities('I have fever and cough and take ibuprofen')
    print('medical_retrieval.extract_medical_entities', ents)
    results.append(('medical_retrieval', 'ok', str(ents)))
except Exception as e:
    print('medical_retrieval error', e)
    results.append(('medical_retrieval', 'err', str(e)))

# multilingual
try:
    from modes import multilingual
    # call detect_and_translate if present
    if hasattr(multilingual, 'detect_and_translate'):
        try:
            out = multilingual.detect_and_translate('Hello')
            print('multilingual.detect_and_translate ->', out)
            results.append(('multilingual.detect_and_translate', 'ok', str(type(out))))
        except Exception as e:
            print('multilingual.detect_and_translate error', e)
            results.append(('multilingual.detect_and_translate', 'err', str(e)))
    else:
        print('multilingual module loaded (no detect_and_translate)')
        results.append(('multilingual', 'ok', 'loaded'))
except Exception as e:
    print('multilingual import error', e)
    results.append(('multilingual', 'err', str(e)))

# multimodal
try:
    from modes.multimodal import handler as mmh
    print('multimodal handler loaded')
    results.append(('multimodal', 'ok', 'loaded'))
except ImportError as e:
    print('multimodal import error (module not found):', e)
    results.append(('multimodal', 'err', f'ImportError: {e}'))
except Exception as e:
    print('multimodal error', e)
    results.append(('multimodal', 'err', str(e)))

# research_assistant
try:
    from modes.research_assistant import arxiv_loader as al
    # call load_arxiv_csv with a non-existing file to ensure error handled
    try:
        al.load_arxiv_csv('nonexistent.csv')
        results.append(('arxiv_loader.load', 'err', 'unexpected success'))
    except FileNotFoundError:
        print('arxiv_loader.load_arxiv_csv correctly raised FileNotFoundError')
        results.append(('arxiv_loader', 'ok', 'FileNotFoundError raised as expected'))
    except Exception as e:
        print('arxiv_loader unexpected error', e)
        results.append(('arxiv_loader', 'err', str(e)))
except Exception as e:
    print('arxiv_loader import error', e)
    results.append(('arxiv_loader', 'err', str(e)))

# dynamic_updater
try:
    from modes.dynamic_updater import updater as up
    print('dynamic_updater module loaded')
    results.append(('dynamic_updater', 'ok', 'loaded'))
except Exception as e:
    print('dynamic_updater error', e)
    results.append(('dynamic_updater', 'err', str(e)))

print('\nSUMMARY:')
for r in results:
    print('-', r)
