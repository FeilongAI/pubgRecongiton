"""
Check the structure and content of features_db.pkl
"""

import pickle
import os

db_path = 'dataset/features_db.pkl'

if not os.path.exists(db_path):
    print(f"ERROR: {db_path} not found!")
    exit(1)

print(f"Loading {db_path}...")
print(f"File size: {os.path.getsize(db_path) / (1024**3):.2f} GB")
print()

with open(db_path, 'rb') as f:
    db = pickle.load(f)

print("Features DB loaded successfully!")
print(f"Type: {type(db)}")
print()

if isinstance(db, dict):
    print(f"Number of keys: {len(db.keys())}")
    print(f"Keys: {list(db.keys())[:10]}")  # Show first 10 keys

    # Check structure of first item
    first_key = list(db.keys())[0]
    first_val = db[first_key]
    print()
    print(f"First item key: {first_key}")
    print(f"First item value type: {type(first_val)}")
    if hasattr(first_val, 'shape'):
        print(f"First item shape: {first_val.shape}")
elif isinstance(db, tuple) or isinstance(db, list):
    print(f"DB is a {type(db)} with {len(db)} elements")
    for i, elem in enumerate(db):
        print(f"Element {i}: type={type(elem)}, ", end='')
        if hasattr(elem, 'shape'):
            print(f"shape={elem.shape}")
        elif isinstance(elem, (list, dict)):
            print(f"len={len(elem)}")
        else:
            print()
else:
    print(f"Unexpected type: {type(db)}")
    if hasattr(db, 'shape'):
        print(f"Shape: {db.shape}")