from sklearn.model_selection import train_test_split
import pickle

input_pkl = '../../data/02_intermediate/processed_pdbs.pkl'
output_train_pkl = '../../data/05_model_input/train_set.pkl'
output_valid_pkl = '../../data/05_model_input/valid_set.pkl'

# === LOAD ===
with open(input_pkl, 'rb') as f:
    data = pickle.load(f)

# === TRAIN-TEST SPLIT ===
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

# === SAVE SPLITS ===
with open(output_train_pkl, 'wb') as f:
    pickle.dump(train_data, f)
with open(output_valid_pkl, 'wb') as f:
    pickle.dump(test_data, f)

print(f"[DONE] Saved train set to {output_train_pkl}")
print(f"[DONE] Saved test set to {output_valid_pkl}")