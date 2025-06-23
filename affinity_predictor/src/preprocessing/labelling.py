import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
input_pkl = "../../data/02_intermediate/processed_pdbs.pkl"
output_train_pkl = "../../05_model_input/train_labelled_processed_pdbs.pkl"
output_valid_pkl = "../../05_model_input/valid_labelled_processed_pdbs.pkl"

mu = 0.0
sigma = 125.0  # 95% of values will fall between -250 and +250
min_affinity_value = 1e-5  # to avoid log(0) in neglog_aff computation

# === LOAD ===
with open(input_pkl, 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} items from {input_pkl}")

# === GENERATE FAKE AFFINITIES ===
for item in data:
    raw_affinity = np.random.normal(loc=mu, scale=sigma)
    affinity_for_log = max(abs(raw_affinity), min_affinity_value)
    neglog_aff = -np.log(affinity_for_log)

    item['label'] = float(neglog_aff) # probably unnecessary!
    item['affinity'] = {'neglog_aff': float(neglog_aff)}

# === TRAIN-TEST SPLIT ===
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

# === SAVE SPLITS ===
with open(output_train_pkl, 'wb') as f:
    pickle.dump(train_data, f)
with open(output_valid_pkl, 'wb') as f:
    pickle.dump(test_data, f)
