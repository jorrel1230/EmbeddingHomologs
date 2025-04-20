import numpy as np
import faiss
import pandas as pd

train_dir = '/scratch/gpfs/jr8867/main/db/family-split-train-test/train'
test_dir = '/scratch/gpfs/jr8867/main/db/family-split-train-test/test'

train_embeddings = np.load(f'{train_dir}/train_embeddings.npy')
test_embeddings = np.load(f'{test_dir}/test_embeddings.npy')

train_metadata = pd.read_csv(f'{train_dir}/train_metadata.csv')
test_metadata = pd.read_csv(f'{test_dir}/test_metadata.csv')

embeddings_full = np.concatenate((train_embeddings, test_embeddings), axis=0)
metadata_full = pd.concat([train_metadata, test_metadata], axis=0)

# FAISS Indexing
index = faiss.IndexFlatL2(embeddings_full.shape[1])
index.add(embeddings_full)
faiss.write_index(index, "/scratch/gpfs/jr8867/main/db/indices/baseline/baseline.index")

np.save("/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_embeddings.npy", embeddings_full)
metadata_full.to_csv("/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_metadata.csv", index=False)

print("Saved Embeddings, Metadata, and FAISS index!")


index = faiss.IndexFlatL2(train_embeddings.shape[1])
index.add(train_embeddings)
faiss.write_index(index, "/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_train.index")

np.save("/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_train_embeddings.npy", train_embeddings)
train_metadata.to_csv("/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_train_metadata.csv", index=False)


index = faiss.IndexFlatL2(test_embeddings.shape[1])
index.add(test_embeddings)
faiss.write_index(index, "/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_test.index")

np.save("/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_test_embeddings.npy", test_embeddings)
test_metadata.to_csv("/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_test_metadata.csv", index=False)


print("Saved Train and Test Embeddings, Metadata, and FAISS index!")

