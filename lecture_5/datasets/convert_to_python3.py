import pickle
import os
import shutil

DATASET_DIR = 'cifar-10-batches-py'
OUTPUT_DIR = 'cifar-10-batches-py3'

def convert_batch_to_python3_format(batch_path):
    with open(batch_path, 'rb') as bfile:
        batch = pickle.load(bfile, encoding='bytes')
        converted_batch = {key.decode(): value for key, value in batch.items()}
        converted_batch['filenames'] = [x.decode() for x in batch[b'filenames']]
        converted_batch['batch_label'] = batch[b'batch_label'].decode()
        return converted_batch

if __name__ == '__main__':
    print('converting cifar10 to python3 format')

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for f in os.scandir(DATASET_DIR):
        if 'batch' in f.name and 'meta' not in f.name:
            converted_batch = convert_batch_to_python3_format(f.path)
            with open(os.path.join(OUTPUT_DIR, f.name), 'wb') as converted_batch_file:
                pickle.dump(converted_batch, converted_batch_file)
        elif 'meta' in f.name:
            shutil.copy(f.path, OUTPUT_DIR)

