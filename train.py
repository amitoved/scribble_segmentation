import os

import numpy as np
from tensorflow.keras import optimizers
from tqdm import tqdm

import constants
from models.architectures import unet2d
from utils.utils import get_paths

pool_folder = os.path.join(constants.DATA_DIR, 'pool')


def data_generator(pool_folder, batch_size=1):
    image_filenames = [filename for filename in os.listdir(pool_folder) if 'image' in filename]
    indices = [filename.split('image_')[1].split('.npy')[0] for filename in image_filenames]
    sample_image_path = get_paths(pool_folder, indices[0])[0]
    sample_image = np.load(sample_image_path)
    n_rows, n_cols = sample_image.shape
    x = np.zeros((batch_size, n_rows, n_cols, 1))
    y = np.zeros((batch_size, n_rows, n_cols, constants.n_classes))
    while True:
        for i in range(batch_size):
            found_non_empty_scibble = False
            while found_non_empty_scibble == False:
                index = np.random.choice(indices)
                image_path, _, scribble_path = get_paths(pool_folder, index)
                scribble = np.load(scribble_path)
                if np.any(scribble):
                    x[i, :, :, 0] = np.load(image_path)
                    y[i] = scribble.astype(int)
                    found_non_empty_scibble = True

        target_rows = max(128, 2 ** np.ceil(np.log2(n_rows)))
        target_cols = max(128, 2 ** np.ceil(np.log2(n_cols)))
        rows_to_add = int(target_rows - n_rows)
        cols_to_add = int(target_cols - n_cols)
        x_ = np.pad(x, ((0, 0), (0, rows_to_add), (0, cols_to_add), (0, 0)), mode='constant', constant_values=-1024)
        y_ = np.pad(y, ((0, 0), (0, rows_to_add), (0, cols_to_add), (0, 0)), mode='constant', constant_values=0)
        yield x_, y_


model = unet2d()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(1e-4))
training_generator = data_generator(pool_folder, batch_size=1)

while True:
    model.fit_generator(training_generator, steps_per_epoch=4, epochs=10)
    image_paths = [os.path.join(pool_folder, file) for file in os.listdir(pool_folder) if 'image' in file]
    pred_paths = [image_path.replace('image', 'pred') for image_path in image_paths]
    for image_path, pred_path in tqdm(zip(image_paths, pred_paths)):
        image = np.load(image_path).squeeze()
        n_rows, n_cols = image.shape
        target_rows = max(128, 2 ** np.ceil(np.log2(n_rows)))
        target_cols = max(128, 2 ** np.ceil(np.log2(n_cols)))
        rows_to_add = int(target_rows - n_rows)
        cols_to_add = int(target_cols - n_cols)
        image = np.pad(image, ((0, rows_to_add), (0, cols_to_add)), mode='constant', constant_values=-1024)
        pred = model.predict(image[None, :, :, None])[0]
        pred = pred[:n_rows, :n_cols, :]
        np.save(arr=pred, file=pred_path)
