import csv
import io
import os
import random
import zipfile
from fnmatch import fnmatch

import matplotlib.pyplot as plt
import tensorflow as tf


path_to_dataset = '/mimer/NOBACKUP/Datasets/tiny-imagenet-200/tiny-imagenet-200.zip'
with zipfile.ZipFile(path_to_dataset) as archive:
    tiny_imagenet_train_size = len([fn for fn in archive.namelist() if fnmatch(fn, '*train*.JPEG')])
    tiny_imagenet_val_size = len([fn for fn in archive.namelist() if fnmatch(fn, '*val*.JPEG')])

def tiny_imagenet_generator(
    path=path_to_dataset,
    split='train',
    shuffle=True,
):
    with zipfile.ZipFile(path) as archive:
        # Find filename to label mapping
        wnids = archive.read(
            'tiny-imagenet-200/wnids.txt'
        ).decode(
            'utf-8'
        ).split()
        wnid2label = {wnid: [label] for label, wnid in enumerate(wnids)}
        if split == 'val':
            with archive.open(
                'tiny-imagenet-200/val/val_annotations.txt',
            ) as f:
                f = io.TextIOWrapper(f)
                filename2label = {
                    fn: wnid2label[wnid]
                    for fn, wnid, _, _, _, _
                    in csv.reader(f, delimiter='\t')
                }

        # Iterate over images
        namelist = archive.namelist()
        if shuffle:
            random.shuffle(namelist)

        for filename in namelist:
            # Filter for JPEG files and split
            if not fnmatch(filename, f'*{split}*.JPEG'):
                continue

            # Read label
            if split == 'train':
                wnid = filename.split('/')[-1].split('_')[0]
                label = wnid2label[wnid]
            elif split == 'val':
                label = filename2label[os.path.basename(filename)]
            else:
                raise NotImplementedError(f'Reading label not implemented for split {split}.')

            # Read image
            with archive.open(filename) as imgfile:
                img = plt.imread(imgfile) / 255
            if img.ndim == 2:
                # Not all images in tiny-imagenet are RGB valued
                img = img[..., None]
                img = tf.image.grayscale_to_rgb(
                    tf.convert_to_tensor(
                        img,
                        dtype=tf.float32,
                    )
                )

            yield img, label


tiny_imagenet_signature = (
    tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(1), dtype=tf.int32),
)
#tiny_imagenet_train_dataset = tf.data.Dataset.from_generator(
#    generator=tiny_imagenet_generator,
#    output_signature=(
#        tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
#        tf.TensorSpec(shape=(1), dtype=tf.int32),
#    ),
#)


if __name__ == '__main__':
    from functools import partial

    import tensorflow as tf

    train_dataset = tf.data.Dataset.from_generator(
        generator=partial(tiny_imagenet_generator, split='train', shuffle=True),
        output_signature=tiny_imagenet_signature,
    )
    val_dataset = tf.data.Dataset.from_generator(
        generator=partial(tiny_imagenet_generator, split='val', shuffle=True),
        output_signature=tiny_imagenet_signature,
    )

    for dataset in (train_dataset, val_dataset):
        for img, label in val_dataset:
            print(label)
            break
