"""
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
[2]: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from utils import plot_images


def get_train_valid_loader(batch_size,
                           augment,
                           valid_size=0.1,
                           show_sample=False,
                           num_workers=4):
   
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    train_split = 100 - int(valid_size * 100)

    (train_ds, val_ds), info = tfds.load(
        "cifar10",
        split=[f'train[:{train_split}%]', f'train[{train_split}%:]'],
        with_info=True,
        as_supervised=True
    )

    normalize = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x,y: (normalize(x), y), num_parallel_calls=num_workers)
    val_ds = val_ds.map(lambda x,y: (normalize(x), y), num_parallel_calls=num_workers)

    if augment:
        transform = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.Resizing(32, 32, crop_to_aspect_ratio=(4,4))
        ])
        train_ds = train_ds.map(lambda x,y: (transform(x), y), num_parallel_calls=num_workers)
    


    train_ds = train_ds.cache().shuffle(buffer_size=int(info.splits['train'].num_examples * valid_size))
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.cache().shuffle(buffer_size=info.splits['train'].num_examples)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


    # visualize some images
    if show_sample:
        data_iter = next(iter(train_ds))
        images, labels = data_iter
        plot_images(images.numpy(), labels)

    return (train_ds, val_ds)


def get_test_loader(batch_size,
                    num_workers=4):
    
    normalize = tf.keras.layers.Rescaling(1./255)

    test_ds = tfds.load(
        "cifar10",
        split=['test'],
        with_info=False,
        as_supervised=True
    )

    test_ds = test_ds.map(lambda x,y: (normalize(x), y), num_parallel_calls=num_workers)
    test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return test_ds
