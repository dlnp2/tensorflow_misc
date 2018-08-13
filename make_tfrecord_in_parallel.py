"""
Make a .tfrecord file in parallel via multiprocessing
"""

import os
import time
from multiprocessing import Queue, Process
import pandas as pd
import tensorflow as tf


def make_chunks(data, n_procs, data_len):
    """Make `n_procs`-fold chunks"""
    batch_size = data_len // n_procs
    batches = [data[batch_size*i: batch_size*(i+1)] for i in range(n_procs-1)]
    batches.append(data[batch_size*(n_procs-1):])
    return batches


def make_example(img, label):
    """Make an `example` protobuf"""
    img_proto = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))
    label_proto = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    return tf.train.Example(features=tf.train.Features(feature={
            "image": img_proto,
            "label": label_proto
        }))


def image_reader(img_paths, labels, img_size, q):
    """Create and put data protobufs into the queue reading image files"""
    ipath = tf.placeholder(tf.string)
    size = tf.placeholder(tf.int32)
    img = tf.read_file(ipath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, [size, size])

    # Invoke one session per worker process
    with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
        for img_path, label in zip(img_paths, labels):
            resized_img = sess.run(img, feed_dict={ipath: img_path, size: img_size})
            example = make_example(resized_img, label)
            q.put(example)
    print("Done producing image protos @ {}".format(os.getpid()))


def write_to_tfrecord(img_paths, labels, out_path, img_size, n_prod_process):
    print("Start writing data with {} producer processes".format(n_prod_process))
    q = Queue()

    # Activate worker processes
    data_len = len(img_paths)
    img_batches = make_chunks(img_paths, n_prod_process, data_len)
    label_batches = make_chunks(labels, n_prod_process, data_len)
    prod_ps = [Process(target=image_reader,
                       args=(img_batches[i], label_batches[i], img_size, q))
               for i in range(n_prod_process)]
    for p in prod_ps:
        p.daemon = True
        p.start()

    # Consume tasks in the queue in this process
    get_count = 0
    span = 100
    writer = tf.python_io.TFRecordWriter(out_path)
    s = time.time()
    s_total = time.time()
    while True:
        if get_count == data_len:
            break
        if q.empty():
            time.sleep(0.1)
            continue
        example = q.get()
        writer.write(example.SerializeToString())
        get_count += 1
        if get_count % span == 0:
            e = time.time()
            print("Done {:,} images. Mean writing time: {:.04f} s."
                  .format(get_count, (e - s) / span))
            s = e

    # Finalize
    writer.close()
    for p in prod_ps:
        p.join()
    print("Done writing all ({:,} of {:,}) images in {:.02f} seconds."
          .format(get_count, data_len, time.time() - s_total))


def main(img_label_file, out_path, img_size=256, n_processes=1):
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # Here we assume we have Image and Label columns in this file.
    # Image: paths to each image files
    # Label: corresponding class label integers
    img_label = pd.read_csv(img_label_file)
    write_to_tfrecord(
        img_label.Image, img_label.Label, out_path, img_size, n_processes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("img_label_file", type=str,
                        help="a csv file containing image paths (Image column) "
                             "and labels (Label column)")
    parser.add_argument("out_path", type=str,
                        help="path to the output .tfrecord")
    parser.add_argument("--img_size", type=int, default=256,
                        help="resize width and height")
    parser.add_argument("--n_processes", type=int, default=1,
                        help="#processes in which each image file is read")
    args = parser.parse_args()
    main(args.img_label_file, args.out_path, img_size=args.img_size,
         n_processes=args.n_processes)
