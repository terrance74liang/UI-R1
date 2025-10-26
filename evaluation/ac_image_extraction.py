import tensorflow as tf
import os
import json
from pathlib import Path
from tqdm import tqdm
import re

def episode_parser(path:str):
    with open(Path(path),'r') as eps:
        data = json.load(eps)
        episode_list = [x['image'] for x in data]

    def episode_screenshot_extract(x:str):
        numbers = re.search('episode_(.+?)-screenshot_(.+?).png', x)
        if numbers:
            return numbers.group(1,2)
        else: 
            return None
    
    return list(map(episode_screenshot_extract,episode_list))


def _parse_fn(example_proto):
    feature_description = {
        'screenshots': tf.io.VarLenFeature(tf.string),
        'episode_id' : tf.io.FixedLenFeature([1], tf.int64)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def filter_wrap(allowed_ids):
    def _filter_fn(example):
        return tf.reduce_any(tf.equal(example['episode_id'][0], allowed_ids))
    return _filter_fn


def run(input_path = '/home/teliang/scratch/android_control',output_path = '/home/teliang/scratch/android_control_data',episode_path = '/home/teliang/scratch/UI-R1/dataset/ac_test.json'):

    data_path = input_path
    subfiles = os.listdir(data_path)
    output_dir = Path(output_path)

    parsed_episodes = episode_parser(episode_path)
    episodes, _ = tf.unique(tf.constant([int(x[0]) for x in parsed_episodes], dtype=tf.int64),out_idx=tf.int64)

    for file in tqdm(subfiles):
        if 'android' not in file:
            continue

        gzip_dataset = tf.data.TFRecordDataset([data_path + '/' +file],compression_type = 'GZIP').map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE).filter(filter_wrap(episodes))

        for record in iter(gzip_dataset):

            screenshots = tf.sparse.to_dense(record['screenshots'])

            parsed_episode_id = record['episode_id'].numpy()[0]

            for i in range(len(screenshots)):
                screenshot_number = [int(x[1]) for x in parsed_episodes if int(x[0]) == parsed_episode_id]
                p = Path(output_dir)

                if i in screenshot_number:
                    ep_folder_name = p.joinpath(f'episode_{parsed_episode_id}-screenshot_{i}.png')
                    tf.io.write_file(str(ep_folder_name), screenshots[i])
 

if __name__ == '__main__':
    run()

