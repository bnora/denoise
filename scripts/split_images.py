import random
import argparse
from pathlib import Path
import json
import dataclasses


@dataclasses.dataclass
class DataSplit:
    train: list[str]
    val: list[str]
    test: list[str]


parser = argparse.ArgumentParser(description='create fashionMNIST_data split')
parser.add_argument('--split', type=int, nargs='+', default=[80, 10, 10],
                    help='Percentage Train/validation/test')
parser.add_argument('--fashionMNIST_data-dir', dest='data_dir', type=Path, default=Path("/media/nora/DATA/Pictures/Kepek/Italy2012"),
                    help='directory where the images are')
parser.add_argument('--output-dir', dest='out_dir', type=Path, default="/home/nora/Work/image_denoise/",
                    help='directory where output will be written')
args = parser.parse_args()

list_of_images = [f for f in args.data_dir.iterdir() if f.suffix == '.JPG']
random.shuffle(list_of_images)
n_images = len(list_of_images)

n_training = n_images * args.split[0] // 100
n_validation = n_images * args.split[1] // 100
n_test = n_images - n_training - n_validation
print(f"Splitting {n_images} images into {n_training} training, {n_validation} validation and {n_test} test samples")
my_split = DataSplit([str(x) for x in list_of_images[0:n_training]],
                     [str(x) for x in list_of_images[n_training:n_training+n_validation]],
                     [str(x) for x in list_of_images[-n_test:]])

out_file = args.out_dir / "split.json"
json_object = json.dumps(dataclasses.asdict(my_split), indent=4)
with open(out_file, 'w') as f_out:
    f_out.write(json_object)

# todo write test for this
