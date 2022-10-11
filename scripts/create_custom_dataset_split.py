import argparse
from pathlib import Path
import src.custom_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create data split')
    parser.add_argument('--split', type=int, nargs='+', default=(80, 10, 10),
                        help='Percentage Train/validation/test')
    parser.add_argument('--data-dir', dest='data_dir', type=Path,
                        default=Path("/media/nora/DATA/Pictures/Kepek/Italy2012"),
                        help='directory where the images are')
    parser.add_argument('--output-dir', dest='out_dir', type=Path, default="/home/nora/Work/denoise/",
                        help='directory where output will be written')
    args = parser.parse_args()

    my_split = src.custom_data.get_dataset_split(args.data_dir, tuple(args.split))
    src.custom_data.save_split_as_json(my_split, args.out_dir)
