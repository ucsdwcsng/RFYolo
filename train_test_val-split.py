import argparse
import random
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/temp', help='Path to dataset')
    parser.add_argument('--file', type=str, default='all.txt', help='File with all data points')
    opt = parser.parse_args()
        
    return opt

def main (opt):
    
    datadir = opt.datadir;
    allfile = opt.file;
    
    fname, _   = os.path.splitext(allfile);
    _, suffix  = os.path.splitext(fname);
    
    print(f"Dataset: {datadir}/{allfile}")
    
    # Load the text file
    with open(f'{datadir}/{allfile}', 'r') as file:
        data = file.readlines()

    # Shuffle the data randomly
    random.shuffle(data)

    # Calculate the sizes of each split
    n = len(data)
    train_size = int(0.65 * n)
    test_size = int(0.25 * n)
    val_size = int(0.15 * n)

    # Split the data into train, test, and validation sets
    train_data = data[:train_size]
    test_data = data[train_size:train_size+test_size]
    val_data = data[train_size+test_size:]

    # Write the data into three separate files
    with open(f'{datadir}/train{suffix}.txt', 'w') as file:
        file.writelines(train_data)

    with open(f'{datadir}/test{suffix}.txt', 'w') as file:
        file.writelines(test_data)

    with open(f'{datadir}/val{suffix}.txt', 'w') as file:
        file.writelines(val_data)



if __name__ == '__main__':
    opt = parse_args();
    
    main(opt);