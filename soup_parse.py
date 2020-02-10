

import os, re

from tqdm import tqdm
from bs4 import BeautifulSoup
from argparse import ArgumentParser


# DIR_SET = ['/local/cfwelch/LDC2011T07/gigaword_eng_5_d1/data',
#            '/local/cfwelch/LDC2011T07/gigaword_eng_5_d2/data',
#            '/local/cfwelch/LDC2011T07/gigaword_eng_5_d3/data']
DIR_TEMPLATE = '/local/cfwelch/LDC2011T07/gigaword_eng_5_d{}/data'


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_part', dest='data_part', help='Directory to use for threading.', default=1, type=int)
    opt = parser.parse_args()

    dir_loc = DIR_TEMPLATE.format(opt.data_part)
    print('Parsing files from directory ' + dir_loc)

    # get all the files from all subdirectories
    file_set = []
    subdirs = os.listdir(dir_loc)
    for subd in subdirs:
        files = os.listdir(dir_loc + '/' + subd)
        for file in files:
            file_set.append(dir_loc + '/' + subd + '/' + file)


    print('Number of Gigaword Files: ' + str(len(file_set)))

    # iterate over files, parse, and write
    out_file = open('flat_gigaword_' + str(opt.data_part), 'w')
    for file in tqdm(file_set):
        soup = BeautifulSoup(open(file), 'html.parser')
        for paragraph in soup('p'):
            # turn inter-paragraph newlines into spaces
            paragraph = paragraph.get_text()
            paragraph = re.sub(r'\n+', '\n', paragraph)
            # paragraph = paragraph.replace('\n', ' ')
            out_file.write(paragraph)


if __name__ == '__main__':
    main()
