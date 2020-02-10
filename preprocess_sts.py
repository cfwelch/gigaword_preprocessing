

import unicodedata, re

from tqdm import tqdm

IN_FILE = 'flat_gigaword_1'

def preprocess(sent):
    sent = sent.strip().lower().split(' ')
    sent = ' '.join(['N' if re.match('^\+?\-?([0-9]+[,+-]?)+(.[0-9]+)?$', i) else i for i in sent])
    return ''.join(c for c in unicodedata.normalize('NFD', sent) if unicodedata.category(c) != 'Mn')

def main():
    out_file = open(IN_FILE + '_preproc', 'w')

    print('Preprocessing ' + IN_FILE + '_sts...')
    with open(IN_FILE + '_sts') as handle:
        for line in tqdm(handle.readlines()):
            out_file.write(preprocess(line) + '\n')

if __name__ == '__main__':
    main()
