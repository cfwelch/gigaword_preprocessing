

import os

from tqdm import tqdm
#from nltk.tokenize import sent_tokenize
from stanford_corenlp_pywrapper import CoreNLP

CORENLP_PATH = '/local/cfwelch/stanford-corenlp-full-2018-02-27/*'
IN_FILE = 'flat_gigaword_1'

def main():
    if not os.path.exists(IN_FILE + '_rf'):
        print('First reformatting file...')
        out_format = open(IN_FILE + '_rf', 'w')
        with open(IN_FILE) as handle:
            for line in tqdm(handle):
                tline = line.strip()
                if tline == '':
                    out_format.write('\n')
                else:
                    out_format.write(tline + ' ')

    print('Sentence tokenizer!')
    print('Loading Stanford CoreNLP...')
    proc = CoreNLP(configdict={'annotators': 'tokenize,ssplit', 'tokenize.options': 'ptb3Escaping=False'}, output_types=['tokenize,ssplit'], corenlp_jars=[CORENLP_PATH])

    out_file = open(IN_FILE + '_sts', 'w')
    sentence_count = 0

    print('Opening file ' + IN_FILE + '_rf' + '...')
    with open(IN_FILE + '_rf') as handle:
        lines = handle.readlines()
        for line in tqdm(lines):
            the_text = line.strip()
            # Use Stanford instead
            parsed = proc.parse_doc(the_text)

            sentence_count += len(parsed['sentences'])
            for sent in parsed['sentences']:
                the_tokens = [i.replace(' ', '') for i in sent['tokens']]
                the_sent = ' '.join(the_tokens)
                assert len(the_sent.split(' ')) == len(sent['tokens'])
                out_file.write(the_sent.encode('utf-8') + '\n')
    print('Number of sentences so far: ' + '{:,}'.format(sentence_count))

    out_file.close()

if __name__ == '__main__':
    main()
