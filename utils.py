"""
Некоторые функции для упрощения работы с большим количеством данных.
Независимы от основной программы, необходимы при создании датасетов. 
"""

import os
import subprocess
import glob


def extract_files():
    """ Извлечение всех файлов из директории (в планах, пока не работает)
    """
    input_dir = '../www.chopinmusic.net/'
    output_dir = 'chopin_clean/'

    os.makedirs(output_dir, exist_ok=True)

    print('Extracting:')
    i = 0
    for filename in glob.iglob(os.path.join(input_dir, '**/*.mid'), recursive=True):
        print(filename)
        os.rename(filename, os.path.join(output_dir, os.path.basename(filename)))
        i += 1
    print('{} files extracted.'.format(i))


def rename_files():
    """ Переименование всех файлов в папке по правилу
    """
    input_dir = 'chopin/'
    output_dir = 'chopin_clean/'

    assert os.path.exists(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    list_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]

    print('Renaming {} files:'.format(len(list_files)))
    for prev_name in list_files:
        new_name = prev_name.replace('midi.asp?file=', '')
        new_name = new_name.replace('%2F', '_')
        print('{} -> {}'.format(prev_name, new_name))
        os.rename(os.path.join(input_dir, prev_name), os.path.join(output_dir, new_name))


def convert_midi2mp3():
    """ Функция конвертации всех midi в mp3 (взята из интернета, проверить пока не вышло)
    """
    input_dir = 'docs/midi/'
    output_dir = 'docs/mp3/'

    assert os.path.exists(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('Преобразование:')
    i = 0
    for filename in glob.iglob(os.path.join(input_dir, '**/*.mid'), recursive=True):
        print(filename)
        input_name = filename
        output_name = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + '.mp3')
        command = 'timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {}'.format(input_name, output_name)
        subprocess.call(command, shell=True)
        i += 1
    print('{} Файл преобразован'.format(i))


if __name__ == '__main__':
    convert_midi2mp3()
