"""
В планах воспроизведение и генерация картинок = треков
Код частично адаптирован из https://github.com/Conchylicultor/MusicGenerator
"""

#import cv2 as cv 
import numpy as np

import musicsource.songstruct as music  # Should we use that to tuncate the top and bottom image ?


class ImgConnector:
    """ Класс, считывающий или воспроизводящий музыкальные произведения в картинках. В планах.
    """

    @staticmethod
    def load_file(filename):
        """ Извлечение данных из midi
        Args:
            filename (str): a valid img file
        Return:
            np.array: piano roll
        """

    @staticmethod
    def write_song(piano_roll, filename):
        """ Сохранение композиции на диск
        Args:
            piano_roll (np.array): 
            filename (str): Путь, куда сохраняется композиций
        """
        note_played = piano_roll > 0.5
        piano_roll_int = np.uint8(piano_roll*255)

        b = piano_roll_int * (~note_played).astype(np.uint8)  # Note silenced
        g = np.zeros(piano_roll_int.shape, dtype=np.uint8)    # Empty channel
        r = piano_roll_int * note_played.astype(np.uint8)     # Notes played

        #img = cv.merge((b, g, r)) ----- cv2

        #cv.imwrite(filename + '.png', img) -------  cv2

    @staticmethod
    def get_input_type():
        return 'array'
