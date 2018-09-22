"""
Иерархическая структура данных в композиции
Код частично адаптирован из https://github.com/Conchylicultor/MusicGenerator
"""

import operator  # 


MIDI_NOTES_RANGE = [21, 108]

NB_NOTES = MIDI_NOTES_RANGE[1] - MIDI_NOTES_RANGE[0] + 1

BAR_DIVISION = 16  


class Note:
    """ Структура ноты
    """
    def __init__(self):
        self.tick = 0
        self.note = 0
        self.duration = 32  

    def get_relative_note(self):
        """ 
        Return
            int: The new position relative to the range (position on keyboard)
        """
        return self.note - MIDI_NOTES_RANGE[0]

    def set_relative_note(self, rel):
        """ 
        Args:
            rel (int): The new position relative to the range (position on keyboard)
        """
        self.note = rel + MIDI_NOTES_RANGE[0]


class Track:
    """ Трек - последовательность нот (партия)
    """
    def __init__(self):
        self.instrum = None
        self.notes = []  # List[Note]
        self.is_drum = False

    def set_instrum(self, msg):
        """ Initialize from a mido message
        Args:
            msg (mido.MidiMessage): a valid control_change message
        """
        if self.instrum is not None:  # Already an instrum set
            return False

        assert msg.type == 'program_change'

        self.instrum = msg.program
        if msg.channel == 9 or msg.program > 112:  # Warning: Mido shift the channels (start at 0)
            self.is_drum = True

        return True


class Song:
    """ Композиция - список партий
    """


    MAXIMUM_SONG_RESOLUTION = 4
    NOTES_PER_BAR = 4  

    def __init__(self):
        self.ticks_pb = 96
        self.tempo_map = []
        self.tracks = []  # List[Track]

    def __len__(self):
        """ 
        """
        return max([max([n.tick + n.duration for n in t.notes]) for t in self.tracks])

    def _get_scale(self):
        """ 

        Return:
            int: the scale factor for the current song
        """

        return 4 * self.ticks_pb // (Song.MAXIMUM_SONG_RESOLUTION*Song.NOTES_PER_BAR)

    def normalize(self, inverse=False):
        """ 
        Args:
            inverse (bool): if true, we reverse the normalization
        """
        scale = self._get_scale()
        op = operator.floordiv if not inverse else operator.mul

        # Сдвиг всех нот
        for track in self.tracks:
            for note in track.notes:
                note.tick = op(note.tick, scale)  # //= or *=
