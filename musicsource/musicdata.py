"""
Загрузка midi-композиций, построение датасета
"""

from tqdm import tqdm  
import pickle  # Сохранение данных
import os  
import numpy as np  
import json  

from musicsource.moduleloader import ModuleLoader
from musicsource.midiconnect import midiconnect
from musicsource.midiconnect import MidiInvalidException
import musicsource.songstruct as music


class MusicData:
    """Класс датасета
    """

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Имена файлов и пути до папок
        self.DATA_VERSION = '0.2'  # Assert compatibility between versions
        self.DATA_DIR_MIDI = 'data/midi'  # Originals midi files
        self.DATA_DIR_PLAY = 'data/play'  # Target folder to show the reconstructed files
        self.DATA_DIR_SAMPLES = 'data/samples'  # Training/testing samples after pre-processing
        self.DATA_SAMPLES_RAW = 'raw'  # Unpreprocessed songs container tag
        self.DATA_SAMPLES_EXT = '.pkl'
        self.TEST_INIT_FILE = 'data/test/initiator.json'  # Initial input for the generated songs
        self.FILE_EXT = '.mid'  # Could eventually add support for other format later ?

        # Параметры модели
        self.args = args

        # Dataset
        self.songs = []
        self.songs_train = None
        self.songs_test = None

        self.batch_builder = ModuleLoader.batch_builders.build_module(args)

        if not self.args.test:  
            self._restore_dataset()

            if self.args.play_dataset:
                print('Play some songs from the formatted data')
                # Генерация композиций
                for i in range(min(10, len(self.songs))):
                    raw_song = self.batch_builder.reconstruct_song(self.songs[i])
                    midiconnect.write_song(raw_song, os.path.join(self.DATA_DIR_PLAY, str(i)))
                raise NotImplementedError('Can\'t play a song for now')

            self._split_dataset()  

            #Вывод статистики
            print('Loaded: {} songs ({} train/{} test)'.format(
                len(self.songs_train) + len(self.songs_test),
                len(self.songs_train),
                len(self.songs_test))
            )  

    def _restore_dataset(self):
        """
        """

        # Construct the dataset names
        samples_path_generic = os.path.join(
            self.args.root_dir,
            self.DATA_DIR_SAMPLES,
            self.args.dataset_tag + '-{}' + self.DATA_SAMPLES_EXT
        )
        samples_path_raw = samples_path_generic.format(self.DATA_SAMPLES_RAW)
        samples_path_preprocessed = samples_path_generic.format(ModuleLoader.batch_builders.get_chosen_name())


        # Restoring precomputed database
        if os.path.exists(samples_path_preprocessed):
            print('Restoring dataset from {}...'.format(samples_path_preprocessed))
            self._restore_samples(samples_path_preprocessed)

        # First time we load the database: creating all files
        else:
            print('Training samples not found. Creating dataset from the songs...')
            # Restoring raw songs
            if os.path.exists(samples_path_raw):
                print('Restoring songs from {}...'.format(samples_path_raw))
                self._restore_samples(samples_path_raw)

            # First time we load the database: creating all files
            else:
                print('Raw songs not found. Extracting from midi files...')
                self._create_raw_songs()
                print('Saving raw songs...')
                self._save_samples(samples_path_raw)

            # At this point, self.songs contain the list of the raw songs. Each
            # song is then preprocessed by the batch builder

            # Generating the data from the raw songs
            print('Pre-processing songs...')
            for i, song in tqdm(enumerate(self.songs), total=len(self.songs)):
                self.songs[i] = self.batch_builder.process_song(song)

            print('Saving dataset...')
            np.random.shuffle(self.songs)  # Important to do that before saving so the train/test set will be fixed each time we reload the dataset
            self._save_samples(samples_path_preprocessed)

    def _restore_samples(self, samples_path):
        """ Load samples from file
        Args:
            samples_path (str): The path where to load the model (all dirs should exist)
        Return:
            List[Song]: The training data
        """
        with open(samples_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset

            # Check the version
            current_version = data['version']
            if current_version != self.DATA_VERSION:
                raise UserWarning('Present configuration version {0} does not match {1}.'.format(current_version, self.DATA_VERSION))

            # Restore parameters
            self.songs = data['songs']

    def _save_samples(self, samples_path):
        """ Save samples to file
        Args:
            samples_path (str): The path where to save the model (all dirs should exist)
        """

        with open(samples_path, 'wb') as handle:
            data = {  
                'version': self.DATA_VERSION,
                'songs': self.songs
            }
            pickle.dump(data, handle, -1)  

    def _create_raw_songs(self):
        """ Создание нового датасета из файлов
        """
        midi_dir = os.path.join(self.args.root_dir, self.DATA_DIR_MIDI, self.args.dataset_tag)
        midi_files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith(self.FILE_EXT)]

        for filename in tqdm(midi_files):

            try:
                new_song = midiconnect.load_file(filename)
            except MidiInvalidException as e:
                tqdm.write('File ignored ({}): {}'.format(filename, e))
            else:
                self.songs.append(new_song)
                tqdm.write('Song loaded {}: {} tracks, {} notes, {} ticks/beat'.format(
                    filename,
                    len(new_song.tracks),
                    sum([len(t.notes) for t in new_song.tracks]),
                    new_song.ticks_pb
                ))

        if not self.songs:
            raise ValueError('Empty dataset. Check that the folder exist and contains supported midi files.')

    def _convert_song2array(self, song):
        """ 
        Args:
            song (Song): The song to convert
        Return:
            Array: the numpy array: a binary matrix of shape [NB_NOTES, song_length]
        """

       
        song_length = len(song)
        scale = self._get_scale(song)


        piano_roll = np.zeros([music.NB_NOTES, int(np.ceil(song_length/scale))], dtype=int)

        for track in song.tracks:
            for note in track.notes:
                piano_roll[note.get_relative_note()][note.tick//scale] = 1

        return piano_roll

    def _convert_array2song(self, array):
        """ 
        Args:
            np.array: the numpy array (Warning: could be a array of int or float containing the prediction before the sigmoid)
        Return:
            song (Song): The song to convert
        """

        new_song = music.Song()
        main_track = music.Track()

        scale = self._get_scale(new_song)

        for index, x in np.ndenumerate(array):  # Add some notes
            if x > 1e-12:  # Note added 
                new_note = music.Note()

                new_note.set_relative_note(index[0])
                new_note.tick = index[1] * scale  # Absolute time in tick

                main_track.notes.append(new_note)

        new_song.tracks.append(main_track)

        return new_song

    def _split_dataset(self):
        """ 
        """
        split_nb = int(self.args.ratio_dataset * len(self.songs))
        self.songs_train = self.songs[:split_nb]
        self.songs_test = self.songs[split_nb:]
        self.songs = None  

    def get_batches(self):
        """ 
        Return:
            list[Batch], list[Batch]: The batches for the training and testing set (can be generators)
        """
        return (
            self.batch_builder.get_list(self.songs_train, name='train'),
            self.batch_builder.get_list(self.songs_test, name='test'),
        )


    def get_batches_test_old(self):  
        """ Return the batches which initiate the RNN when generating
        The initial batches are loaded from a json file containing the first notes of the song. The note values
        are the standard midi ones. Here is an examples of an initiator file:
        ```
        {"initiator":[
            {"name":"Simple_C4",
             "seq":[
                {"notes":[60]}
            ]},
            {"name":"some_chords",
             "seq":[
                {"notes":[60,64]}
                {"notes":[66,68,71]}
                {"notes":[60,64]}
            ]}
        ]}
        ```
        Return:
            List[Batch], List[str]: The generated batches with the associated names
        """
        assert self.args.batch_size == 1

        batches = []
        names = []

        with open(self.TEST_INIT_FILE) as init_file:
            initiators = json.load(init_file)

        for initiator in initiators['initiator']:
            raw_song = music.Song()
            main_track = music.Track()

            current_tick = 0
            for seq in initiator['seq']:  # We add a few notes
                for note_pitch in seq['notes']:
                    new_note = music.Note()
                    new_note.note = note_pitch
                    new_note.tick = current_tick
                    main_track.notes.append(new_note)
                current_tick += 1

            raw_song.tracks.append(main_track)
            raw_song.normalize(inverse=True)

            batch = self.batch_builder.process_batch(raw_song)

            names.append(initiator['name'])
            batches.append(batch)

        return batches, names

    @staticmethod
    def _convert_to_piano_rolls(outputs):
        """ 
        Args:
            outputs (List[np.array]): The list of the predictions of the decoder
        Return:
            List[np.array]: the list of the songs (one song by batch) as piano roll
        """

        # Extract the batches and recreate the array for each batch
        piano_rolls = []
        for i in range(outputs[0].shape[0]):  # Iterate over the batches
            piano_roll = None
            for j in range(len(outputs)):  # Iterate over the sample length
                # outputs[j][i, :] has shape [NB_NOTES, 1]
                if piano_roll is None:
                    piano_roll = [outputs[j][i, :]]
                else:
                    piano_roll = np.append(piano_roll, [outputs[j][i, :]], axis=0)
            piano_rolls.append(piano_roll.T)

        return piano_rolls

    def visit_recorder(self, outputs, base_dir, base_name, recorders, chosen_labels=None):
        """ 
        Args:
            outputs (List[np.array]): The list of the predictions of the decoder
            base_dir (str): Path were to save the outputs
            base_name (str): filename of the output (without the extension)
            recorders (List[Obj]): Interfaces called to convert the song into a file (ex: midi). The recorders
                need to implement the method write_song (the method has to add the file extension) and the
                method get_input_type.
            chosen_labels (list[np.Array[batch_size, int]]): the chosen class at each timestep (useful to reconstruct the generated song)
        """

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for batch_id in range(outputs[0].shape[0]):  # Loop over batch_size
            song = self.batch_builder.reconstruct_batch(outputs, batch_id, chosen_labels)
            for recorder in recorders:
                if recorder.get_input_type() == 'song':
                    input = song
                elif recorder.get_input_type() == 'array':
                    #input = self._convert_song2array(song)
                    continue 
                else:
                    raise ValueError('Unknown recorder input type.'.format(recorder.get_input_type()))
                base_path = os.path.join(base_dir, base_name + '-' + str(batch_id))
                recorder.write_song(input, base_path)
