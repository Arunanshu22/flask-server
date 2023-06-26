import os
import wave
import time
import pickle
import soundfile as sf
import librosa
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
from flask import Flask, request, redirect, url_for
import subprocess
import re
import glob


app = Flask(__name__)


warnings.filterwarnings("ignore")


def calculate_delta(array):
    rows, cols = array.shape
    print(rows)
    print(cols)
    deltas = np.zeros((rows, 20))
    n = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= n:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def record_audio_train():
    Name = (input("Please Enter Your Name:"))
    for count in range(5):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        print("----------------------record device list---------------------")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        print("-------------------------------------------------------------")
        index = int(input())
        print("recording via index " + str(index))
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK)
        print("recording started")
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print("recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = Name + "-sample" + str(count) + ".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
        trainedfilelist = open("training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME + "\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()


def record_audio_test():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 10
    device_index = 2
    audio = pyaudio.PyAudio()
    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = int(input())
    print("recording via index " + str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    OUTPUT_FILENAME = "sample.wav"
    WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)
    trainedfilelist = open("testing_set_addition.txt", 'a')
    trainedfilelist.write(OUTPUT_FILENAME + "\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()


def train_model(training_set, model_loc, training_set_filenames):
    file_paths = open(training_set_filenames, 'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        print(path)
        sr, audio = read(training_set + path)
        print(sr)
        vector = extract_features(audio, sr)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        if count == 5:
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)
            # dumping the trained gaussian model
            picklefile = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(model_loc + picklefile, 'wb'))
            print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
            features = np.asarray(())
            count = 0
        count = count + 1


def train_model_noTXT(training_set, model_loc):
    wav_files = glob.glob(training_set + "*.wav")
    count = 1
    features = np.asarray(())
    for wav_file in wav_files:
        print(wav_file)
        sr, audio = read(wav_file)
        print(sr)
        vector = extract_features(audio, sr)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        if count == 5:
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)
            # dumping the trained Gaussian model
            picklefile = wav_file.split("/")[-1].split(".")[0] + ".gmm"
            pickle.dump(gmm, open(model_loc + picklefile, 'wb'))
            print('+ modeling completed for speaker:', picklefile, " with data point =", features.shape)
            features = np.asarray(())
            count = 0
        count += 1


def test_model(source, modelpath, test_file):
    # give path to the test files (wav) here.
    # source = "C://xampp/htdocs/TvsInternshipCodes/Phase2/Final/testing_set/ArunLudClips/"
    # give path to the models locations here.
    # modelpath = "C://xampp/htdocs/TvsInternshipCodes/Phase2/Final/trained_models/"
    # give path to the test files names containing text file so that we can iterate through the names
    # and predict the speaker one by one.
    # test_file = "C://xampp/htdocs/TvsInternshipCodes/Phase2/Final/testing_set_addition.txt"

    file_paths = open(test_file, 'r')
    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

    # Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

    # Read the test directory and get the list of test audio files
    temp = []
    for path in file_paths:
        path = path.strip()
        print(path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
        temp.append(speakers[winner])
        time.sleep(1.0)
    # return temp
    result = re.search("/([^/]+)/?$", temp[0])
    output = result.group(1)
    return output


def test_model_for_enroll(source, modelpath, test_file):
    file_paths = open(test_file, 'r')
    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

    # Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

    # Read the test directory and get the list of test audio files
    temp = []
    for path in file_paths:
        path = path.strip()
        print(path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
        temp.append(speakers[winner])
        time.sleep(1.0)

    with open('names.txt', 'r') as file:
        name = file.readline().strip()
    count = temp.count(name)
    percent = (count / len(temp)) * 100
    return percent

def convert_mp3_to_wav(input_file, output_file):
    try:
        subprocess.run(['ffmpeg', '-i', input_file, output_file], check=True)
        print(f"Conversion successful: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")


def split_audio_into_clips(audio_path, output_dir, split_file_names):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=44100)
    # Calculate the total number of clips
    clip_duration = 9  # 9 second
    clip_samples = int(sr * clip_duration)
    total_clips = len(audio) // clip_samples
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Split the audio into clips and save each clip as a separate file
    for i in range(total_clips):
        clip_start = i * clip_samples
        clip_end = (i + 1) * clip_samples
        clip = audio[clip_start:clip_end]
        clip_filename = f"sample{i}.wav"
        clip_path = os.path.join(output_dir, clip_filename)
        trainedfilelist = open(split_file_names, 'w')
        trainedfilelist.write(clip_filename + "\n")
        sf.write(clip_path, clip, sr)


def split_audio_into_clips_train(audio_path, output_dir, split_file_names, user_name):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=44100)
    # Calculate the total number of clips
    clip_duration = 9  # 9 second
    clip_samples = int(sr * clip_duration)
    total_clips = len(audio) // clip_samples
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Split the audio into clips and save each clip as a separate file
    for i in range(total_clips):
        clip_start = i * clip_samples
        clip_end = (i + 1) * clip_samples
        clip = audio[clip_start:clip_end]
        clip_filename = f"{user_name}-sample{i}.wav"
        clip_path = os.path.join(output_dir, clip_filename)
        trainedfilelist = open(split_file_names, 'w')
        trainedfilelist.write(clip_filename + "\n")
        sf.write(clip_path, clip, sr)


@app.route('/getName', methods=['GET', 'POST'])
def getName():
    if request.method == 'POST':
        name = request.form.get('name')  # Retrieve the name from the request form data

        # Example usage: print the name
        print("Name:", name)

        # Write the name to a text file
        with open('names.txt', 'w') as file:
            file.write(name + '\n')

    return "Name handled"


@app.route('/getTest', methods=['GET', 'POST'])
def getTest():
    # code the capturing of the 1 min test file here
    # handle existing 1 min files
    input_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//test//input.mp3'
    output_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//test//output.wav'
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)

    # get the 1 min file
    if 'audio' not in request.files:
        return 'No audio file found', 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return 'No selected file', 400
    if audio_file:
        # Specify the directory where you want to store the uploaded audio file
        upload_dir = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//test//'
        audio_path = os.path.join(upload_dir, 'input.mp3')
        audio_file.save(audio_path)

    # text functionalities -
    input_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//test//input.mp3'
    output_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//test//output.wav'
    convert_mp3_to_wav(input_file, output_file)

    # break file and write name
    audio_file = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//test//output.wav"
    output_directory = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//testClips//"
    test_file_names = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//test.txt"
    split_audio_into_clips(audio_file, output_directory, test_file_names)

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    # flag = 1
    # handle existing 5 min files

    # handle existing 5 min files
    input_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//input.mp3'
    output_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//output.wav'
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)

    # get the 5 min file
    if 'audio' not in request.files:
        return 'No audio file found', 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return 'No selected file', 400
    if audio_file:
        # Specify the directory where you want to store the uploaded audio file
        upload_dir = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//'
        audio_path = os.path.join(upload_dir, 'input.mp3')
        audio_file.save(audio_path)

    # convert the mp3 to wav
    input_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//input.mp3'
    output_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//output.wav'
    convert_mp3_to_wav(input_file, output_file)

    # break the 5 min audio into 9 sec clips to train
    audio_file = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//output.wav"
    output_directory = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//train//"
    train_file_names = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//train.txt"
    with open('names.txt', 'r') as file:
        name = file.readline().strip()
    split_audio_into_clips_train(audio_file, output_directory, train_file_names, name)

    # train the model
    training_set = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//train//"
    model_loc = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//models//"
    training_set_filenames = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//train.txt"
    train_model_noTXT(training_set, model_loc)

    # test on the sample and get accuracy
    source = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//testClips//"
    modelpath = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//models//"
    test_file = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//5MinCrafts//test.txt"
    percent = test_model_for_enroll(source, modelpath, test_file)
    return percent


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # call functionalities -
    input_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//input.mp3'
    output_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//output.wav'
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)

    # upload functionalities -
    if 'audio' not in request.files:
        return 'No audio file found', 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return 'No selected file', 400
    if audio_file:
        # Specify the directory where you want to store the uploaded audio file
        upload_dir = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//'
        audio_path = os.path.join(upload_dir, 'input.mp3')
        audio_file.save(audio_path)

    # text functionalities -
    input_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//input.mp3'
    output_file = 'C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//output.wav'
    convert_mp3_to_wav(input_file, output_file)

    # break file and write name
    audio_file = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//output.wav"
    output_directory = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//FinalPredFiles/"
    split_audio_into_clips(audio_file, output_directory)

    # call test function
    source = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//FinalPredFiles//"
    modelpath = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//trained_models//"
    test_file = "C://xampp//htdocs//TvsInternshipCodes//Phase2//Final//AudioMedium//FinalPredFiles.txt"
    preds = test_model(source, modelpath, test_file)
    return preds


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

# train_model()

