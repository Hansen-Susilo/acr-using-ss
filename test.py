import os
import mir_eval
import pretty_midi as pm
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
from utils.fix_overlap import fix_overlapping_intervals
from demucs_combine import split
import argparse
import warnings


warnings.filterwarnings('ignore')
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--voca', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--song_name', type=str, default='input.mp3')
parser.add_argument('--save_dir', type=str, default='./test-songs_results')
args = parser.parse_args()

config = HParams.load("run_config.yaml")

if args.voca is True:
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170
    model_file = './test/btc_model_large_voca.pt'
    idx_to_chord = idx2voca_chord()
    logger.info("label type: large voca")
else:
    model_file = './test/btc_model.pt'
    idx_to_chord = idx2chord
    logger.info("label type: Major and minor")

model = BTC_model(config=config.model).to(device)

# Load model
if os.path.isfile(model_file):
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    logger.info("restore model")

# Audio files with format of wav and mp3
song_name = args.song_name
audio_file = './test-songs/' + song_name
split(audio_file)

audio_paths_splitted = get_audio_paths('./test-splitted')
audio_paths = get_audio_paths('./test-songs')

# A. Chord recognition WITH SS and save lab file
for i, audio_path in enumerate(audio_paths_splitted):
    logger.info("======== %d of %d in progress ========" % (i + 1, len(audio_paths_splitted)))
    # Load mp3
    feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
    logger.info("audio file loaded and feature computation success : %s" % audio_path)

    # Majmin type chord recognition
    feature = feature.T
    feature = (feature - mean) / std
    time_unit = feature_per_second
    n_timestep = config.model['timestep']

    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    start_time = 0.0
    lines = []
    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for i in range(n_timestep):
                if t == 0 and i == 0:
                    prev_chord = prediction[i].item()
                    continue
                if prediction[i].item() != prev_chord:
                    lines.append(
                        '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    start_time = time_unit * (n_timestep * t + i)
                    prev_chord = prediction[i].item()
                if t == num_instance - 1 and i + num_pad == n_timestep:
                    if start_time != time_unit * (n_timestep * t + i):
                        lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    break

    # lab file write
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, song_name.replace('.mp3', '').replace('.wav', '') + ' - result_with_ss.lab')
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)

    logger.info("label file saved : %s" % save_path)

# B. Chord recognition WITHOUT SS and save lab file
for i, audio_path in enumerate(audio_paths):
    logger.info("======== %d of %d in progress ========" % (i + 1, len(audio_paths)))
    # Load mp3
    feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
    logger.info("audio file loaded and feature computation success : %s" % audio_path)

    # Majmin type chord recognition
    feature = feature.T
    feature = (feature - mean) / std
    time_unit = feature_per_second
    n_timestep = config.model['timestep']

    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    start_time = 0.0
    lines = []
    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for i in range(n_timestep):
                if t == 0 and i == 0:
                    prev_chord = prediction[i].item()
                    continue
                if prediction[i].item() != prev_chord:
                    lines.append(
                        '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    start_time = time_unit * (n_timestep * t + i)
                    prev_chord = prediction[i].item()
                if t == num_instance - 1 and i + num_pad == n_timestep:
                    if start_time != time_unit * (n_timestep * t + i):
                        lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    break

    # lab file write
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, song_name.replace('.mp3', '').replace('.wav', '') + ' - result_without_ss.lab')
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)

    logger.info("label file saved : %s" % save_path)


# EVALUATION
ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals('./test-songs_results/' + song_name.replace('.mp3', '').replace('.wav', '') + '.lab')
pred_intervals_with_ss, pred_labels_with_ss = mir_eval.io.load_labeled_intervals('./test-songs_results/' + song_name.replace('.mp3', '').replace('.wav', '') + ' - result_with_ss.lab')
pred_intervals_without_ss, pred_labels_without_ss = mir_eval.io.load_labeled_intervals('./test-songs_results/' + song_name.replace('.mp3', '').replace('.wav', '') + ' - result_without_ss.lab')

# Fix overlapping intervals (if any)
ref_intervals, ref_labels = fix_overlapping_intervals(ref_intervals, ref_labels)
pred_intervals_with_ss, pred_labels_with_ss = fix_overlapping_intervals(pred_intervals_with_ss, pred_labels_with_ss)
pred_intervals_without_ss, pred_labels_without_ss = fix_overlapping_intervals(pred_intervals_without_ss, pred_labels_without_ss)

# Standard mir_eval chord evaluation
results_with_ss = mir_eval.chord.evaluate(ref_intervals, ref_labels, pred_intervals_with_ss, pred_labels_with_ss)
results_without_ss = mir_eval.chord.evaluate(ref_intervals, ref_labels, pred_intervals_without_ss, pred_labels_without_ss)

print(f"Mirex with SS: {results_with_ss['mirex']:.4f}")
print(f"Mirex without SS: {results_without_ss['mirex']:.4f}")