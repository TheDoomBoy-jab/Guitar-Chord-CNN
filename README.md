# Guitar-Chord-CNN

An end to end deep learning pipeline that uses tensorflow for the sequential model and transforms raw audio files(.wav) into log- mel spectrograms

---------------------------------------------------------------------------------------------

*PROJECT OVERVIEW*

Classifying music is fundamentally different from classifying speech or environmental noise. This project involved recording/augmenting a custom dataset of 8 guitar chords, building a signal processing pipeline, and manually debugging severe gradient and architectural issues to achieve a stable, learning neural network.

Tech Stack--
TensorFlow, Keras
librosa, tensorflow signal for audio processing
matplotlib, numpy for arrays and plotting 

16KB native sampling: to allow silicon models to scale to 16KHz to use tensorflow.io
Log-Mel Spectrogram: For converting the waveform into 2D-heatmaps using STFT
High Pass Filter: To remove the muddy strum of open chords like Em and Am
Z-Score Normalisation: To prevent the ReLU layers from dying due to the high "negative" spectrogram numbers produced and center it around 0

Input=128*128 Spectrogram Resized Heatmaps
3 Conv2D layers
Usage of AdamOptimizer

Training Callbacks used:
ReduceLROnPlateau: To scale down the learning rate in order to prevent overfitting condition and hitting a flat.
EarlyStopping(10): To end the training if val_loss degrades for first 15 epochs and restores the weights

CHALLENGES FACED:
The Dead Network: Caused due to 4 Dropout layers(each >=0.3) that almost vanished the training dataset
Validation Explosion: Dropout placement before BatchNormalization that caused violent spikes in validation_loss

RESULTS:
model achieves highly stable -75% val_accuracy with zero overfitting

FUTURE WORK:
Usage of Chromagrams that use the 12 pitch classes to extract the pitch classes for a particular chord and instantly identify it, this can push the model beyond 90% accuracy

_________________________________________________________________________________________________	     
