# https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english
# https://huggingface.co/bert-base-uncased

import sounddevice as sd
from scipy.io.wavfile import write
import os
 
fs = 44100  # this is the frequency sampling; also: 4999, 64000
seconds = 10  # Duration of recording
 
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
print("Starting: Speak now!")
sd.wait()  # Wait until recording is finished
print("finished")
write('output.wav', fs, myrecording)  # Save as WAV file
os.startfile("output.wav")

from huggingsound import SpeechRecognitionModel
from transformers import pipeline
from transformers import logging
import numpy as np
from transformers import BertTokenizer, BertModel

w2vmodel = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
logging.set_verbosity_error() #change'error' to 'warning' or remove this if you want to see the warning
unmasker = pipeline('fill-mask', model='bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

def levenshtein_distance(s, t):
    m, n = len(s), len(t)
    d = [[0] * (n+1) for _ in range(m+1)]
    
    for i in range(m+1):
        d[i][0] = i
    
    for j in range(n+1):
        d[0][j] = j
        
    for j in range(1, n+1):
        for i in range(1, m+1):
            if s[i-1] == t[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
                
    return d[m][n]

def collate(input):
    pun_marks = [",", ".", "?", "!", ";", ":", "-", "—", "(", ")", "[", "]", "{", "}", "'", "\"", "`"]
    output = ""
    Capital = True
    Dash = False
    for i in range(len(input)):
        if input[i] in pun_marks:
            output += input[i]
            if input[i] in [".", "("]:
                Capital = True
            if input[i] in ["-", "'"]:
                Dash = True
        else:
            str = ""
            if (Dash == False):
                str += " "
            if (Dash == True):
                Dash = False
            if Capital:
                str += input[i].capitalize()
                Capital = False
            else:
                str += input[i]
            output += str
    return output

audio_paths = ["output.wav"]

transcriptions = w2vmodel.transcribe(audio_paths)
input = transcriptions[0]["transcription"]
input = input.split()

#(1) is a strategy where tokens are used to determine lexicographic distance
#(2) is a strategy where replaced words 
for t in range(1):
    # output = [] #(2)
    for i in range(len(input)):
        temp = input[i]
        token = tokenizer(temp)['input_ids'][1]
        input[i] = "[MASK]"
        apiint = unmasker(' '.join(input))
        dist = []
        for r in range(5):
            # if (np.abs((apiint[r]['token'] - token)) < 2): #(1)
            dist.append(levenshtein_distance(temp, apiint[r]['token_str']))
        lindex = 0
        l = dist[0]
        for r in range(5):
            if dist[r] < l:
                lindex = r
                l = dist[r]
        if l <= 2:
            input[i] = apiint[lindex]['token_str']
            # output.append(apiint[lindex]['token_str']) #(2)
        else:
            input[i] = temp
            # output.append(temp) #(2)
        # input[i] = temp #(2)

for t in range(1):
    inndex = 1
    for i in range(len(input)):
        input.insert(inndex, "[MASK]")
        # print(' '.join(input))
        apiint = unmasker(' '.join(input))
        if (apiint[0]['token'] < 1500):
            input[inndex] = apiint[0]["token_str"]
            inndex += 2
        else:
            del input[inndex]
            inndex += 1

print(collate(input))

# In comparison, a plain autocorrect gives this output:

# "The b-movie by Jerry Sinclair, the sound of buzzing 
# bees, can be heard according to all known laws of 
# aviation that is no way for b to be able to fly its 
# wings are too small to get its start little body off 
# the ground, the be, of course, flies anyway because 
# bees don't care what humans think is possible. 
# Barbuda is guaranteed one member of the House of 
# Representatives and two members of the Senate."

# - https://huggingface.co/oliverguhr/spelling-correction-english-base?text=lets+do+a+comparsion