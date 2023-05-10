# Import necessary libraries
from __future__ import unicode_literals
import youtube_dl
from IPython.display import Audio 
import librosa 
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import re
import stanfordnlp

# Download required resources for StanfordNLP if not already present
# (uncomment below line if necessary)
# stanfordnlp.download('en')

# Initialize StanfordNLP pipeline for tokenizing and part-of-speech tagging
stf_nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')

# Set up options for downloading audio from YouTube
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'outtmpl':"." + '/video.%(ext)s',
}

# Download audio from a YouTube video
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=wtCp2CK2k34'])

# Get the file path of the downloaded audio file
absolute_path = "video.wav"

# Load the audio file and get its sampling rate
sampling_rate = 16_000
speech, rate = librosa.load(absolute_path)

# Display the audio in the notebook
Audio(speech, rate=rate)

# Set up options for the speech-to-text model
model = "facebook/wav2vec2-large-960h-lv60-self"
pipe = pipeline(model=model)

# Transcribe the audio using the speech-to-text model
# and save the transcription to a text file
text = pipe(absolute_path, chunk_length_s=10)
text_file = open("original_text.txt", "w")
n = text_file.write(text["text"])
text_file.close()

# Read the transcribed text from the text file
text_article = open("original_text.txt", "r").read()

# Display the number of words in the transcribed text
print("Number of words in the transcribed text: ", len(text_article.split()))

# Set up options for the summarization model
summarizer = pipeline("summarization", "facebook/bart-large-cnn")
tokenizer_kwargs = {'truncation': True, 'max_length': 512}

# Summarize the transcribed text and print the summary
text_summarization = summarizer(text_article, min_length=30, do_sample=False, **tokenizer_kwargs)
print("Summary of the transcribed text: ", text_summarization[0]['summary_text'])

# Function to restore capitalization to a summarized text
def truecasing(input_text):
    # Split the text into sentences
    sentences = sent_tokenize(input_text, language='english')
    
    # Capitalize the first letter of each sentence
    sentences_capitalized = [s.capitalize() for s in sentences]
    
    # Join the capitalized sentences
    text_truecase = re.sub(" (?=[\.,'!?:;])", "", ' '.join(sentences_capitalized))
    
    # Capitalize words according to part-of-speech tagging (POS)
    doc = stf_nlp(text_truecase)
    text_truecase =  ' '.join([w.text.capitalize() if w.upos in ["PROPN","NNS"] \
                                                   else w.text for sent in doc.sentences \
                               for w in sent.words])
    
    # Restore punctuation marks and return the truecased text
    text_truecase = re.sub(r'\s([?.!])', r'\1', text_truecase)
    # Return the truecased text
    return text_truecase
  
# Read the original text from file
with open("original_text.txt", "r") as f:
  original_text = f.read()

# Tokenize the original text into sentences
sentences = sent_tokenize(original_text)

# Initialize the StanfordNLP pipeline for part-of-speech tagging
stanford_nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')

# Create a list to hold the truecased sentences
truecased_sentences = []

# Loop through each sentence and truecase it
for sentence in sentences:
# Capitalize the sentence
  sentence = sentence.capitalize()
  
# Truecase the sentence using part-of-speech tagging
doc = stanford_nlp(sentence)
truecased_sentence = ' '.join([w.text.capitalize() if w.upos in ["PROPN","NNS"] else w.text for sent in doc.sentences for w in sent.words])

# Add the truecased sentence to the list
truecased_sentences.append(truecased_sentence)
# Join the truecased sentences together to form the truecased text
truecased_text = ' '.join(truecased_sentences)

# Write the truecased text to a new file
with open("truecased_text.txt", "w") as f:
f.write(truecased_text)

# Print a success message
print("Truecasing complete! Output saved to truecased_text.txt.")
