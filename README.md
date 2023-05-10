## Automatic Speech-to-Text and Text Summarization
This Github repository contains Python code for converting audio from a YouTube video to text using Facebook's wav2vec2 speech-to-text model, summarizing the text using Facebook's BART model, and truecasing the summary using StanfordNLP.

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies using the following command: pip install -r requirements.txt.
3. Run the summarize.py file.
4. The script will download the audio from a specified YouTube video, transcribe it to text using the Facebook wav2vec2 model, summarize the text using the Facebook BART model, and truecase the summary using StanfordNLP.
5. The original text and the truecased summary will be saved in original_text.txt and truecased_text.txt, respectively.

## License
This code is released under the MIT License. See LICENSE.md for details.
