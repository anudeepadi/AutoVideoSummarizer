# 🎥 AutoVideoSummarizer

## 📝 Overview
AutoVideoSummarizer is an advanced tool that automatically generates text summaries from video content using state-of-the-art AI models. It converts video audio to text and creates concise, meaningful summaries, making it perfect for content creators, students, and professionals who need to quickly understand video content.

## ✨ Features

### Core Functionality
- 🎯 Automatic video downloading from YouTube
- 🔊 High-quality audio extraction
- 📝 Advanced speech-to-text conversion
- 🤖 AI-powered text summarization
- 📊 Text analysis and processing
- ✍️ Proper capitalization and punctuation restoration

### Technical Features
- 🎯 Supports multiple video platforms
- 🔄 Batch processing capability
- 📊 Configurable summary length
- 🎚️ Adjustable summarization parameters
- 📋 Multiple output formats
- 🔍 Advanced text processing

## 🛠️ Technology Stack
- **Python 3.8+**
- **AI/ML Libraries**:
  - Transformers (Hugging Face)
  - TensorFlow/PyTorch
  - NLTK
- **Audio Processing**:
  - yt-dlp
  - librosa
  - FFmpeg
- **Text Processing**:
  - NLTK
  - SpaCy
  - Transformers

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg
- Required Python packages

### Setup Steps
1. Clone the repository:
```bash
git clone https://github.com/anudeepadi/AutoVideoSummarizer.git
cd AutoVideoSummarizer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (if not already installed):
- **Windows**: Download from official website
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

## 🚀 Usage

### Basic Usage
```python
from video_summarizer import VideoSummarizer

# Initialize the summarizer
summarizer = VideoSummarizer()

# Process a YouTube video
summary = summarizer.summarize_video("https://www.youtube.com/watch?v=example")

# Print the summary
print(summary)
```

### Advanced Configuration
```python
# Configure with custom parameters
summarizer = VideoSummarizer(
    model="facebook/bart-large-cnn",
    min_length=30,
    max_length=130,
    audio_quality="192K"
)

# Process with additional options
summary = summarizer.summarize_video(
    url="https://www.youtube.com/watch?v=example",
    generate_subtitles=True,
    save_transcript=True,
    output_format="markdown"
)
```

## ⚙️ Configuration Options

### Audio Processing
- Sampling rate: 16kHz (default)
- Audio quality: 192K (default)
- Audio format: WAV

### Text Processing
- Minimum summary length: 30 words
- Maximum summary length: 130 words
- Truncation: True
- Text cleaning: Enabled

### Model Options
- Speech recognition: facebook/wav2vec2-large-960h-lv60-self
- Summarization: facebook/bart-large-cnn
- Text processing: spaCy english model

## 📊 Output Formats
- Plain text
- Markdown
- JSON
- HTML
- Subtitles (SRT)

## 🤝 Contributing
Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Maintain type hints
- Use descriptive commit messages

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Hugging Face for transformer models
- YouTube-DL developers
- FFmpeg team
- Open source AI community

## 📞 Contact
For questions and support:
- GitHub Issues: [Create an issue](https://github.com/anudeepadi/AutoVideoSummarizer/issues)
- GitHub: [@anudeepadi](https://github.com/anudeepadi)

## 🔮 Future Plans
- Support for more video platforms
- Real-time summarization
- Multiple language support
- GUI interface
- API integration
- Cloud deployment options
