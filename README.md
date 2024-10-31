# Local AI URL Reader Assistant - Ollama

## Overview

URL Reader Assistant is a powerful documentation analysis tool that combines web crawling capabilities with local Large Language Models (LLMs) through Ollama. The system efficiently processes web content and provides an interactive question-answering interface using Retrieval-Augmented Generation (RAG) technology.

The assistant specializes in crawling documentation websites, processing their content through local language models, and creating an intelligent knowledge base that users can query. By leveraging local LLM processing, it offers both privacy and cost-effectiveness while maintaining high-quality responses with source citations.

## Core Features

### Content Processing
- Multi-threaded web crawling for efficient content gathering
- Intelligent URL filtering and domain-specific content extraction
- Automated content chunking and optimization
- Vector database storage for efficient retrieval

### AI Capabilities
- Local LLM processing using Ollama
- Context-aware query processing
- Source-cited responses
- Conversation memory management
- Interactive Q&A interface

### System Features
- Database inspection and management tools
- Configurable crawling parameters
- Command history navigation
- Automatic cleanup and session management

## Installation

### Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally
3. Git for repository cloning
4. Virtual environment (recommended)

### Setup Process

```bash
# Clone the repository
git clone https://github.com/AIAfterDark/AI-URL-Read.git
cd url-reader

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Ensure Ollama is running and pull required model
ollama pull llama3.2
```

### Required Dependencies

Create a requirements.txt file containing:
```
langchain
langchain-community
langchain-ollama
beautifulsoup4
requests
chromadb
colorama
```

## Usage

### Basic Operation

The most straightforward way to use the URL Reader Assistant is:

```bash
python url-read.py https://example.com
```

### Advanced Configuration

For more control over the processing:

```bash
python url-read.py https://example.com \
    --model llama3.2 \
    --max-pages 5 \
    --verbose \
    --save-db \
    --memory-size 5
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| url | Target URL to analyze (required) | None |
| --model | Ollama model name | llama3.2 |
| --max-pages | Maximum pages to crawl | 50 |
| --verbose | Enable detailed logging | False |
| --save-db | Save database snapshot | False |
| --memory-size | Recent interactions to remember | 5 |

### Interactive Commands

During the Q&A session, the following commands are available:

- `/quit` - Exit the application
- `/db info` - Display database information
- `/db inspect <id>` - Inspect specific document chunk
- `/db save [filename]` - Save database snapshot

Use arrow keys (↑↓) for command history navigation.

## Example Session

```
$ python url-read.py https://docs.example.com

Database Information:
Total documents: 25
Database path: ./chroma_db

Article Overview:
[Generated content overview]

Documentation processed! Enter your questions (/quit to exit)

Question: What are the main features?

Answer:
[AI-generated response]

Sources:
- Documentation Home
  docs.example.com/home
- Features Page
  docs.example.com/features
```

## Best Practices

### Content Processing
1. Start with a small number of pages for initial testing
2. Enable verbose mode when debugging issues
3. Use database snapshots for important content
4. Verify source citations in responses

### Model Selection
1. Choose appropriate models based on content type
2. Consider memory requirements for larger sites
3. Balance between speed and accuracy needs

### Query Optimization
1. Ask specific, focused questions
2. Utilize conversation context for follow-ups
3. Review source citations for verification

## Known Limitations

1. Currently supports HTML content only
2. Single domain processing per session
3. Requires active Ollama installation
4. Memory usage scales with content size

## Troubleshooting

### Common Issues
1. Database Connection Errors
   - Verify ChromaDB installation
   - Check directory permissions
   - Ensure sufficient disk space

2. Ollama Connection Issues
   - Confirm Ollama is running
   - Verify model availability
   - Check network connectivity

3. Memory Problems
   - Reduce max pages parameter
   - Adjust chunk sizes
   - Increase available system memory

## Contributing

We welcome contributions to the URL Reader Assistant project. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

For detailed contribution guidelines, see CONTRIBUTING.md.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain team for the fundamental framework
- Ollama project for local LLM capabilities
- ChromaDB for vector storage solutions

## Contact

For issues and feature requests, please use the GitHub issue tracker.
