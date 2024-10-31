import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
import logging
import time
import queue
import threading
from colorama import init, Fore, Style
import shutil
import json
from pathlib import Path
import readline
import atexit
import signal

# System prompt for the LLM
SYSTEM_PROMPT = """You are an AI assistant analyzing web content from vectorstore documents. Your role is to:

1. ALWAYS use English unless specifically asked to use another language
2. When answering questions about articles or documents:
   - Provide clear, structured responses with bullet points for multiple items
   - Only state facts that are explicitly present in the provided source documents
   - If the information isn't in the source documents, say "The article doesn't mention this" or "That information isn't included in the source material"
   - Never make assumptions or fill in missing information
   - Never reference information that isn't in the current source documents
   - Quote relevant text when appropriate using "quotes"
3. For article analysis:
   - Focus on information directly stated in the text
   - If asked about a specific topic, check if it's actually covered in the article before responding
   - Always ground your response in the source material
   - Structure responses with clear sections when appropriate
4. Quality control:
   - Double check that your response matches the source documents
   - If you're uncertain about any information, acknowledge the uncertainty
   - Don't switch languages randomly
   - Never start responses with "I don't know" if you then provide information
   
Remember: It's better to say "The article doesn't cover this" than to provide incorrect or assumed information."""

# Configure readline
histfile = os.path.join(os.path.expanduser("~"), ".url_read_history")
try:
    readline.read_history_file(histfile)
except FileNotFoundError:
    pass
atexit.register(readline.write_history_file, histfile)
readline.set_history_length(1000)

# Set logging levels
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

@dataclass
class PageContent:
    url: str
    title: str
    content: str
    links: List[Dict[str, str]]

class ColorOutput:
    def __init__(self):
        init()  # Initialize colorama
        
    def print_header(self, text):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
        
    def print_answer(self, text):
        print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")
        
    def print_sources(self, sources):
        print(f"\n{Fore.YELLOW}Sources:{Style.RESET_ALL}")
        for source in sources:
            print(f"{Fore.YELLOW}  {source}{Style.RESET_ALL}")
            
    def print_error(self, text):
        print(f"{Fore.RED}{text}{Style.RESET_ALL}")
        
    def print_info(self, text):
        print(f"{Fore.BLUE}{text}{Style.RESET_ALL}")

class DocumentProcessor:
    def __init__(self, base_url: str, model_name: str = "llama2", verbose: bool = False, memory_size: int = 5):
        self.base_url = base_url
        self.parsed_base = urlparse(base_url)
        self.visited_urls = set()
        self.url_queue = queue.Queue()
        self.content_store = {}
        self.color_output = ColorOutput()
        self.db_path = Path("./chroma_db")
        self.memory_size = memory_size
        self.article_overview = None
        
        # Clean up any existing database
        self.cleanup_database()
        
        # Register cleanup handlers
        atexit.register(self.cleanup_database)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.setup_logging(verbose)
        
        # Initialize LangChain components
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.llm = OllamaLLM(
            model=model_name,
            system=SYSTEM_PROMPT
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.conversation_chain = None
        # Enhanced memory with configurable window size
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=memory_size
        )

    def generate_article_overview(self, content: str) -> str:
        """Generate an overview of the article content"""
        overview_prompt = f"""Analyze this article and provide a concise overview including:
1. Main topic or purpose
2. Key points or arguments
3. Important details or findings
4. Target audience (if apparent)

Article content:
{content}"""
        
        try:
            return self.llm.invoke(overview_prompt)
        except Exception as e:
            self.logger.error(f"Error generating overview: {str(e)}")
            return "Error generating article overview"

    def cleanup_database(self):
        """Clean up the database directory"""
        try:
            if self.db_path.exists():
                shutil.rmtree(self.db_path)
        except Exception as e:
            print(f"Error cleaning up database: {e}")

    def signal_handler(self, signum, frame):
        """Handle cleanup on signal"""
        self.cleanup_database()
        exit(0)

    def setup_logging(self, verbose: bool):
        level = logging.WARNING if not verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def should_process_url(self, url: str) -> bool:
        """Determine if URL should be processed based on domain and path rules"""
        parsed_url = urlparse(url)
        
        # Check if URL is in the same domain
        if parsed_url.netloc != self.parsed_base.netloc:
            return False
            
        # Skip common non-documentation paths
        skip_patterns = [
            '/assets/', '/static/', '/images/',
            '.jpg', '.png', '.gif', '.css', '.js'
        ]
        
        return not any(pattern in url.lower() for pattern in skip_patterns)

    def get_database_info(self) -> Dict:
        """Get information about the vector store database"""
        if not self.vectorstore:
            return {"error": "Vector store not initialized"}
            
        try:
            collection = self.vectorstore._collection
            return {
                "total_documents": collection.count(),
                "db_path": str(self.db_path.absolute()),
                "ids": collection.get()["ids"][:10],  # Show first 10 IDs
                "total_ids": len(collection.get()["ids"])
            }
        except Exception as e:
            return {"error": f"Failed to get database info: {str(e)}"}
            
    def inspect_document(self, doc_id: str) -> Dict:
        """Retrieve a specific document from the vector store"""
        if not self.vectorstore:
            return {"error": "Vector store not initialized"}
            
        try:
            collection = self.vectorstore._collection
            result = collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if not result["ids"]:
                return {"error": f"Document {doc_id} not found"}
                
            return {
                "id": doc_id,
                "content": result["documents"][0],
                "metadata": result["metadatas"][0],
                "embedding_size": len(result["embeddings"][0]) if result["embeddings"] else None
            }
        except Exception as e:
            return {"error": f"Failed to retrieve document: {str(e)}"}
            
    def save_database_snapshot(self, output_path: str = "db_snapshot.json"):
        """Save a snapshot of the database content"""
        if not self.vectorstore:
            self.color_output.print_error("Vector store not initialized")
            return
            
        try:
            collection = self.vectorstore._collection
            data = collection.get()
            
            snapshot = {
                "documents": [],
                "total_count": len(data["ids"])
            }
            
            for idx, doc_id in enumerate(data["ids"]):
                snapshot["documents"].append({
                    "id": doc_id,
                    "content": data["documents"][idx],
                    "metadata": data["metadatas"][idx]
                })
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2)
                
            self.color_output.print_info(f"Database snapshot saved to {output_path}")
        except Exception as e:
            self.color_output.print_error(f"Failed to save database snapshot: {str(e)}")

    def extract_page_content(self, url: str) -> Optional[PageContent]:
        """Extract content and links from a webpage"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (customize selectors based on site structure)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': 'content'})
            if not main_content:
                main_content = soup.body
                
            # Extract title
            title = soup.title.string if soup.title else url
            
            # Extract relevant links
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href:
                    absolute_url = urljoin(url, href)
                    if self.should_process_url(absolute_url):
                        links.append({
                            'url': absolute_url,
                            'text': link.get_text(strip=True)
                        })
            
            return PageContent(
                url=url,
                title=title,
                content=main_content.get_text(separator=' ', strip=True),
                links=links
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return None

    def crawl_documentation(self, max_pages: int = 50, max_threads: int = 5):
        """Crawl documentation pages using multiple threads"""
        self.url_queue.put(self.base_url)
        threads = []
        
        def crawler_worker():
            while True:
                try:
                    url = self.url_queue.get(timeout=5)
                    if url not in self.visited_urls and len(self.visited_urls) < max_pages:
                        self.logger.info(f"Processing: {url}")
                        content = self.extract_page_content(url)
                        
                        if content:
                            self.content_store[url] = content
                            self.visited_urls.add(url)
                            
                            # Add new links to queue
                            for link in content.links:
                                if link['url'] not in self.visited_urls:
                                    self.url_queue.put(link['url'])
                                    
                    self.url_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error in crawler worker: {str(e)}")
                    self.url_queue.task_done()
        
        # Start crawler threads
        for _ in range(max_threads):
            thread = threading.Thread(target=crawler_worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        # Wait for initial queue to be processed
        self.url_queue.join()
        
        # Process content into vector store
        self.process_content_to_vectorstore()

    def process_content_to_vectorstore(self):
        """Process crawled content into vector store"""
        documents = []
        
        # First, create a summary of all content
        all_content = ""
        for url, page_content in self.content_store.items():
            all_content += f"\nTitle: {page_content.title}\n{page_content.content}\n"
            
        # Generate article overview
        self.article_overview = self.generate_article_overview(all_content)
        
        # Optimize chunk size based on content length
        avg_content_length = len(all_content) / len(self.content_store)
        chunk_size = min(2000, max(500, int(avg_content_length / 3)))
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.2),  # 20% overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for url, page_content in self.content_store.items():
            # Split content into chunks
            chunks = self.text_splitter.split_text(page_content.content)
            
            # Create Document objects for each chunk
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "url": url,
                        "title": page_content.title
                    }
                )
                documents.append(doc)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents,
            self.embeddings,
            collection_name="doc_chunks",
            persist_directory=str(self.db_path)
        )
        
        # Initialize conversation chain with enhanced memory
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
            ),
            memory=self.memory,
            return_source_documents=True
        )

    def query_documentation(self, query: str) -> Dict:
        """Query the documentation using RAG"""
        if not self.conversation_chain:
            raise ValueError("Documentation not processed. Run crawl_documentation first.")
            
        try:
            # Use invoke instead of __call__
            result = self.conversation_chain.invoke({
                "question": query,
                "chat_history": self.memory.chat_memory.messages
            })
            
            # Format response with sources
            sources = set()
            for doc in result.get("source_documents", []):
                if "url" in doc.metadata:
                    # Clean up the URL for better readability
                    url = doc.metadata["url"]
                    title = doc.metadata["title"]
                    
                    # Remove common URL prefixes for cleaner display
                    url = url.replace("https://", "").replace("http://", "")
                    
                    # Format title - remove redundant site names if present
                    title = title.split(" - ")[0].strip()
                    
                    sources.add(f"- {title}\n  {url}")
            
            # Clean up the answer text
            answer = result["answer"].strip()
            
            # Add bullet points for lists if they're not already present
            lines = answer.split("\n")
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("•", "-", "*", "1.", "2.", "3.")):
                    if any(keyword in line.lower() for keyword in ["first", "second", "third", "finally", "additionally", "moreover"]):
                        formatted_lines.append(f"• {line}")
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            
            formatted_answer = "\n".join(formatted_lines)
            
            return {
                "answer": formatted_answer,
                "sources": sorted(list(sources))  # Sort sources for consistent display
            }
            
        except Exception as e:
            self.logger.error(f"Error querying documentation: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": []
            }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="An AI-powered documentation assistant that crawls web pages, processes their content using local LLMs through Ollama, and enables interactive Q&A. The tool extracts content from documentation sites, stores it in a vector database, and uses RAG (Retrieval-Augmented Generation) to provide context-aware answers with source citations. Features include multi-threaded crawling, conversation memory, and database inspection capabilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python url-read_v1.py https://example.com
  
  # Use a specific Ollama model and limit page crawling
  python url-read_v1.py https://example.com --model llama2 --max-pages 5
  
  # Enable verbose logging and save database snapshot
  python url-read_v1.py https://example.com -v --save-db
  
Available Commands During Runtime:
  /quit               Exit the application
  /db info           Show current database information
  /db inspect <id>   Inspect a specific document chunk
  /db save [file]    Save database snapshot to file
  
Features:
  - Intelligent URL content parsing and chunking
  - Conversational memory of recent interactions
  - Arrow key navigation for command history
  - Database inspection and snapshot capabilities
  - Clean session management (fresh database per run)
""")
    
    parser.add_argument(
        "url",
        help="URL to parse and analyze. The content will be processed and made available for questions."
    )
    
    parser.add_argument(
        "--model",
        default="llama2",
        help="Ollama model name to use for embeddings and chat (default: llama2)"
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum number of pages to crawl if the URL contains links (default: 50)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging purposes"
    )
    
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Save a snapshot of the database after processing for later analysis"
    )
    
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path to store the vector database (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--memory-size",
        type=int,
        default=5,
        help="Number of recent interactions to remember in conversation (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        processor = DocumentProcessor(
            base_url=args.url,
            model_name=args.model,
            verbose=args.verbose,
            memory_size=args.memory_size
        )
        
        processor.crawl_documentation(max_pages=args.max_pages)
        
        # Print database info
        db_info = processor.get_database_info()
        processor.color_output.print_info("\nDatabase Information:")
        processor.color_output.print_info(f"Total documents: {db_info.get('total_documents', 'unknown')}")
        processor.color_output.print_info(f"Database path: {db_info.get('db_path', 'unknown')}")
        
        # Print article overview
        processor.color_output.print_header("\nArticle Overview:")
        processor.color_output.print_info(processor.article_overview)
        
        if args.save_db:
            processor.save_database_snapshot()
        
        processor.color_output.print_header("\nDocumentation processed! Enter your questions (/quit to exit)")
        processor.color_output.print_info("Available commands:")
        processor.color_output.print_info("  /quit - Exit the application")
        processor.color_output.print_info("  /db info - Show database information")
        processor.color_output.print_info("  /db inspect <id> - Inspect a specific document")
        processor.color_output.print_info("  /db save [filename] - Save database snapshot")
        processor.color_output.print_info("  Use arrow keys ↑↓ for command history")
        
        while True:
            try:
                query = input(f"\n{Fore.CYAN}Question: {Style.RESET_ALL}").strip()
                
                if not query:
                    continue
                
                if query.lower() == '/quit':
                    processor.cleanup_database()
                    break
                    
                if query.startswith('/db'):
                    # Handle database inspection commands
                    parts = query.split()
                    if len(parts) > 1 and parts[1] == 'info':
                        db_info = processor.get_database_info()
                        processor.color_output.print_info("\nDatabase Information:")
                        processor.color_output.print_info(json.dumps(db_info, indent=2))
                    elif len(parts) > 2 and parts[1] == 'inspect':
                        doc_info = processor.inspect_document(parts[2])
                        processor.color_output.print_info("\nDocument Information:")
                        processor.color_output.print_info(json.dumps(doc_info, indent=2))
                    elif len(parts) > 1 and parts[1] == 'save':
                        output_path = parts[2] if len(parts) > 2 else "db_snapshot.json"
                        processor.save_database_snapshot(output_path)
                    continue
                
                # Print a separator line before the answer
                print(f"\n{Fore.BLUE}{'─' * 80}{Style.RESET_ALL}")
                
                result = processor.query_documentation(query)
                processor.color_output.print_answer("\nAnswer:\n" + result["answer"])
                
                if result["sources"]:
                    processor.color_output.print_sources(result["sources"])
                
                # Print a separator line after the answer
                print(f"\n{Fore.BLUE}{'─' * 80}{Style.RESET_ALL}")
                    
            except KeyboardInterrupt:
                print("\nUse /quit to exit")
                continue
            except Exception as e:
                processor.color_output.print_error(f"\nError: {str(e)}")
                continue
                
    except Exception as e:
        processor.color_output.print_error(f"\nError: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
