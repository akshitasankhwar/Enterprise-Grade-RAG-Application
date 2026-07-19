"""API Client - Test and interact with the FastAPI backend"""

import requests
import json
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Console for pretty output
console = Console()

BASE_URL = "http://localhost:8000"


class RAGClient:
    """Client for interacting with the RAG API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health(self) -> dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def query(self, query: str, session_id: Optional[str] = None) -> dict:
        """Send a query to the API"""
        payload = {
            "query": query,
            "session_id": session_id
        }
        response = self.session.post(
            f"{self.base_url}/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def info(self) -> dict:
        """Get API info"""
        response = self.session.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()


def print_response(response_data: dict):
    """Pretty print API response"""
    
    # Query and classification
    console.print(f"\n📝 Query: {response_data['query']}")
    console.print(f"🏷️  Classification: {response_data['classification']}")
    
    # Response
    console.print(Panel(
        response_data['response'],
        title="Response",
        expand=False
    ))
    
    # Sources
    if response_data['sources']:
        table = Table(title="Sources Used")
        table.add_column("Source", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Text Preview", style="green")
        
        for source in response_data['sources']:
            preview = source['text'][:80] + "..." if len(source['text']) > 80 else source['text']
            table.add_row(
                source['source'],
                f"{source['score']:.4f}",
                preview
            )
        
        console.print(table)
    else:
        console.print("[yellow]No sources used (conversational response)[/yellow]")
    
    # Reasoning
    console.print(f"\n💭 Reasoning: {response_data['reasoning']}")


def main():
    """Main CLI"""
    console.print("[bold cyan]Enterprise Agentic RAG - CLI Client[/bold cyan]\n")
    
    client = RAGClient()
    
    # Check health
    try:
        console.print("🔍 Checking API health...")
        health = client.health()
        
        if health['status'] == "healthy":
            console.print("[green]✅ API is healthy[/green]")
        else:
            console.print(f"[yellow]⚠️  API status: {health['status']}[/yellow]")
            console.print(f"Services: {health['services']}")
    except Exception as e:
        console.print(f"[red]❌ Cannot connect to API: {e}[/red]")
        console.print("Make sure to run: uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Get info
    try:
        console.print("\n📊 Getting API info...")
        info = client.info()
        console.print(f"Collection: {info['qdrant']['collection']}")
        console.print(f"Documents: {info['qdrant']['point_count']}")
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not get info: {e}[/yellow]")
    
    # Interactive query loop
    console.print("\n[bold]Enter queries (type 'exit' to quit)[/bold]\n")
    
    while True:
        try:
            query = console.input("[cyan]You:[/cyan] ").strip()
            
            if query.lower() == "exit":
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query:
                continue
            
            console.print("\n⏳ Processing...\n")
            response = client.query(query)
            print_response(response)
        
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error: {e}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
