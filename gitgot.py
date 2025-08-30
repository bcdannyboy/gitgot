#!/usr/bin/env python3
"""
gitgot - GitHub Repository Analyzer using OpenAI API
Analyzes public GitHub repositories and provides AI-powered insights
"""

import os
import sys
import json
import base64
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from urllib.parse import urlparse

import requests
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

class GitHubAnalyzer:
    """Analyzes GitHub repositories using OpenAI API"""
    
    def __init__(self, github_token: Optional[str] = None, openai_api_key: Optional[str] = None):
        """Initialize the analyzer with API tokens"""
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.session = requests.Session()
        
        # Set up GitHub headers
        self.github_headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitGot-Analyzer'
        }
        if self.github_token:
            self.github_headers['Authorization'] = f'token {self.github_token}'
        
        # Model configuration - using gpt-4.1 for its 1M token context window
        self.model = "gpt-4.1"  # 1M token context window for comprehensive analysis
        self.max_files_per_repo = 20  # Limit files to analyze per repo
        self.max_file_size = 50000  # Max file size in bytes to include
        
    def extract_username(self, github_url: str) -> str:
        """Extract username from GitHub URL"""
        parsed = urlparse(github_url)
        path_parts = parsed.path.strip('/').split('/')
        if path_parts and path_parts[0]:
            return path_parts[0]
        raise ValueError(f"Could not extract username from URL: {github_url}")
    
    def get_user_repos(self, username: str) -> List[Dict[str, Any]]:
        """Fetch all public repositories for a user"""
        repos = []
        page = 1
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Fetching repositories for {username}...", total=None)
            
            while True:
                url = f"https://api.github.com/users/{username}/repos"
                params = {
                    'page': page,
                    'per_page': 100,
                    'type': 'public',
                    'sort': 'updated'
                }
                
                response = self.session.get(url, headers=self.github_headers, params=params)
                
                if response.status_code == 404:
                    console.print(f"[red]User {username} not found[/red]")
                    return []
                elif response.status_code != 200:
                    console.print(f"[yellow]Warning: GitHub API returned status {response.status_code}[/yellow]")
                    break
                
                page_repos = response.json()
                if not page_repos:
                    break
                
                repos.extend(page_repos)
                progress.update(task, description=f"Fetched {len(repos)} repositories...")
                
                # Check if there are more pages
                if 'Link' not in response.headers or 'rel="next"' not in response.headers['Link']:
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting
        
        return repos
    
    def get_repo_content(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch repository content including README and sample code files"""
        content = {
            'name': repo['name'],
            'description': repo.get('description', ''),
            'language': repo.get('language', 'Unknown'),
            'stars': repo.get('stargazers_count', 0),
            'forks': repo.get('forks_count', 0),
            'created_at': repo.get('created_at', ''),
            'updated_at': repo.get('updated_at', ''),
            'topics': repo.get('topics', []),
            'readme': None,
            'files': [],
            'structure': None
        }
        
        # Get README
        readme_url = f"{repo['url']}/readme"
        readme_response = self.session.get(readme_url, headers=self.github_headers)
        if readme_response.status_code == 200:
            readme_data = readme_response.json()
            if 'content' in readme_data:
                try:
                    content['readme'] = base64.b64decode(readme_data['content']).decode('utf-8')
                except:
                    content['readme'] = "Could not decode README"
        
        # Get repository tree structure
        tree_url = f"{repo['url']}/git/trees/{repo['default_branch']}?recursive=1"
        tree_response = self.session.get(tree_url, headers=self.github_headers)
        
        if tree_response.status_code == 200:
            tree_data = tree_response.json()
            
            # Create a simple directory structure
            structure = []
            files_to_fetch = []
            
            for item in tree_data.get('tree', [])[:100]:  # Limit to first 100 items
                if item['type'] == 'blob':
                    structure.append(f"📄 {item['path']}")
                    
                    # Select important files to fetch
                    path = item['path']
                    if (self._is_important_file(path) and 
                        item.get('size', 0) < self.max_file_size and
                        len(files_to_fetch) < self.max_files_per_repo):
                        files_to_fetch.append(item)
                elif item['type'] == 'tree':
                    structure.append(f"📁 {item['path']}/")
            
            content['structure'] = '\n'.join(structure[:50])  # Limit structure display
            
            # Fetch selected files
            for file_item in files_to_fetch:
                file_url = file_item['url']
                file_response = self.session.get(file_url, headers=self.github_headers)
                
                if file_response.status_code == 200:
                    file_data = file_response.json()
                    if 'content' in file_data:
                        try:
                            file_content = base64.b64decode(file_data['content']).decode('utf-8')
                            content['files'].append({
                                'path': file_item['path'],
                                'content': file_content[:10000]  # Limit file content
                            })
                        except:
                            pass
                
                time.sleep(0.1)  # Rate limiting
        
        return content
    
    def _is_important_file(self, path: str) -> bool:
        """Determine if a file is important for analysis"""
        important_extensions = [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
            '.yml', '.yaml', '.json', '.toml', '.ini', '.conf',
            '.md', '.txt', '.rst', 'Dockerfile', 'Makefile'
        ]
        
        important_names = [
            'package.json', 'requirements.txt', 'setup.py', 'pyproject.toml',
            'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', '.gitignore'
        ]
        
        path_lower = path.lower()
        
        # Check if it's a known important file
        if any(path.endswith(name) for name in important_names):
            return True
        
        # Check extensions
        return any(path_lower.endswith(ext) for ext in important_extensions)
    
    def analyze_repository(self, repo_content: Dict[str, Any]) -> str:
        """Use OpenAI to analyze a repository"""
        # Prepare content for analysis
        analysis_prompt = self._create_analysis_prompt(repo_content)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert software engineer and code reviewer. 
                        Analyze the provided repository information and create a comprehensive assessment covering:
                        1. Project Purpose and Functionality
                        2. Code Quality and Architecture
                        3. Completeness and Maturity Level
                        4. Technical Stack and Dependencies
                        5. Strengths and Notable Features
                        6. Areas for Improvement
                        7. Overall Assessment Score (1-10)
                        
                        Be specific, technical, and constructive in your analysis."""
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error during analysis: {str(e)}"
    
    def _create_analysis_prompt(self, repo_content: Dict[str, Any]) -> str:
        """Create a detailed prompt for repository analysis"""
        prompt_parts = [
            f"Repository: {repo_content['name']}",
            f"Description: {repo_content['description'] or 'No description provided'}",
            f"Primary Language: {repo_content['language']}",
            f"Stars: {repo_content['stars']} | Forks: {repo_content['forks']}",
            f"Topics: {', '.join(repo_content['topics']) if repo_content['topics'] else 'None'}",
            f"Last Updated: {repo_content['updated_at']}",
            ""
        ]
        
        if repo_content['readme']:
            prompt_parts.append("README CONTENT:")
            prompt_parts.append(repo_content['readme'][:5000])  # Limit README length
            prompt_parts.append("")
        
        if repo_content['structure']:
            prompt_parts.append("REPOSITORY STRUCTURE (sample):")
            prompt_parts.append(repo_content['structure'][:2000])
            prompt_parts.append("")
        
        if repo_content['files']:
            prompt_parts.append("KEY FILES:")
            for file_info in repo_content['files'][:10]:  # Limit to 10 files
                prompt_parts.append(f"\n--- File: {file_info['path']} ---")
                prompt_parts.append(file_info['content'][:2000])  # Limit content per file
        
        return "\n".join(prompt_parts)
    
    def generate_profile_summary(self, analyses: List[Dict[str, Any]]) -> str:
        """Generate an overall profile summary of the developer"""
        if not analyses:
            return "No repositories analyzed."
        
        # Prepare summary prompt
        summary_data = []
        for analysis in analyses:
            summary_data.append({
                'name': analysis['name'],
                'language': analysis['language'],
                'description': analysis['description'],
                'stars': analysis['stars'],
                'analysis_summary': analysis['analysis'][:500]  # Brief excerpt
            })
        
        prompt = f"""Based on the analysis of {len(analyses)} repositories, create a comprehensive developer profile that includes:

1. **Technical Expertise**: Primary languages, frameworks, and technologies
2. **Project Domains**: Types of applications and problem domains
3. **Coding Style and Practices**: Observed patterns, quality indicators
4. **Open Source Contribution Profile**: Activity level, project maintenance
5. **Strengths and Specializations**: What this developer excels at
6. **Growth Areas**: Suggested areas for skill development
7. **Overall Developer Profile**: A summary characterization

Repository summaries:
{json.dumps(summary_data, indent=2)[:10000]}

Provide a detailed, professional assessment suitable for a technical portfolio review."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior technical recruiter and engineering manager with deep expertise in evaluating developer portfolios."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def analyze_github_profile(self, github_url: str, limit: Optional[int] = None):
        """Main method to analyze a GitHub profile"""
        # Extract username
        try:
            username = self.extract_username(github_url)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return
        
        console.print(Panel(f"[bold blue]Analyzing GitHub Profile: {username}[/bold blue]"))
        
        # Fetch repositories
        repos = self.get_user_repos(username)
        if not repos:
            console.print("[yellow]No repositories found.[/yellow]")
            return
        
        console.print(f"[green]Found {len(repos)} public repositories[/green]")
        
        # Apply limit if specified
        if limit:
            repos = repos[:limit]
            console.print(f"[yellow]Analyzing top {limit} repositories[/yellow]")
        
        # Analyze each repository
        analyses = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for i, repo in enumerate(repos, 1):
                task = progress.add_task(
                    f"Analyzing {repo['name']} ({i}/{len(repos)})...",
                    total=1
                )
                
                # Fetch repository content
                repo_content = self.get_repo_content(repo)
                
                # Analyze with OpenAI
                analysis = self.analyze_repository(repo_content)
                
                analyses.append({
                    'name': repo['name'],
                    'url': repo['html_url'],
                    'description': repo.get('description', ''),
                    'language': repo.get('language', 'Unknown'),
                    'stars': repo.get('stargazers_count', 0),
                    'analysis': analysis
                })
                
                progress.update(task, completed=1)
                time.sleep(1)  # Rate limiting for OpenAI
        
        # Display individual analyses
        console.print("\n[bold]Repository Analyses:[/bold]\n")
        
        for analysis in analyses:
            console.print(Panel(
                Markdown(f"""
## {analysis['name']}
**URL:** {analysis['url']}  
**Language:** {analysis['language']} | **Stars:** ⭐ {analysis['stars']}

{analysis['analysis']}
                """),
                title=f"[bold]{analysis['name']}[/bold]",
                border_style="blue"
            ))
            console.print()
        
        # Generate overall profile summary
        console.print("\n[bold cyan]Generating Developer Profile Summary...[/bold cyan]")
        profile_summary = self.generate_profile_summary(analyses)
        
        console.print(Panel(
            Markdown(profile_summary),
            title=f"[bold green]Developer Profile: {username}[/bold green]",
            border_style="green"
        ))
        
        # Save results to file
        output_file = f"gitgot_analysis_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_data = {
            'username': username,
            'analysis_date': datetime.now().isoformat(),
            'repositories_analyzed': len(analyses),
            'individual_analyses': analyses,
            'profile_summary': profile_summary
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓ Analysis saved to {output_file}[/green]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze GitHub repositories using OpenAI API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://github.com/bcdannyboy
  %(prog)s https://github.com/username --limit 5
  %(prog)s username  # Just the username also works
        """
    )
    
    parser.add_argument(
        'github_url',
        help='GitHub profile URL or username to analyze'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of repositories to analyze',
        default=None
    )
    
    parser.add_argument(
        '--model',
        help='OpenAI model to use (default: gpt-4.1)',
        default='gpt-4.1'
    )
    
    args = parser.parse_args()
    
    # Handle plain username input
    if not args.github_url.startswith('http'):
        args.github_url = f'https://github.com/{args.github_url}'
    
    try:
        analyzer = GitHubAnalyzer()
        analyzer.model = args.model
        analyzer.analyze_github_profile(args.github_url, args.limit)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()