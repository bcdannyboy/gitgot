# gitgot

Analyzes GitHub account public repositories and provides insights on the repositories / account

## Overview

gitgot is a Python tool that leverages OpenAI's GPT-4.1 model to provide comprehensive analysis of GitHub profiles. It examines public repositories to generate detailed insights about code quality, project completeness, technical expertise, and overall developer profile.

## Features

- 🔍 **Deep Repository Analysis**: Examines README files, code structure, and key source files
- 🤖 **AI-Powered Insights**: Uses GPT-4.1's 1M token context window for thorough analysis
- 📊 **Comprehensive Assessment**: Evaluates code quality, architecture, completeness, and technical stack
- 👤 **Developer Profile Generation**: Creates an overall profile summarizing expertise and project patterns
- 🎨 **Rich Terminal Output**: Color-coded, formatted output with progress indicators
- 💾 **JSON Export**: Saves complete analysis results for future reference

## Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) GitHub personal access token for higher rate limits

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gitgot.git
cd gitgot
```

2. Install dependencies:
```bash
pip install openai requests python-dotenv rich
```

3. Create a `.env` file in the project directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
# Optional: For higher GitHub API rate limits (60 → 5000 requests/hour)
GITHUB_TOKEN=your_github_personal_access_token_here
```

## Usage

### Basic Usage
```bash
# Analyze using GitHub URL
python gitgot.py https://github.com/bcdannyboy

# Or just use the username
python gitgot.py bcdannyboy
```

### Advanced Options
```bash
# Limit analysis to top N repositories
python gitgot.py bcdannyboy --limit 5

# Use a different OpenAI model
python gitgot.py bcdannyboy --model gpt-4o

# Analyze your own profile (assuming you set GITHUB_TOKEN)
python gitgot.py yourusername
```

### Example Output Structure

The tool generates both terminal output and a JSON file with the following structure:

```json
{
  "username": "bcdannyboy",
  "analysis_date": "2025-01-01T12:00:00",
  "repositories_analyzed": 10,
  "individual_analyses": [
    {
      "name": "repo-name",
      "url": "https://github.com/user/repo",
      "language": "Python",
      "stars": 42,
      "analysis": "Detailed AI-generated analysis..."
    }
  ],
  "profile_summary": "Comprehensive developer profile..."
}
```

## What Gets Analyzed

For each repository, gitgot examines:

1. **Project Documentation**: README content and structure
2. **Code Architecture**: File organization and design patterns
3. **Technical Stack**: Languages, frameworks, and dependencies
4. **Code Quality**: Style, documentation, and best practices
5. **Project Maturity**: Completeness, maintenance, and activity
6. **Unique Features**: Notable implementations or approaches

The final profile summary includes:

- Primary technical expertise and languages
- Project domains and application types
- Coding style and observed patterns
- Open source contribution patterns
- Strengths and specializations
- Suggested growth areas

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `GITHUB_TOKEN` (optional): GitHub personal access token for increased rate limits

### Command Line Arguments

- `github_url`: GitHub profile URL or username
- `--limit`: Maximum number of repositories to analyze
- `--model`: OpenAI model to use (default: gpt-4.1)

## Limitations

- Only analyzes public repositories
- Rate limited by GitHub API (60/hour without token, 5000/hour with token)
- OpenAI API costs apply (approximately $0.01-0.10 per repository depending on size)
- Large repositories may be truncated to fit within token limits

## Cost Considerations

gitgot uses OpenAI's GPT-4.1 model which charges per token. Typical costs:
- Small repository (< 10 files): ~$0.01-0.02
- Medium repository (10-50 files): ~$0.03-0.05
- Large repository (50+ files): ~$0.05-0.10
- Profile summary: ~$0.02-0.05

## Contributing

Contributions are welcome! Some areas for potential enhancement:

- **Comparative Analysis**: Compare multiple GitHub profiles
- **Organization Support**: Analyze organization/enterprise accounts
- **Additional Export Formats**: HTML reports, PDF generation
- **Caching Layer**: Reduce API calls for previously analyzed repos
- **More Granular Analysis**: Security patterns, test coverage, documentation quality
- **Web Interface**: Browser-based UI for easier access

## Technical Details

- **Language Model**: GPT-4.1 with 1M token context window
- **File Selection**: Prioritizes important files (source code, configs, documentation)
- **Rate Limiting**: Built-in delays to respect API limits
- **Error Handling**: Graceful handling of API failures and missing data

## Output Files

Analysis results are saved as `gitgot_analysis_[username]_[timestamp].json` in the current directory.

## Requirements

- `openai`: OpenAI Python client library
- `requests`: HTTP library for GitHub API calls
- `python-dotenv`: Environment variable management
- `rich`: Terminal formatting and progress indicators

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**: Ensure `.env` file exists with valid `OPENAI_API_KEY`
2. **Rate limiting errors**: Add a GitHub token or reduce `--limit` parameter
3. **Timeout errors**: Large repositories may timeout; try using `--limit` to analyze fewer repos
4. **Cost concerns**: Use `--limit` to control the number of repositories analyzed

### Debug Mode

For verbose output, check the generated JSON file which contains full analysis details.
