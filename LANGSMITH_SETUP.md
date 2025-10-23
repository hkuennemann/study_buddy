# LangSmith Setup Guide for Study Buddy

This guide shows you how to set up LangSmith tracking for your Study Buddy project to monitor performance, debug issues, and analyze your LangChain applications.

## 🚀 Quick Setup

### 1. Get Your LangSmith API Key

1. Go to [LangSmith](https://smith.langchain.com/)
2. Sign up or log in to your account
3. Navigate to Settings → API Keys
4. Create a new API key and copy it

### 2. Set Environment Variables

Add these to your `.env` file:

```bash
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=study-buddy

# Optional: Enable detailed tracing
LANGCHAIN_VERBOSE=true

# Optional: Enable debugging
LANGCHAIN_DEBUG=true
```

### 3. Install LangSmith (if not already installed)

```bash
pip install langsmith
```

## 📊 What You Can Track

### Question Generation Pipeline
- **Input**: Document chunks and processing parameters
- **Output**: Generated questions with quality metrics
- **Performance**: Processing time, token usage, model costs
- **Debugging**: Step-by-step refinement process

### Answer Generation Pipeline
- **Input**: Questions and vector store queries
- **Output**: Generated answers with retrieval context
- **Performance**: Retrieval accuracy, answer quality
- **Debugging**: Vector search results, context relevance

## 🔍 Monitoring Your Runs

### 1. View Runs in LangSmith Dashboard
- Go to your LangSmith dashboard
- Navigate to the "study-buddy" project
- See all your runs with timestamps, inputs, and outputs

### 2. Analyze Performance
- **Latency**: How long each operation takes
- **Token Usage**: Track API costs and usage
- **Error Rates**: Monitor failed requests
- **Quality Metrics**: Evaluate output quality

### 3. Debug Issues
- **Trace Details**: See exactly what happened in each step
- **Input/Output**: Review what was sent to and received from models
- **Error Messages**: Detailed error information
- **Context**: Full conversation history

## 🛠️ Advanced Configuration

### Custom Project Names
You can create separate projects for different workflows:

```python
# In your code
tracer = LangChainTracer(project_name="study-buddy-questions")
tracer = LangChainTracer(project_name="study-buddy-answers")
```

### Custom Tags
Add tags to categorize your runs:

```python
from langchain_core.tracers import LangChainTracer

tracer = LangChainTracer(
    project_name="study-buddy",
    tags=["production", "v1.0", "gemini"]
)
```

### Filtering and Search
In the LangSmith dashboard, you can:
- Filter by project, tags, or date range
- Search for specific runs or errors
- Compare different model performances
- Export data for analysis

## 📈 Performance Analytics

### Key Metrics to Monitor
1. **Question Generation Quality**
   - Number of questions generated
   - Question relevance and coverage
   - Processing time per document

2. **Answer Generation Performance**
   - Retrieval accuracy
   - Answer completeness
   - Response time per question

3. **Cost Analysis**
   - Token usage per operation
   - API costs by provider
   - Efficiency improvements

### Sample Queries
```python
# Get runs from the last 24 hours
runs = langsmith_client.list_runs(
    project_name="study-buddy",
    start_time=datetime.now() - timedelta(days=1)
)

# Get runs with errors
error_runs = langsmith_client.list_runs(
    project_name="study-buddy",
    filter='eq(status, "error")'
)
```

## 🐛 Debugging Common Issues

### 1. API Key Issues
- Verify your `LANGCHAIN_API_KEY` is correct
- Check that `LANGCHAIN_TRACING_V2=true`

### 2. No Traces Appearing
- Ensure you're running code with the tracer context
- Check your internet connection
- Verify the LangSmith endpoint

### 3. Performance Issues
- Monitor token usage in the dashboard
- Check for rate limiting
- Analyze latency patterns

## 🎯 Best Practices

### 1. Organize Your Projects
- Use descriptive project names
- Separate development and production
- Tag runs appropriately

### 2. Monitor Regularly
- Check the dashboard daily
- Set up alerts for errors
- Review performance trends

### 3. Optimize Based on Data
- Identify slow operations
- Optimize prompt engineering
- Adjust model parameters

## 📚 Additional Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Tracing Guide](https://python.langchain.com/docs/langsmith/)
- [Performance Monitoring Best Practices](https://docs.smith.langchain.com/tracing)

---

**Happy Monitoring! 📊✨**
