from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
from pathlib import Path

load_dotenv()

# ============================================================================
# CONVERSATIONAL MEMORY SYSTEM
# ============================================================================

class ConversationMemory:
    """Manages conversation history for context-aware agent responses."""
    
    def __init__(self, memory_file="agent_conversation_memory.json"):
        self.memory_file = Path(memory_file)
        self.conversation_history = []  # Current session history
        self.load_memory()
    
    def load_memory(self):
        """Load previous conversation history from file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = data.get('history', [])
            except Exception as e:
                print(f"Error loading memory: {e}")
    
    def save_memory(self):
        """Save conversation history to file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'history': self.conversation_history,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def add_message(self, role: str, content: str, tools_used: list = None):
        """Add a message to conversation history."""
        message = {
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content,
            'tools_used': tools_used or []
        }
        self.conversation_history.append(message)
        self.save_memory()
    
    def get_context(self, limit: int = 5) -> str:
        """Get recent conversation context for the agent."""
        recent = self.conversation_history[-limit:]
        if not recent:
            return ""
        
        context = ""
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content']
            # Don't truncate - keep full content for better context
            context += f"{role}: {content}\n\n"
        return context
    
    def get_full_history(self) -> list:
        """Get complete conversation history."""
        return self.conversation_history
    
    def clear_memory(self):
        """Clear all conversation history."""
        self.conversation_history = []
        if self.memory_file.exists():
            self.memory_file.unlink()
    
    def get_stats(self) -> dict:
        """Get conversation statistics."""
        return {
            'total_messages': len(self.conversation_history),
            'user_messages': len([m for m in self.conversation_history if m['role'] == 'user']),
            'assistant_messages': len([m for m in self.conversation_history if m['role'] == 'assistant']),
            'tools_used': list(set([tool for msg in self.conversation_history for tool in msg.get('tools_used', [])]))
        }

# Initialize memory
memory = ConversationMemory()

# Define a simple web search tool
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(
            f"https://api.duckduckgo.com/",
            params={"q": query, "format": "json"},
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            return f"Search results for '{query}': {response.text[:500]}"
        return f"Could not fetch results for '{query}'"
    except Exception as e:
        return f"Error searching: {str(e)}"

# Define a calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Define a weather tool
@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # Using Open-Meteo API (free, no API key required)
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                location = data["results"][0]
                lat = location.get("latitude")
                lon = location.get("longitude")
                city_name = location.get("name")
                country = location.get("country")
                
                # Get weather data
                weather_response = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,weather_code,wind_speed_10m",
                        "temperature_unit": "celsius"
                    },
                    headers=headers,
                    timeout=5
                )
                if weather_response.status_code == 200:
                    weather_data = weather_response.json()
                    current = weather_data.get("current", {})
                    temp = current.get("temperature_2m", "N/A")
                    wind = current.get("wind_speed_10m", "N/A")
                    return f"Weather in {city_name}, {country}: Temperature: {temp}°C, Wind Speed: {wind} km/h"
                return f"Could not fetch weather for {city}"
            return f"City '{city}' not found"
        return f"Error fetching weather data"
    except Exception as e:
        return f"Error getting weather: {str(e)}"



# Initialize the model
model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create the agent with all tools
agent = create_react_agent(model, [web_search, calculator, get_weather])

# Function to extract tool calls from agent response
def get_tool_calls(result):
    """Extract tool names called during agent execution."""
    tools_called = []
    for message in result['messages']:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tools_called.append(tool_call.get('name', 'unknown'))
    return tools_called

# Function to run agent with memory
def run_agent_with_memory(query: str, agent, memory):
    """Run agent with conversation context from memory."""
    # Get conversation context
    context = memory.get_context(limit=6)
    
    # Create a comprehensive system prompt with memory
    system_prompt = """You are a helpful assistant with access to conversation history. 
Always remember and reference information the user has shared with you in previous messages.
If the user mentions their city, location, or any personal information, remember it for future questions.
When answering questions about 'my city' or 'there', use the information they previously provided.
"""
    
    # Enhance query with context and system prompt
    if context:
        enhanced_query = f"{system_prompt}\n\nConversation History:\n{context}\n\nCurrent Question: {query}"
    else:
        enhanced_query = f"{system_prompt}\n\nQuestion: {query}"
    
    # Run agent
    result = agent.invoke({"messages": enhanced_query})
    tools = get_tool_calls(result)
    answer = result['messages'][-1].content
    
    # Store in memory
    memory.add_message('user', query, [])
    memory.add_message('assistant', answer, tools)
    
    return result, tools

# Helper function for formatted output
def print_query_result(query_num, query, result, tools):
    """Print formatted query and result."""
    print(f"\n{'─' * 70}")
    print(f"📝 Query {query_num}: {query}")
    print(f"{'─' * 70}")
    print(f"🔧 Tools Used: {', '.join(tools) if tools else 'None'}")
    print(f"\n💬 Response:")
    print(f"{result['messages'][-1].content}")

# Test the agent
if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("🤖 AGENT WITH CONVERSATIONAL MEMORY SYSTEM")
    print("═" * 70)
    
    # Test query 1: Calculator
    result, tools = run_agent_with_memory("What is 25 * 4 + 100?", agent, memory)
    print_query_result(1, "What is 25 * 4 + 100?", result, tools)
    
    # Test query 2: Web search
    result, tools = run_agent_with_memory("Search for C programming tutorials", agent, memory)
    print_query_result(2, "Search for C programming tutorials", result, tools)
    
    # Test query 3: Remember user info
    result, tools = run_agent_with_memory("I live in London, England", agent, memory)
    print_query_result(3, "I live in London, England", result, tools)
    
    # Test query 4: Weather
    result, tools = run_agent_with_memory("What's the weather in mumbai?", agent, memory)
    print_query_result(4, "What's the weather in Mumbai?", result, tools)

    # Test query 5: Search
    result, tools = run_agent_with_memory("Search for Tajmahal", agent, memory)
    print_query_result(5, "Search for Tajmahal", result, tools)

    # Test query 6: Use memory - Ask about the city
    result, tools = run_agent_with_memory("Tell me about the city where I live", agent, memory)
    print_query_result(6, "Tell me about my city (Using Memory)", result, tools)
    
    # Test query 7: More memory usage - Ask about weather
    result, tools = run_agent_with_memory("What is the weather in the city I mentioned?", agent, memory)
    print_query_result(7, "What's the weather in my city? (Using Memory)", result, tools)

    # Display memory statistics
    print(f"\n\n{'═' * 70}")
    print("📊 CONVERSATION MEMORY STATISTICS")
    print(f"{'═' * 70}")
    stats = memory.get_stats()
    print(f"\n✅ Total Messages: {stats['total_messages']}")
    print(f"\n💾 Memory saved to: agent_conversation_memory.json")
    print(f"{'═' * 70}\n")
