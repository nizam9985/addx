from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import requests

load_dotenv()

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

# Test the agent
if __name__ == "__main__":
    print("=" * 60)
    
    
    # Test query 1: Calculator
    result = agent.invoke({"messages": "What is 25 * 4 + 100?"})
    tools = get_tool_calls(result)
    print(f"question asked: {result['messages'][0].content}")
    print(f"Tools called: {tools if tools else 'None'}")
    print(f"Result: {result['messages'][-1].content}"+ '\n')
    
    # Test query 2: Web search
    result = agent.invoke({"messages": "Search for C programming tutorials"})
    tools = get_tool_calls(result)
    print(f"question asked: {result['messages'][0].content}")
    print(f"Tools called: {tools if tools else 'None'}")
    print(f"Result: {result['messages'][-1].content}"+ '\n')
    
    # Test query 3: Weather
    result = agent.invoke({"messages": "What's the weather in mumbai?"})
    tools = get_tool_calls(result)
    print(f"question asked: {result['messages'][0].content}")
    print(f"Tools called: {tools if tools else 'None'}")
    print(f"Result: {result['messages'][-1].content}"+ '\n')

    # Test query 4: Search
    result = agent.invoke({"messages": "Search for Tajmahal"})
    tools = get_tool_calls(result)
    print(f"question asked: {result['messages'][0].content}")
    print(f"Tools called: {tools if tools else 'None'}")
    print(f"Result: {result['messages'][-1].content}"+ '\n')

    # Test query 5: Weather
    result = agent.invoke({"messages": "how about chennai"})
    tools = get_tool_calls(result)
    print(f"question asked: {result['messages'][0].content}")
    print(f"Tools called: {tools if tools else 'None'}")
    print(f"Result: {result['messages'][-1].content}"+ '\n')

    # Test query 6: Search
    result = agent.invoke({"messages": "how about charminar"})
    tools = get_tool_calls(result)
    print(f"question asked: {result['messages'][0].content}")
    print(f"Tools called: {tools if tools else 'None'}")
    print(f"Result: {result['messages'][-1].content}"+ '\n')
    
    # Test query 7: Testing memory
    result = agent.invoke({"messages": "tell me about my city"})
    tools = get_tool_calls(result)
    print(f"question asked: {result['messages'][0].content}")
    print(f"Tools called: {tools if tools else 'None'}")
    print(f"Result: {result['messages'][-1].content}"+ '\n')

    print("=" * 60)
