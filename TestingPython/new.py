import requests
import time


# Function to make a single POST request
def make_post_request():
    # Get real-time weather data
    weather_api_url = "http://api.weatherapi.com/v1/current.json?key=0a3d0ae5b7ab4a90971152553240503&q=30.9091,77.1087"
    weather_response = requests.get(weather_api_url)
    weather_data = weather_response.json()

    # Extract temperature and humidity from weather data
    temperature = weather_data['current']['temp_c']
    humidity = weather_data['current']['humidity']

    #mongodb atlas
    weather_document = {
        "temperature": temperature,
        "humidity": humidity,
        "timestamp": time.time()  # Add timestamp for reference
    }
    

    print("Data added to MongoDB.")

    # Prepare data for POST request
    post_data = {
        "data": {
            "temperature": str(temperature),
            "humidity": str(humidity)
        }
    }

    # Make POST request
    post_url = "https://major-aytg.onrender.com/add"
    response = requests.post(post_url, json=post_data)

    # Check the response
    if response.status_code == 200:
        print("POST request successful!")
    else:
        print(f"POST request failed with status code: {response.status_code}")

num_requests_per_day = 144  # 144 requests for every 10 minutes in a day

# Time interval between requests (in seconds)
time_interval = 600  # 10 minutes * 60 seconds

# Make requests
for _ in range(num_requests_per_day):
    make_post_request()
    time.sleep(time_interval)
