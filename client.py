import requests
import time

def query_api(query):
    # Define the API endpoint
    url = 'http://127.0.0.1:8000/ai-query'
    
    # Define the headers
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # Define the payload
    payload = {
        'query': query
    }
    
    # Start the timer
    start_time = time.time()
    
    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)
    
    # End the timer
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    # Print the response from the API
    print(f"Query: {query}")
    print("Response from API:")
    print(response.json())
    
    # Print the time taken for the request
    print(f"Time taken for the request: {elapsed_time:.2f} seconds\n")

def main():
    # Set to track processed domains
    processed_domains = set()

    # Open the input file and read line by line
    with open("input.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            at_pos = line.find('@')
            if at_pos != -1:
                domain = line[at_pos+1:].strip()
                if domain and domain not in processed_domains:
                    query = f"who is the ceo of {domain}"
                    query_api(query)
                    processed_domains.add(domain)
                elif not domain:
                    print(f"Could not extract domain from email: {line.strip()}")
                else:
                    print(f"Domain already processed: {domain}")

if __name__ == "__main__":
    main()
