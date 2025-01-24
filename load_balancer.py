from flask import Flask, redirect, render_template_string
import random
import requests

app = Flask(__name__)

# List of backend endpoints to redirect to
backend_endpoints = [
    "https://s1.voiceslab.org",  # Backend 1
    "https://s2.voiceslab.org",  # Backend 2
]

# Health check endpoint on each backend server
health_check_url = "/contact"

def is_backend_healthy(backend_url):
    try:
        # Send a GET request to the health check endpoint
        response = requests.get(f"{backend_url}{health_check_url}")
        # Return True if the response is 200 OK, indicating the backend is healthy
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # If any exception occurs, consider the backend unhealthy
        return False

@app.route('/', methods=["GET", "POST"])
def load_balancer():
    # Filter healthy backends
    healthy_backends = [backend for backend in backend_endpoints if is_backend_healthy(backend)]

    if not healthy_backends:
        # If no backends are healthy, show a custom page with a redirect after 5 seconds
        return render_template_string("""
            <html>
                <head>
                    <title>Service Unavailable</title>
                    <meta http-equiv="refresh" content="5; url=https://celeb.voiceslab.org" />
                </head>
                <body>
                    <h1>No cloning server is available now.</h1>
                    <p>Please use our normal TTS app for the time being.</p>
                    <p>You will be redirected shortly...</p>
                </body>
            </html>
        """), 503

    # Randomly pick a healthy backend endpoint
    backend_server = random.choice(healthy_backends)
    
    # Perform the redirect to the root of the selected backend server
    return redirect(backend_server, code=302)

if __name__ == '__main__':
    app.run(debug=True, port=5555)
