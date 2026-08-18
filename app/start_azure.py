#!/usr/bin/env python3
"""
Script to start Azure Text Analytics for Health container with proper environment variable handling.
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

def main():
    """Start the Azure Text Analytics container with environment variables."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get environment variables
    billing_endpoint = os.getenv('LANGUAGE_BILLING_ENDPOINT')
    billing_key = os.getenv('LANGUAGE_BILLING_KEY')
    
    # Check if environment variables are set
    if not billing_endpoint:
        print("Error: LANGUAGE_BILLING_ENDPOINT environment variable not set")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    if not billing_key:
        print("Error: LANGUAGE_BILLING_KEY environment variable not set")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    # Build the docker command
    docker_cmd = [
        "docker", "run", "--rm", "-it", "-p", "5001:5000",
        "mcr.microsoft.com/azure-cognitive-services/textanalytics/healthcare:latest",
        "Eula=accept",
        "enablelro=true",
        "rai_terms=accept",
        f"BILLING={billing_endpoint}",
        f"ApiKey={billing_key}"
    ]
    
    print("Starting Azure Text Analytics for Health container...")
    print(f"Billing endpoint: {billing_endpoint}")
    print(f"API key: {'*' * len(billing_key)}")  # Hide the key
    
    try:
        # Run the docker command
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nContainer stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
