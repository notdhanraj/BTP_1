from pyngrok import ngrok

# Terminate any existing ngrok tunnels
ngrok.kill()

# IMPORTANT: Replace 'YOUR_NGROK_AUTH_TOKEN' with your actual ngrok authtoken.
# You can get this from your ngrok dashboard: https://dashboard.ngrok.com/get-started/your-authtoken
# You only need to run this line once per Colab session if the authtoken is not persisted.
ngrok.set_auth_token("35ZxI0Z2kbF9dE5tsPgNE8HGTgM_21XEXUHhJf1a7P2vCmMJo")

# Set the port for Streamlit (default is 8501)
STREAMLIT_PORT = 8501

# Start a ngrok tunnel to the Streamlit port
public_url = ngrok.connect(STREAMLIT_PORT)
print(f"Streamlit App URL: {public_url}")

# Run the Streamlit app
# This command will block until the Streamlit app is stopped or the cell execution is interrupted.
!streamlit run app.py --server.port $STREAMLIT_PORT --server.enableCORS false --server.enableXsrfProtection false
