import streamlit as st
import logging
import sys
from logging.handlers import RotatingFileHandler

# Configure logging to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('streamlit_app.log', maxBytes=100000, backupCount=3),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.debug("Starting minimal Streamlit application...")

try:
    # Basic page config
    logger.debug("Setting page configuration...")
    st.set_page_config(
        page_title="Test App",
        page_icon="📈"
    )
    logger.debug("Page configuration complete")

    # Minimal content
    logger.debug("Adding content...")
    st.write("# Welcome to the Test App")
    st.write("If you can see this message, the server is running correctly.")
    logger.debug("Content added successfully")

except Exception as e:
    logger.error(f"Startup error: {str(e)}", exc_info=True)
    st.error("An error occurred during startup. Please check the logs.")

logger.debug("Startup sequence complete")