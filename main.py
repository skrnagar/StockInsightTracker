import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting minimal Streamlit app...")

    # Page configuration
    st.set_page_config(
        page_title="Stock Analysis",
        page_icon="📈",
        layout="wide"
    )
    logger.info("Page configuration complete")

    # Display welcome message
    st.title("Welcome to Stock Analysis")
    st.write("Loading basic functionality...")
    logger.info("Basic UI elements displayed")

except Exception as e:
    logger.error(f"Critical error in app initialization: {str(e)}")
    st.error("Failed to initialize the application. Please refresh the page.")

logger.info("Application startup sequence complete")

if __name__ == "__main__":
    logger.info("Application startup complete")