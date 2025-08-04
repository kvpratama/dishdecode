import requests
import json
import logging
import uuid
# from dishdecode.graph import graph

logger = logging.getLogger(__name__)

class LangGraphClient:
    def __init__(self, base_url="http://127.0.0.1:2024"):
        self.base_url = base_url
        logger.info(f"Initializing LangGraphClient with base URL: {base_url}")
        self.thread_id = self.create_thread()
        self.assistant_id = None
        logger.debug(f"Client initialized with thread_id: {self.thread_id}")
    
    def create_thread(self):
        """Create a new thread"""
        logger.debug("Attempting to create a new thread")
        try:
            response = requests.post(f"{self.base_url}/threads", json={})
            logger.debug(f"Thread creation response status: {response.status_code}")
            
            if response.status_code == 200:
                thread_id = response.json()["thread_id"]
                logger.info(f"Successfully created thread with ID: {thread_id}")
            else:
                error_msg = f"Failed to create thread: {response.text}"
                logger.error(error_msg)
                response.raise_for_status()
                
            return thread_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed while creating thread: {str(e)}", exc_info=True)
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse thread creation response: {str(e)}", exc_info=True)
            raise
    
    def create_assistant(self, graph_id):
        """Create a new assistant with the given graph ID"""
        logger.info(f"Creating new assistant for graph_id: {graph_id}")
        try:
            response = requests.post(
                f"{self.base_url}/assistants",
                headers={"Content-Type": "application/json"},
                json={
                    "assistant_id": "",
                    "graph_id": graph_id,
                    "config": {},
                    "metadata": {},
                    "if_exists": "raise",
                    "name": "",
                    "description": "null"
                }
            )
            response.raise_for_status()
            
            assistant_data = response.json()
            assistant_id = assistant_data["assistant_id"]
            self.assistant_id = assistant_id
            logger.info(f"Successfully created assistant with ID: {assistant_id}")
            return assistant_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create assistant: {str(e)}", exc_info=True)
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse assistant creation response: {str(e)}", exc_info=True)
            raise
    
    def run_graph(self):
        """Run the graph with input data"""
        logger.info("Starting graph execution")
        logger.debug(f"Thread ID: {self.thread_id}, Assistant ID: {self.assistant_id}")
        print(f"Thread ID: {self.thread_id}, Assistant ID: {self.assistant_id}")
        # logger.debug(f"Input data: {json.dumps(input_data, indent=2)}")
        # Read an image file to a base64 string
        import base64
        with open("C:\\Users\\IdeaPad\\Downloads\\menus\\menu1.jpg", "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        input_data = {
            "image": image_base64,
            "max_size": 640,
        }
        print(f"Input data: {json.dumps(input_data, indent=2)}")
        try:
            response = requests.post(
                f"{self.base_url}/threads/{self.thread_id}/runs/wait",
                headers={"Content-Type": "application/json"},
                json={
                    "assistant_id": f"{self.assistant_id}",
                    "input": input_data,
                },
            )
            response.raise_for_status()
            
            logger.info("Graph execution completed successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Graph execution failed: {str(e)}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            logger.error("Failed to parse graph execution response", exc_info=True)
            raise

if __name__ == "__main__":
    client = LangGraphClient()
    client.create_thread()
    client.create_assistant("main_graph")
    print(client.thread_id)
    print(client.assistant_id)
    print(client.run_graph())