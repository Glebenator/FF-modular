import requests
import logging
import json
from datetime import datetime
import threading
from queue import Queue, Empty
import time
import os
from pathlib import Path

class JSONSender:
    def __init__(self, server_url):
        """
        Initialize the sender service
        
        Args:
            server_url: URL of the server to send JSON data to
        """
        self.server_url = server_url
        self.send_queue = Queue()
        self.retry_queue = Queue()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('json_sender.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create directory for JSON backups
        self.json_dir = Path('recordings/json_data')
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # Start sender thread
        self.running = True
        self.send_thread = threading.Thread(target=self._process_queues)
        self.send_thread.daemon = True
        self.send_thread.start()

    def _save_json_locally(self, session_id, data):
        """Save JSON data to local file"""
        try:
            filename = self.json_dir / f"session_{session_id}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"Saved JSON backup for session {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON backup for session {session_id}: {e}")
            return False

    def _send_json(self, session_id, data):
        """Send JSON data to server"""
        try:
            # Add metadata about the sending attempt
            data_to_send = {
                'data': data,
                'sender_metadata': {
                    'sent_timestamp': datetime.now().isoformat(),
                    'session_id': session_id
                }
            }
            
            # Save locally first
            self._save_json_locally(session_id, data_to_send)
            
            response = requests.post(
                self.server_url,
                json=data_to_send,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully sent session {session_id}")
                # Save success status
                status_data = {
                    "original_data": data_to_send,
                    "send_status": {
                        "success": True,
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status_code
                    }
                }
                self._save_json_locally(f"{session_id}_success", status_data)
                return True
            else:
                self.logger.warning(
                    f"Failed to send session {session_id}, status code: {response.status_code}"
                )
                # Save failure status
                status_data = {
                    "original_data": data_to_send,
                    "send_status": {
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}"
                    }
                }
                self._save_json_locally(f"{session_id}_failed", status_data)
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending session {session_id}: {e}")
            # Save error status
            status_data = {
                "original_data": data_to_send,
                "send_status": {
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
            self._save_json_locally(f"{session_id}_error", status_data)
            return False

    def _process_queues(self):
        """Process both send and retry queues"""
        while self.running:
            try:
                # Try to get an item from either queue
                try:
                    session_id, data = self.send_queue.get_nowait()
                except Empty:
                    if not self.retry_queue.empty():
                        session_id, data, retries = self.retry_queue.get_nowait()
                        if retries < 3:  # Maximum 3 retry attempts
                            if not self._send_json(session_id, data):
                                # Failed, put back in retry queue with increased retry count
                                self.retry_queue.put((session_id, data, retries + 1))
                                # Save retry status
                                retry_data = {
                                    "original_data": data,
                                    "retry_status": {
                                        "retry_count": retries + 1,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                }
                                self._save_json_locally(f"{session_id}_retry_{retries + 1}", retry_data)
                            self.retry_queue.task_done()
                    time.sleep(1)  # Sleep if both queues are empty
                    continue

                # Process item from send queue
                if not self._send_json(session_id, data):
                    # Failed, add to retry queue
                    self.retry_queue.put((session_id, data, 0))
                self.send_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing queues: {e}")
            
            time.sleep(0.1)

    def send_recording_data(self, session_id, data):
        """Queue data to be sent"""
        self.send_queue.put((session_id, data))

    def stop(self):
        """Stop the sender service"""
        self.running = False
        if self.send_thread.is_alive():
            self.send_thread.join()