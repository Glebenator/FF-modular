import requests
import logging
import json
from datetime import datetime
import threading
from queue import Queue, Empty, Full
import time
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import socket
import backoff

@dataclass
class QueueItem:
    """Data structure for items in the queue"""
    session_id: str
    data: Dict[str, Any]
    timestamp: float
    retries: int = 0

class JSONSender:
    def __init__(self, server_url: str, 
                 max_queue_size: int = 1000,
                 max_retries: int = 3,
                 retry_delay: float = 5.0,
                 connection_timeout: float = 10.0,
                 queue_timeout: float = 0.5):
        """
        Initialize the sender service
        
        Args:
            server_url: URL of the server to send JSON data to
            max_queue_size: Maximum number of items in queues
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (seconds)
            connection_timeout: Timeout for server connections
            queue_timeout: Timeout for queue operations
        """
        self._validate_url(server_url)
        self.server_url = server_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        self.queue_timeout = queue_timeout
        
        # Initialize queues with size limits
        self.send_queue: Queue = Queue(maxsize=max_queue_size)
        self.retry_queue: Queue = Queue(maxsize=max_queue_size)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize state
        self.running = True
        self._initialize_health_metrics()
        
        # Start processing threads
        self._start_threads()

    def _validate_url(self, url: str) -> None:
        """Validate the server URL format"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")
        except Exception as e:
            raise ValueError(f"Invalid server URL: {e}")

    def _setup_logging(self) -> None:
        """Configure logging"""
        try:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.FileHandler('json_sender.log')
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        except Exception as e:
            raise RuntimeError(f"Failed to setup logging: {e}")

    def _initialize_health_metrics(self) -> None:
        """Initialize health monitoring metrics"""
        self.health_metrics = {
            'successful_sends': 0,
            'failed_sends': 0,
            'retry_attempts': 0,
            'last_successful_send': None,
            'last_error': None
        }
        self.metrics_lock = threading.Lock()

    def _start_threads(self) -> None:
        """Start the processing threads"""
        self.send_thread = threading.Thread(target=self._process_queues)
        self.send_thread.daemon = True
        self.send_thread.start()
        
        self.health_monitor_thread = threading.Thread(target=self._monitor_health)
        self.health_monitor_thread.daemon = True
        self.health_monitor_thread.start()

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, socket.error),
        max_tries=3,
        max_time=30
    )
    def _send_json(self, item: QueueItem) -> bool:
        """
        Send JSON data to server with exponential backoff retry
        
        Returns:
            bool: True if send was successful
        """
        try:
            data_to_send = {
                'data': item.data,
                'sender_metadata': {
                    'sent_timestamp': datetime.now().isoformat(),
                    'session_id': item.session_id,
                    'retry_count': item.retries
                }
            }
            
            response = requests.post(
                self.server_url,
                json=data_to_send,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'JSONSender/1.0'
                },
                timeout=self.connection_timeout
            )
            
            response.raise_for_status()
            
            with self.metrics_lock:
                self.health_metrics['successful_sends'] += 1
                self.health_metrics['last_successful_send'] = time.time()
            
            self.logger.info(f"Successfully sent session {item.session_id}")
            return True
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error for session {item.session_id}: {e}")
            with self.metrics_lock:
                self.health_metrics['failed_sends'] += 1
                self.health_metrics['last_error'] = str(e)
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending session {item.session_id}: {e}")
            with self.metrics_lock:
                self.health_metrics['failed_sends'] += 1
                self.health_metrics['last_error'] = str(e)
            return False

    def _process_queues(self) -> None:
        """Process both send and retry queues"""
        while self.running:
            try:
                # First try to process retry queue
                try:
                    item = self.retry_queue.get(timeout=self.queue_timeout)
                    if item.retries < self.max_retries:
                        if not self._send_json(item):
                            item.retries += 1
                            self._add_to_retry_queue(item)
                        with self.metrics_lock:
                            self.health_metrics['retry_attempts'] += 1
                    else:
                        self.logger.error(
                            f"Max retries exceeded for session {item.session_id}"
                        )
                    self.retry_queue.task_done()
                except Empty:
                    pass

                # Then process send queue
                try:
                    item = self.send_queue.get(timeout=self.queue_timeout)
                    if not self._send_json(item):
                        self._add_to_retry_queue(item)
                    self.send_queue.task_done()
                except Empty:
                    time.sleep(0.1)  # Prevent tight loop if both queues empty
                    
            except Exception as e:
                self.logger.error(f"Error in queue processing: {e}")
                time.sleep(1)  # Prevent rapid cycling on persistent errors

    def _add_to_retry_queue(self, item: QueueItem) -> None:
        """Add item to retry queue with overflow protection"""
        try:
            if self.retry_queue.qsize() < self.retry_queue.maxsize:
                self.retry_queue.put(item, timeout=self.queue_timeout)
            else:
                self.logger.error(
                    f"Retry queue full, dropping session {item.session_id}"
                )
        except Full:
            self.logger.error(
                f"Timeout adding to retry queue, dropping session {item.session_id}"
            )

    def _monitor_health(self) -> None:
        """Monitor and log health metrics"""
        while self.running:
            try:
                queue_sizes = {
                    'send_queue': self.send_queue.qsize(),
                    'retry_queue': self.retry_queue.qsize()
                }
                
                with self.metrics_lock:
                    metrics = self.health_metrics.copy()
                
                self.logger.info(
                    f"Health metrics - Queues: {queue_sizes}, "
                    f"Successful sends: {metrics['successful_sends']}, "
                    f"Failed sends: {metrics['failed_sends']}, "
                    f"Retry attempts: {metrics['retry_attempts']}"
                )
                
                time.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(60)

    def send_recording_data(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Queue data to be sent
        
        Args:
            session_id: Unique identifier for this recording session
            data: Dictionary of data to send
        """
        try:
            item = QueueItem(
                session_id=session_id,
                data=data,
                timestamp=time.time()
            )
            
            if self.send_queue.qsize() < self.send_queue.maxsize:
                self.send_queue.put(item, timeout=self.queue_timeout)
            else:
                self.logger.error(
                    f"Send queue full, dropping session {session_id}"
                )
                
        except Full:
            self.logger.error(
                f"Timeout adding to send queue, dropping session {session_id}"
            )
        except Exception as e:
            self.logger.error(f"Error queueing data: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health metrics"""
        with self.metrics_lock:
            return self.health_metrics.copy()

    def stop(self) -> None:
        """
        Stop the sender service and wait for queues to empty
        """
        try:
            self.running = False
            
            # Wait for queues to empty with timeout
            timeout = 30  # seconds
            start_time = time.time()
            
            while (not self.send_queue.empty() or not self.retry_queue.empty()):
                if time.time() - start_time > timeout:
                    self.logger.warning("Timeout waiting for queues to empty")
                    break
                time.sleep(0.1)
            
            if self.send_thread.is_alive():
                self.send_thread.join(timeout=5)
            if self.health_monitor_thread.is_alive():
                self.health_monitor_thread.join(timeout=5)
                
        except Exception as e:
            self.logger.error(f"Error stopping sender: {e}")