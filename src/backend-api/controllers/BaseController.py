# controllers/BaseController.py - FIXED VERSION
from config import get_settings, Settings
import os
import random
import string
import logging

class BaseController:
    def __init__(self):
        # Initialize settings
        self.app_settings = get_settings()
        
        # Initialize logger - CRITICAL FIX
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize directories
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.files_dir = os.path.join(self.base_dir, "assets/files")
        self.database_dir = os.path.join(self.base_dir, "assets/database")
        
        # Ensure directories exist
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.database_dir, exist_ok=True)

    def generate_random_string(self, length: int = 12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def get_database_path(self, db_name: str):
        database_path = os.path.join(self.database_dir, db_name)
        if not os.path.exists(database_path):
            os.makedirs(database_path)
        return database_path
