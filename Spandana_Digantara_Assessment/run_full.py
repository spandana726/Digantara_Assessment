import sys
import os
import warnings
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore', category=FutureWarning)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from run_pipeline import main

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
