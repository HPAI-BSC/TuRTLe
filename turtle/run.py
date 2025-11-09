import os
import sys

from src.utils.executor import WorkflowExecutor
from src.utils.logger import init_logger
from src.utils.parse_arguments import parse_arguments

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = init_logger(__name__)

def main():
    # Parse command-line arguments
    args = parse_arguments()
    print(args)
    
    # Execute the workflow
    executor = WorkflowExecutor(args, logger)
    executor.execute()

if __name__ == "__main__":
    main()
