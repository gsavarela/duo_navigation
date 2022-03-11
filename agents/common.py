'''The module provides common functionality for classes'''
import dill
import re
from pathlib import Path
snakefy_pattern = re.compile(r'(?<!^)(?=[A-Z])')

# coverts 'CamelCaseName' to 'camel_case_name'
def snakefy(x): return snakefy_pattern.sub('_', x).lower()

class Serializable(object):
    '''Extend serializable to persist the agent'''  
    def save_checkpoints(self, chkpt_dir_path, chkpt_num):
        class_name = snakefy(type(self).__name__)
        file_path = Path(chkpt_dir_path) / chkpt_num / f'{class_name}.chkpt'  
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, mode='wb') as f:
            dill.dump(self, f)
        
    @classmethod
    def load_checkpoint(cls, chkpt_dir_path, chkpt_num):
        class_name = cls.__name__.lower()
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'{class_name}.chkpt'  
        with file_path.open(mode='rb') as f:
            new_instance = dill.load(f)

        return new_instance
