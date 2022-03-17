import dill
from pathlib import Path


class Serializable(object):
    def save_checkpoints(self, chkpt_dir_path, chkpt_num):
        class_name = type(self).__name__.lower()
        file_path = Path(chkpt_dir_path) / chkpt_num / f"{class_name}.chkpt"
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, mode="wb") as f:
            dill.dump(self, f)

    @classmethod
    def load_checkpoint(cls, chkpt_dir_path, chkpt_num):
        class_name = cls.__name__.lower()
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f"{class_name}.chkpt"
        with file_path.open(mode="rb") as f:
            new_instance = dill.load(f)

        return new_instance
