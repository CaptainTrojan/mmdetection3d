from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS
import os
import mlflow

@HOOKS.register_module()
class MLFlowONNXHook(Hook):
    def __init__(self, interval=16000, artifact_name="model"):
        self.interval = interval
        self.artifact_name = artifact_name
        self.run_id = None
        self.version = 0

    def before_run(self, runner):
        # Get the existing MLFlow run ID
        self.run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        if self.run_id is None:
            raise RuntimeError("MLFlow run not initialized. Please call mlflow.start_run() before using this hook.")

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.every_n_iters(runner, self.interval):
            self.upload_checkpoint(runner)

    def upload_checkpoint(self, runner):
        filename = f'iter_{runner.iter + 1}.pth'
        print(filename)
        filepath = os.path.join(runner.work_dir, filename)
        print(filepath)
        runner.save_checkpoint(runner.work_dir, filename=filename)
        
        # Log the checkpoint file as an artifact in MLFlow
        mlflow.log_artifact(filepath, artifact_path=self.artifact_name)
        
        print(f"Checkpoint {filename} uploaded to MLFlow as artifact {self.artifact_name}")
        
        # Increment version for next checkpoint
        self.version += 1

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False