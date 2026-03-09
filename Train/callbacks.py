import time
from transformers import TrainerCallback


class TrainLossCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            elapsed = (time.time() - self.start_time) / 60
            train_loss = logs.get("loss", None)
            learning_rate = logs.get("learning_rate", None)

            if train_loss is not None:
                print(f"Step {state.global_step} | Loss: {train_loss:.4f} | LR: {learning_rate:.2e} | Time: {elapsed:.1f}m")

            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                print(f"Eval Loss: {eval_loss:.4f}")


class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_at_step=1500):
        self.eval_at_step = eval_at_step
        self.has_evaled = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.eval_at_step and not self.has_evaled:
            control.should_evaluate = True
            self.has_evaled = True
            print(f"Evaluation at step {self.eval_at_step}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        print(f"Final evaluation at step {state.global_step}")
        return control
