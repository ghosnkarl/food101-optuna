from typing import Literal
from tqdm.auto import tqdm


class NestedProgressBar:
    """
    Manages nested tqdm progress bars for epoch and batch loops.

    train mode: outer epoch bar + inner batch bar
    eval mode:  single batch bar only
    """

    def __init__(
        self,
        total_epochs: int,
        total_batches: int,
        mode: Literal["train", "eval"] = "train",
        epoch_message_freq: int | None = None,
        batch_message_freq: int | None = None,
    ) -> None:
        self.mode = mode
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq
        self.last_batch_step: int = -1

        if mode == "train":
            self.epoch_bar = tqdm(total=total_epochs, desc="Epoch", position=0, leave=True)
            self.batch_bar = tqdm(total=total_batches, desc="Batch", position=1, leave=False)
        else:
            self.epoch_bar = None
            self.batch_bar = tqdm(total=total_batches, desc="Evaluating", position=0, leave=False)

    def begin_epoch(self) -> None:
        """Reset the batch bar for the new epoch. Does NOT advance or relabel the epoch bar."""
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_epoch(self, epoch: int, postfix_dict: dict | None = None) -> None:
        """Advance the epoch bar by 1 and update its description and postfix."""
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)
            self.epoch_bar.set_description(f"Epoch {epoch}/{self.total_epochs}")
            if postfix_dict:
                self.epoch_bar.set_postfix(postfix_dict)

    def update_batch(self, batch: int, postfix_dict: dict | None = None) -> None:
        """Advance the batch bar to the current batch index."""
        step = batch - self.last_batch_step
        if step > 0:
            self.batch_bar.update(step)
            self.last_batch_step = batch
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch: int, message: str) -> None:
        """Print message every `epoch_message_freq` epochs."""
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            tqdm.write(message)

    def maybe_log_batch(self, batch: int, message: str) -> None:
        """Print message every `batch_message_freq` batches."""
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            tqdm.write(message)

    def start_validation(self, total_val_batches: int) -> None:
        """Switch the batch bar to validation mode with a new total."""
        self.batch_bar.reset(total=total_val_batches)
        self.batch_bar.set_description("Validating")
        self.last_batch_step = -1

    def end_validation(self) -> None:
        """Switch the batch bar back to training mode."""
        self.batch_bar.reset(total=self.total_batches)
        self.batch_bar.set_description("Batch")
        self.last_batch_step = -1

    def close(self, last_message: str | None = None) -> None:
        """Close all bars and optionally print a final message."""
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        self.batch_bar.close()
        if last_message:
            print(last_message)
