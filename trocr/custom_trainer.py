from transformers import Seq2SeqTrainer

class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "num_items_in_batch" in inputs:
            inputs.pop("num_items_in_batch")

        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
