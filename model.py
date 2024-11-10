import lightning as L


class ViT(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()


    def training_step(self, batch, batch_idx):
        pass


    def configure_optimizers(self):
        pass
