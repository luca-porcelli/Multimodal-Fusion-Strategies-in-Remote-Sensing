import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape10(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder10.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )
            
    def check_input_shape20(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder20.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )
            
    def check_input_shape60(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder60.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x10, x20, x60):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape10(x10)
        self.check_input_shape20(x20)
        self.check_input_shape60(x60)

        features10 = self.encoder10(x10)
        features20 = self.encoder20(x20)
        features60 = self.encoder60(x60)
        
        features=[]
        features.append(torch.cat([features10[0], features20[0], features60[0]], dim=1))
        features.append(torch.cat([features10[1], features20[1], features60[1]], dim=1))
        features.append(torch.cat([features10[2], features20[2], features60[2]], dim=1))
        features.append(torch.cat([features10[3], features20[3], features60[3]], dim=1))
        features.append(torch.cat([features10[4], features20[4], features60[4]], dim=1))
        features.append(torch.cat([features10[5], features20[5], features60[5]], dim=1))

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features10[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x10, x20, x60):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x10, x20, x60)

        return x
