import torch

from src.content_loss.scene_vae import ContentLossVAE



if __name__ == '__main__':
    path = '/src/content_loss/content_loss_model.ckpt'
    model = load_model_from_checkpoint(path)
    print("Done")

