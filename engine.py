import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                device: torch.device=device):

    model = model.to(device)
    model.train()

    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
            test_dataloader:torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            device: torch.device=device):

    model.eval()
    test_loss, test_acc = 0, 0
    model = model.to(device)
    with torch.inference_mode():

        for batch, (X, y) in enumerate(test_dataloader):
            
            X, y =  X.to(device), y.to(device)
            
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(y_pred, dim=1)
            test_acc += ((y_pred_class == y).sum().item() / len(y_pred_class))
        
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)

        return test_loss, test_acc

def train(model: torch.nn.Module,
            train_dataloader:torch.utils.data.DataLoader,
            test_dataloader:torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            writer: torch.utils.tensorboard.SummaryWriter,
            device: torch.device=device):
    
        results = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }
                
        for epoch in tqdm(range(epochs)):

            train_loss, train_acc = train_step(model=model,
                                                train_dataloader=train_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                device=device)

            test_loss, test_acc = test_step(model=model,
                                    test_dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            print(f"Epoch {epoch+1} from {epochs}:\n------------")
            print(f"Train Loss: {train_loss:.3f} | "
                  f"Train Acc: %{train_acc*100:.2f} | "
                  f"Test Loss: {test_loss:.3f} | "
                  f"Test Acc: %{test_acc*100:.2f}")

            if writer:
                writer.add_scalars(main_tag="Loss",
                                    tag_scalar_dict={"train_loss": train_loss,
                                                        "test_loss": test_loss},
                                    global_step=epoch)

                writer.add_scalars(main_tag="Accuracy",
                                    tag_scalar_dict={"train_acc": train_acc,
                                                        "test_acc": test_acc},
                                    global_step=epoch)
                writer.close()
        return results
