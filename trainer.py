import os
import logging
import torch
import torch.nn.functional as F

from utils import metrics

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, optimizer, train_dataloader, valid_dataloader, args):
    best_f1 = 0
    logger.info('Start Training!')
    for epoch in range(1, args.epochs+1):
        model.train()
        for step, (x, y) in enumerate(train_dataloader):
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            loss = F.cross_entropy(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % 200 == 0:
                logger.info(f'|EPOCHS| {epoch:>}/{args.epochs} |STEP| {step+1:>4}/{len(train_dataloader)} |LOSS| {loss.item():>.4f}')

        avg_loss, accuracy, _, _, f1, _ = evaluate(model, valid_dataloader, args)
        logger.info('-'*50)
        logger.info(f'|* VALID SET *| |VAL LOSS| {avg_loss:>.4f} |ACC| {accuracy:>.4f} |F1| {f1:>.4f}')
        logger.info('-'*50)

        if f1 > best_f1:
            best_f1 = f1
            logger.info(f'Saving best model... F1 score is {best_f1:>.4f}')
            if not os.path.isdir(args.model_save_path):
                os.mkdir(args.model_save_path)
            torch.save(model.state_dict(), os.path.join(args.model_save_path, "best.pt"))
            logger.info('Model saved!')


def evaluate(model, valid_dataloader, args):
    with torch.no_grad():
        model.eval()
        losses, correct = 0, 0
        y_hats, targets = [], []
        for x, y in valid_dataloader:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            losses += loss.item()

            y_hat = torch.max(pred, 1)[1]
            y_hats += y_hat.tolist()
            targets += y.tolist()
            correct += (y_hat == y).sum().item()

    avg_loss, accuracy, precision, recall, f1, cm = metrics(valid_dataloader, losses, correct, y_hats, targets)
    return avg_loss, accuracy, precision, recall, f1, cm