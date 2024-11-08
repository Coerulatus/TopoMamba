import torch

def CustomTest(model, loader, logger):
    model.train()
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(model.device)
        with torch.no_grad():
            model.test_step(batch, batch_idx)
    results = model.on_test_epoch_end()
    for key in results:
        logger.log({"test_"+key: results[key]})