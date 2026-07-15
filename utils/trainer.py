import torch


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        prediction = model(X)

        loss = criterion(
            prediction,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            prediction = model(X)

            loss = criterion(
                prediction,
                y
            )

            running_loss += loss.item()

    return running_loss / len(loader)