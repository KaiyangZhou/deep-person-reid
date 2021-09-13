import torchreid
import os


def check_unique_path(path: str) -> str:
    if os.path.exists(path):
        ct: int = 2
        new_dir = path + str(ct)
        while os.path.exists(new_dir):
            ct += 1
            new_dir = path + "-" + str(ct)
        path = new_dir
    return path


def build_engine(datamanager: torchreid.data.ImageDataManager, params: dict, cuda: bool):

    model = torchreid.models.build_model(
        name=params["model_type"],
        num_classes=params["num_classes"],
        loss=params["loss_type"],
        pretrained=True
    )

    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=params["optimizer_name"],
        lr=params["optimizer_lr"]
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler=params["scheduler_name"],
        stepsize=params["scheduler_stepsize"]
    )

    engine: torchreid.engine.ImageTripletEngine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    return engine, model, optimizer
